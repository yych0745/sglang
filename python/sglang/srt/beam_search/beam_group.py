# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-request beam search state: frontier, completed pool, lifecycle.

BeamGroup holds search state and consumes joint_select results. Members are
not requests: the group tracks them columnarly as req_to_token rows
(member_rows) that decode in lockstep with the leader row.

advance_*_frontier / commit_pending are this class's halves of the overlap
split documented in coordinator.py.
"""

from __future__ import annotations

import enum
import heapq
import math
from typing import List, Optional, Sequence, Union

import msgspec
import torch

from sglang.srt.beam_search.fork import StagedOrphans
from sglang.srt.beam_search.history import BeamNode, materialize_tokens
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult
from sglang.srt.beam_search.penalty_state import make_beam_penalty_state
from sglang.srt.beam_search.trie import BeamTrie


class BeamGroupState(enum.Enum):
    DECODING = enum.auto()
    FINISHED = enum.auto()


class CompletedBeam(msgspec.Struct):
    """A finished candidate: leaf node + score inputs + finish reason."""

    leaf: Optional[BeamNode]
    cum_logprob: float
    num_tokens: int
    matched_token: Optional[int]  # stop token that ended it; None = length / cutoff


class BeamResult(msgspec.Struct):
    """One output sequence of a finished group."""

    tokens: List[int]
    cum_logprob: float
    beam_score: float
    matched_token: Optional[int]


# Beam-search early-stopping modes. Ordered "off -> eager -> heuristic".
EARLY_STOPPING_MODES = ("off", "false", "true", "never")


def normalize_early_stopping(value: Union[str, bool, None]) -> str:
    """Normalize an ``early_stopping`` value to one of EARLY_STOPPING_MODES.

    Booleans are the historical wire form and are mapped for backward
    compatibility: ``True -> "true"``, ``False -> "off"``. Anything else --
    including a None that means "not given" -- raises a ValueError carrying the
    accepted set, so a typo fails at request construction instead of silently
    degrading to one of the modes.
    """
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "true" if value else "off"
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in EARLY_STOPPING_MODES:
            return mode
    raise ValueError(
        "early_stopping must be one of "
        f"{', '.join(repr(m) for m in EARLY_STOPPING_MODES)} "
        f"(a bool is also accepted: True -> 'true', False -> 'off'), "
        f"got {value!r}."
    )


class BeamGroup:
    """State machine: one prefill selection, then decode selections, then
    finalize. The frontier starts as one pseudo-row (the prompt, cum_logprob 0)."""

    def __init__(
        self,
        *,
        beam_width: int,
        length_penalty: float = 1.0,
        stop_token_ids: Sequence[int] = (),
        max_new_tokens: int,
        num_return: Optional[int] = None,
        # Early-stopping mode, one of EARLY_STOPPING_MODES:
        #   "off"   never stop early; decode to max_new_tokens.
        #   "true"  stop as soon as beam_width finished beams are collected
        #           (transformers `early_stopping=True`).
        #   "false" stop when the best live beam cannot beat the worst returned
        #           candidate *at the length it has right now*.
        #   "never" same, but with the bound taken at max_new_tokens when
        #           length_penalty > 0, so no beam that could still win is cut.
        #
        # BREAKING CHANGE: this used to be a bool defaulting to True (==
        # "true"), i.e. beam search stopped as soon as the completed pool
        # filled. The default is now "off" -- a request runs to
        # max_new_tokens unless it opts in. Booleans are still accepted and
        # mapped (True -> "true", False -> "off") so existing callers keep
        # compiling; their meaning is unchanged.
        early_stopping: Union[str, bool] = "off",
        device: torch.device | str = "cpu",
        trie: Optional[BeamTrie] = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
    ):
        self.beam_width = beam_width
        self.num_candidates = 2 * beam_width
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.num_return = num_return if num_return is not None else beam_width
        self.early_stopping = normalize_early_stopping(early_stopping)
        self.stop_token_ids = torch.tensor(
            sorted(stop_token_ids), dtype=torch.int64, device=device
        )

        self.frontier_cum_logprobs = torch.zeros(1, dtype=torch.float32, device=device)
        # Constrained decoding: the trie and this group's position in it. The
        # cursor is launch-half state exactly like frontier_cum_logprobs --
        # same tensor family, same lifetime, advanced in the same call -- so the
        # two can never disagree. Deliberately NOT on BeamNode: leaves are
        # retained by `completed` long after the frontier moves on, and dummy
        # padding aliases one leaf object k times.
        self.trie = trie
        self.trie_cursor: Optional[torch.Tensor] = (
            None if trie is None else trie.root_cursor(1)
        )
        self.leaves: List[Optional[BeamNode]] = [None]  # parents of the next tokens
        # Repetition penalties (frequency / presence / repetition), read off the
        # user's sampling_params before the leader's row params were neutralized.
        # None at the no-op defaults, which keeps the selection path -- and its
        # device footprint -- exactly as it was. A pure host object until the
        # first selection: the count table is allocated lazily, on the forward
        # stream, sized from the logits being scored.
        self.penalty_state = make_beam_penalty_state(
            frequency_penalty, presence_penalty, repetition_penalty
        )
        # num_generated is the launch half's count, num_committed the deferred
        # half's (the true length); generated may lead by one under overlap.
        self.num_generated = 0
        self.num_committed = 0
        self.completed: List[CompletedBeam] = []
        self.state = BeamGroupState.DECODING
        # Selection results staged by the launch half as (forward tick, sel),
        # consumed in tick order by commit.
        self._pending_steps: List[tuple] = []
        # Highest forward tick whose selection was folded into the DAG. Under
        # overlap the launch half can run one step past the group's terminal
        # commit; that step is discarded, so its tick stays above this mark.
        self.last_committed_tick = -1
        # Set once by the coordinator when the group leaves the live set
        # (finish / abort / dead-leader); guards double bookkeeping.
        self.retired = False

        # Scheduler wiring, filled in from outside the search core. Members have
        # no Req, so their seq len and KV lengths are implied by the leader's.
        self.leader = None
        # Device [k-1] member row indices, and the same rows on host (for
        # row-slot free). None until the post-prefill spawn / after free.
        self.member_rows: Optional[torch.Tensor] = None
        self.member_rows_cpu: Optional[torch.Tensor] = None
        # Device [k]: leader row first, then member_rows (frontier-row order).
        self.all_rows: Optional[torch.Tensor] = None
        # The same k row numbers on the host, in the same order. Kept past the
        # member rows' release so the retire path can withdraw them from a mamba
        # COW snapshot the group armed and never got to consume.
        self.all_rows_cpu: Optional[List[int]] = None
        # Staged by the launch half; the deferred half frees them, gated on the
        # tick whose copy_done sync already happened.
        self.pending_orphans: List[StagedOrphans] = []
        # Running total the GC has returned, so held KV is a host-side
        # arithmetic (allocated - freed) rather than a tensor read.
        self.slots_freed = 0

    @property
    def num_member_rows(self) -> int:
        return 0 if self.member_rows is None else self.member_rows.shape[0]

    def extra_uncached_tokens(self) -> int:
        """Uncached KV the group holds beyond the leader's own window, which the
        generic per-req sum already counts."""
        if self.all_rows is None:
            return 0
        end = self.leader.kv.kv_allocated_len
        start = self.prompt_len
        # Host-side arithmetic, not distinct slots off req_to_token: that would
        # read the launch half's staged tensors, unsafe on the checker's stream.
        held = self.beam_width * (end - start) - self.slots_freed
        return held - (end - start)

    def next_step_is_final(self) -> bool:
        """The upcoming selection hits max_new_tokens (decided host-side)."""
        return self.num_generated + 1 >= self.max_new_tokens

    def advance_frontier(self, sel: SelectResult, tick: int = 0) -> None:
        """Launch half of one selection step: evolve the frontier tensor and
        stage the result for commit, stamped with its forward tick."""
        assert self.state == BeamGroupState.DECODING
        if self.trie is not None:
            # Pure device walk (segment scan inside advance(), no D2H), so this
            # stays legal on the launch path. Kept in the same call as the
            # frontier update because both are launch-half state for one tick.
            #
            # Every row advances, including dummy seats. A dummy's cum_logprob
            # is -inf, so joint_select can never promote it and no candidate of
            # its row is ever examined: whatever node its cursor names is inert.
            # Advancing it unconditionally (rather than pinning it) keeps this a
            # single masked free gather with no data-dependent row selection,
            # and its parent_idx always names a real row, so the walk stays in
            # bounds. A dead node (INVALID_NODE) is likewise harmless: mask
            # releases EOS only, and that row's score is -inf regardless.
            self.trie_cursor = self.trie.advance(
                self.trie_cursor, sel.parent_idx, sel.next_tokens
            )
        if self.penalty_state is not None:
            # Same launch-half state as the cursor and the frontier, advanced in
            # the same call so the three can never disagree. Every row moves,
            # dummy seats included, exactly like the trie cursor: a dummy's cum
            # is -inf so it can never win a later selection, and advancing it
            # keeps this a single gather with no host-side survivor count.
            #
            # Advanced here rather than in the coordinator because prefill is
            # the step that grows the table from one prompt row to k: the same
            # call covers prefill and decode, so there is no second place that
            # could advance it twice.
            self.penalty_state.advance(sel.parent_idx, sel.next_tokens)
        self.frontier_cum_logprobs = sel.new_cum_logprobs
        self.num_generated += 1
        self._pending_steps.append((tick, sel))

    def advance_final_frontier(self, sel: FinalSelect, tick: int = 0) -> None:
        """Launch half of a length-terminated step: stage only (the final step
        needs no next frontier).

        The penalty state is deliberately not advanced either: this step's
        tokens are never scored again, so counting them would be pure cost.
        """
        assert self.state == BeamGroupState.DECODING
        self.num_generated += 1
        self._pending_steps.append((tick, sel))

    def commit_pending(self, up_to_tick: Optional[int] = None) -> bool:
        """Deferred half: consume staged selections into the DAG (the D2H sync
        point). Returns True when this commit finishes the group."""
        if self.state == BeamGroupState.FINISHED:
            self._pending_steps.clear()
            return False
        while self._pending_steps:
            tick, sel = self._pending_steps[0]
            # Tick gate: copy_done covers only kernels enqueued up to that
            # forward, so a later-staged step may not be readable yet.
            if up_to_tick is not None and tick > up_to_tick:
                break
            self._pending_steps.pop(0)
            # Highest tick whose selection actually reached the DAG. Launched
            # steps beyond it are discarded (overlap overshoot) and their staged
            # KV side effects must not be treated as committed history.
            self.last_committed_tick = tick
            finished = (
                self._commit_final(sel)
                if isinstance(sel, FinalSelect)
                else self._commit_step(sel)
            )
            if finished:
                self._pending_steps.clear()
                return True
        return False

    def _commit_step(self, sel: SelectResult) -> bool:
        num_survivors = int(sel.num_survivors)
        num_finished = int(sel.num_finished)
        new_len = self.num_committed + 1

        fin_tokens = sel.fin_tokens[:num_finished].tolist()
        fin_parents = sel.fin_parent_idx[:num_finished].tolist()
        fin_cums = sel.fin_cum_logprobs[:num_finished].tolist()
        for token, parent, cum in zip(fin_tokens, fin_parents, fin_cums):
            leaf = BeamNode(token, self.leaves[parent])
            self.completed.append(
                CompletedBeam(leaf, cum, new_len, matched_token=token)
            )

        surv_tokens = sel.next_tokens[:num_survivors].tolist()
        surv_parents = sel.parent_idx[:num_survivors].tolist()
        real_leaves = [
            BeamNode(token, self.leaves[parent])
            for token, parent in zip(surv_tokens, surv_parents)
        ]
        self.num_committed = new_len

        if num_survivors == 0:
            # Nothing legal remains: every examined candidate was a stop token or
            # masked out. The completed pool is the answer.
            self.leaves = []
            self.state = BeamGroupState.FINISHED
            return True

        if num_survivors < self.beam_width:
            # Short step: keep the batch shape fixed by seating dummy leaves
            # instead of finishing the group. A dummy aliases the best real leaf
            # so its parent row always resolves, and it is carried at -inf by
            # joint_select, so it can never win a later selection nor be
            # committed as a result. This is what constrained decoding needs:
            # a trie layer with fewer than beam_width legal children must not end
            # the request.
            self.leaves = real_leaves + [real_leaves[0]] * (
                self.beam_width - num_survivors
            )
        else:
            self.leaves = real_leaves

        if self._should_early_stop(sel, num_survivors, new_len):
            # The live beams are DROPPED, not folded into `completed`.
            #
            # Early stopping means "the finished candidates we already hold are
            # good enough that no continuation can improve the returned set", so
            # publishing an unfinished beam here would put a truncated sequence
            # into a result set whose whole point is that every entry finished
            # on its own terms. Length truncation is the opposite: that path
            # (`_commit_final`) has no choice -- the budget ran out -- and it
            # does fold its survivors in, which is why the two must not share a
            # helper.
            #
            # Runs on the deferred (CPU) half; under overlap the launch half has
            # already issued one more forward, so one extra step may still be
            # computed before we stop.
            self.leaves = []
            self.state = BeamGroupState.FINISHED
            return True
        return False

    def _completed_beam_scores(self) -> List[float]:
        """Length-normalized scores of the completed pool, best-effort entries
        excluded.

        The internal implementation skips pool entries whose ``beam_score is
        None``; our CompletedBeam always carries a cum_logprob, so the analogue
        of "no score" is a score that is not a finite number -- a padding
        artifact carried in at -inf. Those must not fill a slot in the "worst
        returned candidate" threshold.

        Uses `beam_score`, the same scoring function `finalize` ranks with, so
        an early stop can never be decided against a different length penalty
        than the one that orders the results.
        """
        scores = []
        for beam in self.completed:
            score = self.beam_score(beam.cum_logprob, beam.num_tokens)
            if not math.isfinite(score):
                continue
            scores.append(score)
        return scores

    def _should_early_stop(self, sel: SelectResult, num_survivors: int, hyp_len: int) -> bool:
        """Heuristic early-stop judge; mirrors the internal scheduler's
        `_should_early_stop_beam_search`.

        With `length_penalty > 0` the score `cum_logprob / len ** lp` *rises*
        with length, so a search can keep going to max_new_tokens (repetition
        loops do exactly that). Stop once the best live beam cannot beat the
        worst candidate that will actually be returned:

            best_possible  = beam_score(best_live_cum, bound_len)
            worst_finished = the beam_width-th highest completed score
            stop           iff  best_possible <= worst_finished

        `bound_len` is what separates "false" from "never": "false" scores the
        live beam at the length it has *now* (heuristic, may truncate a beam
        that would have kept improving), while "never" -- for lp > 0 -- scores
        it at max_new_tokens, the longest it can ever get, which makes
        `best_possible` a true upper bound and the stop lossless. For lp <= 0
        longer is worse, so the current length already is the bound and "never"
        degenerates to "false".

        Args:
            sel: this step's selection (read only after every cheap check has
                passed -- see the note on ordering below).
            num_survivors: live beams in `sel` (guaranteed >= 1 here: a
                zero-survivor step has already finished the group).
            hyp_len: generated length *after* this step's expansion.

        The checks are deliberately ordered cheapest-first, so the one tensor
        read happens only for the two heuristic modes on a group that already
        has a full completed pool. That keeps the ordinary decode path free of
        an extra D2H: "off" and "true" never touch the device.

        `hyp_len < max_new_tokens` (the internal version's first precondition)
        needs no check here: the launch half routes the step that reaches
        max_new_tokens to `select_final_topk` (`next_step_is_final`), i.e.
        `_commit_final`, so a step reaching `_commit_step` always has
        num_committed + 1 <= num_generated + 1 < max_new_tokens.
        """
        if len(self.completed) < self.beam_width:
            # Unfinished beams can still fill the empty slots, so no mode stops.
            return False

        mode = self.early_stopping
        if mode == "off":
            return False
        if mode == "true":
            # transformers `early_stopping=True`: every returned beam finished.
            return True

        finished_scores = self._completed_beam_scores()
        if len(finished_scores) < self.beam_width:
            return False
        # Only the top beam_width candidates are returned (finalize slices to
        # beam_width), so the bar to clear is the beam_width-th best.
        worst_finished = heapq.nlargest(self.beam_width, finished_scores)[-1]

        if mode == "never" and self.length_penalty > 0.0:
            bound_len = max(self.max_new_tokens, hyp_len)
        else:
            bound_len = hyp_len

        # The only device read, and it is last: one max on each side, combined
        # on the device, then a single scalar sync.
        #
        # `best_cum_logprob` is the best cumulative logprob over *every*
        # candidate this step expanded -- the ones that hit a stop token
        # included. That is what the internal scheduler feeds its judge
        # (`float(expanded_cum_logprobs[0])`, the head of the descending topk
        # of the whole step), and the point of this port is to stop at the same
        # moment as that version. A survivor-only bound is tighter and
        # defensible on its own terms -- the live beams are all that can still
        # be extended, so their best *is* the best achievable continuation --
        # but it fires strictly earlier and therefore returns different
        # results. Do not "fix" this back to survivor-only without an explicit
        # decision to diverge from the internal version.
        #
        # Survivors plus finished cover the finite candidate set jointly, so no
        # third source is needed: joint_select makes a candidate a survivor iff
        # it is finite, not a stop token, and fewer than beam_width usable
        # candidates precede it; it makes one finished iff it is finite, a stop
        # token, and fewer than beam_width usable candidates precede it. A
        # finite candidate outranked by beam_width usable ones lands in neither
        # set, but it is then not the step's best either. -inf candidates
        # (dummy rows, constraint-masked branches) land in neither set, and
        # both tensors pad with -inf, so a plain max over the padded tensors
        # already ignores them -- no separate isfinite pass, and -inf can only
        # win if there is nothing finite at all, which a zero-survivor step
        # has already returned on.
        best_cum_logprob = float(
            torch.maximum(
                sel.new_cum_logprobs[:num_survivors].max(),
                sel.fin_cum_logprobs.max(),
            )
        )
        best_possible = self.beam_score(best_cum_logprob, max(bound_len, 1))
        return best_possible <= worst_finished

    def _commit_final(self, sel: FinalSelect) -> bool:
        new_len = self.num_committed + 1
        tokens = sel.tokens.tolist()
        parents = sel.parent_idx.tolist()
        cums = sel.cum_logprobs.tolist()
        # A parent outside the committed frontier means a tick-gating bug let an
        # unsynchronized step through; fail rather than build a corrupt DAG.
        assert not parents or max(parents) < len(self.leaves), (
            "beam commit consumed an unsynced or misordered step"
        )
        for token, parent, cum in zip(tokens, parents, cums):
            # A -inf cum means this candidate extends a dummy leaf: it is a
            # padding artifact, not a real sequence, so it must not reach the
            # returned pool.
            if cum == float("-inf"):
                continue
            leaf = BeamNode(token, self.leaves[parent])
            self.completed.append(CompletedBeam(leaf, cum, new_len, matched_token=None))
        self.leaves = []
        self.num_committed = new_len
        self.state = BeamGroupState.FINISHED
        return True

    # Sync wrappers: both halves back-to-back. UT-only -- the scheduler path
    # drives the two halves separately so they can straddle a forward.

    def advance(self, sel: SelectResult) -> bool:
        """Consume one joint_select result; returns True if the group finished."""
        self.advance_frontier(sel)
        return self.commit_pending()

    def advance_final(self, sel: FinalSelect) -> bool:
        """Consume a length-terminated select_final_topk result; always finishes."""
        self.advance_final_frontier(sel)
        return self.commit_pending()

    def beam_score(self, cum_logprob: float, num_tokens: int) -> float:
        """Length-normalized score: cum_logprob / num_tokens ** length_penalty."""
        return cum_logprob / (num_tokens**self.length_penalty)

    def finalize(self) -> List[BeamResult]:
        """Materialize the top beam_width sequences, best score first."""
        assert self.state == BeamGroupState.FINISHED
        results = [
            BeamResult(
                tokens=materialize_tokens(beam.leaf),
                cum_logprob=beam.cum_logprob,
                beam_score=self.beam_score(beam.cum_logprob, beam.num_tokens),
                matched_token=beam.matched_token,
            )
            for beam in self.completed
        ]
        results.sort(key=lambda r: r.beam_score, reverse=True)
        return results[: self.beam_width]
