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
"""Scheduler wiring for beam search (columnar member-row architecture).

A beam_width=k request runs as one leader Req plus k-1 bare member rows:
physical req_to_token rows tracked columnarly on the group, with no Req
object behind them. The member rows are appended to the decode batch's row
tensors just before allocation (batch_tail.append_beam_tail) and sliced
off the logits before sampling (tp_worker), so the reqs-aligned world never
sees them. Hooks: admission (validate_and_init), selection at the forward's
relay point, and lifecycle (member spawn at the leader's prefill relay,
finalize at group finish).

All rows decode in lockstep over [prompt_len, leader_allocated), but
share-on-fork lets several of them reference the same slot, so the group --
not the individual row -- owns that region: it is freed once, deduped, at
group finish.

Every tick splits in two, and the whole file follows the naming. select_* is
the launch half (tensor-side, no D2H) and must run before the next forward
resolves its inputs, so the relayed tokens and reparented KV are the selected
ones. commit_* is the deferred half (DAG build, finish/abort); under overlap it
lags one forward and discards overshoot steps. Sync callers run both within one
tick. Commits are tick-gated; BeamGroup.commit_pending documents why.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Sequence

import msgspec
import torch

from sglang.srt.beam_search.beam_group import BeamGroup, BeamGroupState
from sglang.srt.beam_search.fork import (
    MEMBER_LENGTH_MARGIN,
    StagedOrphans,
    alias_members_prompt_kv,
    collect_orphan_slots,
    free_member_rows,
    neutral_member_sampling_params,
    remap_kv_mapping,
)
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.beam_search.penalty_state import BeamPenaltyState
from sglang.srt.beam_search.trie import BeamTrie
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    get_schedule,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def _rows_topk_logprobs(
    pieces: Sequence[torch.Tensor],
    num_candidates: int,
    mask: Optional[torch.Tensor] = None,
    penalty_state: Optional["BeamPenaltyState"] = None,
):
    """Top-`num_candidates` log-probs per row, over the concatenated pieces.

    `mask` (True = forbidden) is given in concatenated row order, i.e. aligned
    with `torch.cat(pieces)`; empty pieces contribute no rows, so they must not
    advance the row cursor.

    `penalty_state` (None at the no-op defaults) applies the group's
    frequency / presence / repetition penalties to RAW logits, before the
    logsumexp, over each piece's own rows -- see penalty_state.py for the exact
    order and for why the prompt is not counted. It is handed the piece's row
    offset instead of a concatenated tensor so every piece keeps its own lse.
    """
    # One logsumexp per piece: it is not batch-invariant on CUDA, and folding
    # pieces shifts the leader's lse by an ulp, flipping near-equal beams.
    vals, toks = [], []
    row = 0
    for piece in pieces:
        if piece.shape[0] == 0:
            continue
        x = piece.float()
        if penalty_state is not None:
            # Penalties before the lse, on this piece's rows only.
            x = penalty_state.apply(x, row_start=row)
        # lse instead of a full [rows, vocab] log_softmax; topk order on raw
        # logits is identical under the monotone shift.
        lse = torch.logsumexp(x, dim=-1, keepdim=True)
        if mask is not None:
            # Mask AFTER the lse so the normalizer stays the model's own
            # distribution: scores remain comparable across steps and across
            # constrained/unconstrained groups. Masking first would renormalize
            # over the allowed set and inflate every surviving candidate.
            piece_mask = mask[row : row + piece.shape[0]]
            assert piece_mask.shape[0] == piece.shape[0], (
                f"trie mask has {mask.shape[0]} rows but the pieces carry "
                f"more: row offset {row} + piece rows {piece.shape[0]}"
            )
            x = x.masked_fill(piece_mask, float("-inf"))
        v, t = torch.topk(x, num_candidates, dim=-1)
        vals.append(v - lse)
        toks.append(t)
        row += piece.shape[0]
    return torch.cat(vals), torch.cat(toks)


class BeamCoordinator(msgspec.Struct, kw_only=True):
    model_config: ModelConfig
    spec_algorithm: SpeculativeAlgorithm
    dllm_enabled: bool
    max_req_len: int
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    future_map: FutureMap
    # Constrained decoding: compiled once at scheduler init from
    # --beam-trie-path, shared by every group (read-only CSR tensors).
    trie: Optional[BeamTrie] = None
    # --enable-beam-mamba-double-buffer. Beam search on a mamba/hybrid cache is
    # refused without it: a prune cannot reparent SSM state through
    # req_to_token, so it needs the per-row copy-on-write double buffer that
    # this switch is the opt-in for.
    enable_beam_mamba_double_buffer: bool = False
    # --enable-beam-mamba-state-copy. The opt-in for the other way to move SSM
    # state across a prune: a real per-row state copy instead of the one-tick
    # read/write split. Either one is enough to admit a beam request on a mamba
    # cache; when both are set the true copy wins, because it holds no state
    # across a tick boundary and asks nothing of the kernels.
    enable_beam_mamba_state_copy: bool = False

    # Live (non-retired) groups; the O(1) gate for the per-forward relay hook.
    _num_live_groups: int = 0

    @staticmethod
    def request_beam_width(recv_req) -> int:
        """beam_width of an incoming request (1 = not a beam request)."""
        return getattr(recv_req.sampling_params, "beam_width", None) or 1

    def validate_and_init(self, req: Req, recv_req) -> Optional[str]:
        """Validate a beam request and attach its group; returns an error or
        None. On success the leader's row params are neutralized."""
        user_params = req.sampling_params
        beam_width = user_params.beam_width

        if not self.spec_algorithm.is_none():
            return "Beam search is not supported with speculative decoding."
        if get_disagg().disaggregation_mode != "null":
            return "Beam search is not supported with PD disaggregation."
        if get_schedule().page_size > 1:
            return "Beam search currently requires --page-size 1."
        if self.dllm_enabled:
            return "Beam search is not supported with diffusion LLM."
        if get_memory().enable_hisparse:
            return "Beam search is not supported with hisparse."
        if get_parallel().pp_size > 1:
            return "Beam search is not supported with pipeline parallelism."
        if is_dp_attention_enabled():
            # The cross-rank token sync reads batch_size() (== len(reqs)), which
            # excludes the member rows the forward actually runs.
            return "Beam search is not supported with dp attention."
        if get_memory().enable_hierarchical_cache:
            return "Beam search is not supported with hierarchical cache."
        if self.model_config.is_encoder_decoder:
            return "Beam search is not supported with encoder-decoder models."
        if self.tree_cache.supports_swa():
            # SWA: no reparenting story at all -- a window is a sliding range of
            # token slots, and the beam's reparent rewrites req_to_token in
            # place, so a survivor would silently lose the window it inherited.
            return "Beam search is not supported with SWA hybrid caches."
        # Either mamba path is an acceptable opt-in. `supports_mamba` is only a
        # proxy for "this cache carries SSM state" -- ChunkCache does not
        # override it, so a --disable-radix-cache hybrid run reaches here with
        # it False while the prune below still has to move state. The real
        # dispatch therefore lives in _apply_mamba_cow, on the pool.
        if self.tree_cache.supports_mamba() and not (
            self.enable_beam_mamba_double_buffer or self.enable_beam_mamba_state_copy
        ):
            return (
                "Beam search is not supported with mamba hybrid caches unless "
                "--enable-beam-mamba-double-buffer or "
                "--enable-beam-mamba-state-copy is set: a beam prune cannot "
                "reparent SSM state by rewriting req_to_token the way it "
                "reparents attention KV, so each pruned row has to be given its "
                "parent's state -- either by reading the parent's slot while "
                "writing a freshly allocated one (double buffer), or by copying "
                "that state outright (state copy)."
            )

        if req.session_id is not None or recv_req.session_params is not None:
            return "Beam search is not supported for session requests."
        if req.lora_id is not None:
            return "Beam search is not supported with LoRA."
        if recv_req.return_logprob:
            return "Beam search does not support return_logprob."
        if recv_req.return_hidden_states:
            return "Beam search does not support return_hidden_states."
        if recv_req.return_sampling_mask or recv_req.return_routed_experts:
            return "Beam search does not support sampling-mask/routed-experts returns."
        if any(
            x is not None
            for x in (
                user_params.json_schema,
                user_params.regex,
                user_params.ebnf,
                user_params.structural_tag,
            )
        ):
            return "Beam search is not supported with constrained decoding."
        if user_params.stop_strs or user_params.stop_regex_strs:
            return "Beam search does not support stop strings/regex yet; use stop_token_ids."
        if user_params.min_new_tokens > 0:
            return "Beam search does not support min_new_tokens yet."
        if user_params.n > beam_width:
            return f"n ({user_params.n}) cannot exceed beam_width ({beam_width})."
        if 2 * beam_width > self.model_config.vocab_size:
            return f"beam_width ({beam_width}) is too large for the vocabulary."
        if beam_width > self.req_to_token_pool.size:
            return (
                f"beam_width ({beam_width}) needs {beam_width} req-to-token slots "
                f"but the pool holds only {self.req_to_token_pool.size}. Reduce "
                f"beam_width or raise --max-running-requests."
            )

        # Effective generation budget: keep prompt + budget + member margin
        # within the row width so member-side length can never truncate first.
        prompt_len = len(req.origin_input_ids)
        max_new_tokens = min(
            (
                user_params.max_new_tokens
                if user_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - prompt_len - 1 - MEMBER_LENGTH_MARGIN,
        )
        if max_new_tokens < 1:
            return (
                f"Beam search needs at least 1 generated token within the "
                f"context budget (prompt_len={prompt_len}, max_req_len={self.max_req_len})."
            )

        if self.trie is not None:
            # A constrained request must be able to walk the trie to a leaf and
            # then emit the terminator: anything shorter can only ever return
            # truncated paths, which are not dictionary entries. Rejecting here
            # keeps the failure at admission instead of silently degrading.
            needed = self.trie.depth + 1
            if max_new_tokens < needed:
                return (
                    f"Trie-constrained beam search needs max_new_tokens >= "
                    f"{needed} (trie depth {self.trie.depth} + a terminator "
                    f"step), but the effective budget is {max_new_tokens}."
                )

        group = BeamGroup(
            beam_width=beam_width,
            stop_token_ids=self._collect_stop_token_ids(req, user_params),
            max_new_tokens=max_new_tokens,
            num_return=user_params.n,
            length_penalty=user_params.length_penalty,
            # Already normalized by SamplingParams to one of
            # "off" | "false" | "true" | "never" (booleans from older callers
            # are mapped there: True -> "true", False -> "off"), so the group
            # never has to guess. "off" -- the default -- means decode to
            # max_new_tokens.
            early_stopping=user_params.early_stopping,
            # Read off the USER's params, so they must be taken before the
            # leader's row params are neutralized below: from here on the
            # leader's sampler only ever sees the neutral 0 / 0 / 1.
            frequency_penalty=user_params.frequency_penalty,
            presence_penalty=user_params.presence_penalty,
            repetition_penalty=user_params.repetition_penalty,
            # Frontier state lives on device: selection consumes device top-2k
            # tensors in place; only k-sized results ever reach the host.
            device=self.req_to_token_pool.device,
            trie=self.trie,
        )
        group.leader = req
        group.prompt_len = prompt_len

        # Neutralize the leader's row params (raw log_softmax scoring, no
        # self-finish path); the user's semantics now live on the group.
        neutral = neutral_member_sampling_params(user_params)
        neutral.max_new_tokens = max_new_tokens + MEMBER_LENGTH_MARGIN
        neutral.no_stop_trim = user_params.no_stop_trim
        req.sampling_params = neutral
        req.beam_group = group
        # The leader's decode suffix is a beam path, never a tree entry; this
        # also skips the prefill-time unfinished insert.
        req.skip_radix_cache_insert = True
        self._num_live_groups += 1
        return None

    def pending_member_rows(self, batch: ScheduleBatch) -> int:
        """Rows admitted-but-not-yet-spawned groups will claim; the admission
        gate subtracts this so it never over-commits the req slot pool."""
        if self._num_live_groups == 0:
            return 0
        return sum(
            r.beam_group.beam_width - 1
            for r in batch.reqs
            if r.beam_group is not None
            and r.beam_group.member_rows is None
            and not r.beam_group.retired
            and not r.finished()
        )

    @staticmethod
    def _collect_stop_token_ids(req: Req, user_params) -> List[int]:
        if user_params.ignore_eos:
            return []
        stop_ids = set(user_params.stop_token_ids or ())
        stop_ids |= set(req.eos_token_ids or ())
        tokenizer = req.tokenizer
        if tokenizer is not None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                stop_ids.add(tokenizer.eos_token_id)
            stop_ids |= set(getattr(tokenizer, "additional_stop_token_ids", None) or ())
        return sorted(stop_ids)

    def maybe_select_and_relay(
        self, batch: ScheduleBatch, batch_result, chunked_req: Optional[Req] = None
    ) -> None:
        """Per-forward relay hook: overwrite beam rows' relayed tokens with
        joint-selected ones. O(1) when no beam group is live."""
        # Runs before the early returns on purpose: the slots a group staged
        # still have to be released even when that group has since left the
        # live set, or the last prune of a finished group leaks them.
        self._flush_beam_cow_source_slots()
        if self._num_live_groups == 0:
            return
        if not batch.spec_algorithm.is_none():
            return
        logits_output = batch_result.logits_output
        if logits_output is None or logits_output.next_token_logits is None:
            return
        if batch.forward_mode.is_decode():
            self.select_and_relay_decode(batch, logits_output)
        elif batch.forward_mode.is_extend():
            capture = logits_output.beam
            leader_pos = {
                row: pos
                for pos, row in enumerate(
                    capture.leader_rows if capture is not None else ()
                )
            }
            for i, req in enumerate(batch.reqs):
                group = req.beam_group
                if (
                    group is None
                    or group.state != BeamGroupState.DECODING
                    or group.num_generated > 0
                    or req is chunked_req  # mid-chunk leader: no selection yet
                    or req.is_retracted
                    or req.finished()
                ):
                    continue
                assert i in leader_pos, (
                    "beam leader prefill logits were not captured pre-sample "
                    "(worker capture_pre_sample_logits wiring)"
                )
                self.select_leader_prefill(
                    req, leader_pos[i], logits_output, tick=batch.forward_iter
                )

    def _flush_beam_cow_source_slots(self) -> None:
        """Hand last relay's detached mamba slots back to the allocator.

        Mounted here, at the top of the per-forward relay hook, rather than at
        the top of a forward: the slots staged by relay N are read by forward
        N+1 (they are the parents' old slots, and the rows now write fresh
        ones), so they may only be released once that forward has been
        launched -- which is exactly the point this hook runs at. Releasing
        them at the top of forward N+1 instead would put them back in the free
        list while the kernel is still about to read them.

        One call per forward, and a no-op on a pool without the double buffer.
        """
        flush = getattr(self.req_to_token_pool, "flush_beam_cow_source_slots", None)
        if flush is not None:
            flush()

    def select_leader_prefill(
        self,
        req: Req,
        pos: int,
        logits_output: LogitsProcessorOutput,
        tick: int = 0,
    ) -> None:
        """Launch half of the leader's prefill tick: first selection, member-row
        spawn, and the relay overwrite (the sampled token is void)."""
        group: BeamGroup = req.beam_group
        top_logprobs, top_tokens = _rows_topk_logprobs(
            [logits_output.beam.leader_logits[pos : pos + 1]],
            group.num_candidates,
            # One leader row: its cursor is the whole mask. Built before
            # _select_group, which advances the cursor for the next tick.
            self._trie_mask(group),
            # Counts are all zero at prefill, so this step is scored raw.
            penalty_state=group.penalty_state,
        )
        final = group.next_step_is_final()
        next_tokens, _ = self._select_group(group, top_logprobs, top_tokens, tick)
        if final:
            # Prefill-terminated (max_new_tokens == 1): no spawn, but the relay
            # slot still needs a token so an overshoot step has a valid input.
            self._stash_next_tokens([req.kv.req_pool_idx], next_tokens[:1])
            return
        self._spawn_member_rows(group, req)
        req.output_ids.append(0)  # length placeholder; DAG owns history
        self._stash_next_tokens(group.all_rows, next_tokens)

    def commit_prefill(self, req: Req, up_to_tick: Optional[int] = None) -> None:
        """Deferred half of the leader's prefill tick: fold the staged selection
        into the DAG (the designated D2H point) and apply finish/abort."""
        group: BeamGroup = req.beam_group
        if group.retired:
            return
        if req.to_finish is not None:
            self._abort_group(group)
            return
        self._reclaim_orphans(group, up_to_tick)
        if group.commit_pending(up_to_tick):
            self._finish_group(group)

    def _spawn_member_rows(self, group: BeamGroup, leader: Req) -> None:
        rows = self.req_to_token_pool.alloc_rows(group.beam_width - 1)
        assert rows is not None, (
            f"Beam member spawn needs {group.beam_width - 1} req-to-token slots "
            f"but only {self.req_to_token_pool.available_size()} are free; the "
            f"admission gate (get_num_allocatable_reqs) must reserve them."
        )
        device = self.req_to_token_pool.device
        member_rows = torch.tensor(rows, dtype=torch.int64, device=device)
        alias_members_prompt_kv(
            self.req_to_token_pool.req_to_token,
            member_rows,
            leader.kv.req_pool_idx,
            group.prompt_len,
        )
        group.member_rows = member_rows
        group.member_rows_cpu = torch.tensor(rows, dtype=torch.int64)
        leader_row = torch.tensor(
            [leader.kv.req_pool_idx], dtype=torch.int64, device=device
        )
        group.all_rows = torch.cat([leader_row, member_rows])
        # Host mirror, same order as `all_rows` (leader first). It outlives the
        # member rows: fork.free_member_rows nulls all_rows / member_rows_cpu on
        # release, and the retire path still needs these numbers to withdraw the
        # group from a mamba COW snapshot it armed but never consumed.
        group.all_rows_cpu = [int(leader.kv.req_pool_idx), *rows]
        # Member rows are bare rows: alloc() never ran for them, so on a hybrid
        # pool their row -> mamba-slot entry is still 0 (the reserved padding
        # slot) and all k-1 members would read and write it. Own one slot each.
        #
        # Ownership alone is not enough: what a slot *holds* is whatever the
        # allocator last issued it with (zeros on a fresh server, a stranger's
        # state once slots recycle). A member is a fork of the leader at
        # prompt_len, so its slot has to start from the state the leader's
        # prefill just wrote -- otherwise its very first decode tick reads an
        # SSM state that has never seen the prompt, and the beam decodes as if
        # it started from a different prefix than it did.
        alloc_mamba = getattr(self.req_to_token_pool, "alloc_mamba_for_rows", None)
        if alloc_mamba is not None and alloc_mamba(member_rows) is None:
            # Hand the req rows back before failing. They are already out of
            # free_slots at this point, and the admission gate does not yet
            # reserve mamba slots for members, so this is reachable on a full
            # pool -- leaking k-1 rows per attempt would wedge the pool.
            self.req_to_token_pool.free_rows(rows)
            group.member_rows = None
            group.member_rows_cpu = None
            group.all_rows = None
            # Nothing was ever armed under these rows, so the retire path must
            # not try to withdraw them either.
            group.all_rows_cpu = None
            raise AssertionError(
                f"Beam member spawn needs {group.beam_width - 1} mamba slots but "
                f"only {self.req_to_token_pool.mamba_allocator.available_size()} "
                f"are free; raise --max-mamba-cache-size or lower beam_width."
            )
        # Past the failure branch, so the members own their slots: inherit the
        # leader's state into them. Unconditional on the COW / state-copy
        # switch -- neither prune-time strategy can recover a state that was
        # never inherited, and both are equally wrong without it. Runs on the
        # same stream as the prefill that produced the state (the relay point),
        # so the copy is ordered behind that write and ahead of the members'
        # first decode.
        fork_mamba = getattr(self.req_to_token_pool, "fork_mamba_from_row", None)
        if fork_mamba is not None:
            fork_mamba(leader.kv.req_pool_idx, member_rows)

    def select_and_relay_decode(
        self, batch: ScheduleBatch, logits_output: LogitsProcessorOutput
    ) -> None:
        """Launch half: joint-select every group in this decode batch on device,
        reparent KV, and overwrite the relayed next tokens."""
        tail = batch.beam_tail
        if tail is None:
            return
        capture = logits_output.beam
        for gi, entry in enumerate(tail.entries):
            group = entry.group
            if group.retired or group.state != BeamGroupState.DECODING:
                continue
            top_logprobs, top_tokens = _rows_topk_logprobs(
                [
                    capture.leader_logits[gi : gi + 1],
                    capture.tail_logits[entry.start : entry.end],
                ],
                group.num_candidates,
                # Row order matches pieces: leader first, then member rows in
                # frontier order -- the same [k, ...] layout as trie_cursor.
                self._trie_mask(group),
                penalty_state=group.penalty_state,
            )
            next_tokens, parent_idx = self._select_group(
                group, top_logprobs, top_tokens, batch.forward_iter
            )
            self._apply_survivors(group, next_tokens, parent_idx, batch.forward_iter)

    def commit_decode(self, batch: ScheduleBatch) -> set:
        """Deferred half: fold staged selections into the DAG and apply
        finish/abort. Returns groups finished by THIS call -- under overlap a
        finished leader reappears for one overshoot tick, so the caller runs the
        shared finish machinery only on the committing tick."""
        newly_finished = set()
        if self._num_live_groups == 0:
            return newly_finished
        for req in batch.reqs:
            group = req.beam_group
            if group is None or group.retired:
                continue
            # Reclaim even if the DAG commit is discarded: the launch path
            # already mutated req_to_token, so the orphans are real either way.
            self._reclaim_orphans(group, batch.forward_iter)
            if req.to_finish is not None:
                self._abort_group(group)
                newly_finished.add(id(group))
                continue
            if group.commit_pending(batch.forward_iter):
                self._finish_group(group)
                newly_finished.add(id(group))
        return newly_finished

    def _trie_mask(self, group: BeamGroup) -> Optional[torch.Tensor]:
        """[rows, vocab] forbidden-mask for this group's current cursors, or
        None when the group is unconstrained.

        Pure device work, so it stays legal before the launch-half selection.
        Callers MUST build it before _select_group, which advances the cursor:
        the mask describes the frontier this selection expands, not the next.
        Vocab is implied by the trie's own bounds; the select path checks the
        row count separately, so a mismatched group fails loudly instead of
        silently masking the wrong rows.
        """
        if group.trie is None or group.trie_cursor is None:
            return None
        return group.trie.build_mask(group.trie_cursor, self.model_config.vocab_size)

    def _select_group(
        self,
        group: BeamGroup,
        top_logprobs: torch.Tensor,
        top_tokens: torch.Tensor,
        tick: int,
    ):
        k = group.beam_width
        if group.next_step_is_final():
            # parent_idx is None on a final step: every selection finishes, so
            # no row moves onto another's slots.
            fsel = select_final_topk(
                group.frontier_cum_logprobs, top_logprobs, top_tokens, k
            )
            group.advance_final_frontier(fsel, tick)
            return fsel.tokens, None
        sel = joint_select(
            group.frontier_cum_logprobs,
            top_logprobs,
            top_tokens,
            group.stop_token_ids,
            k,
        )
        group.advance_frontier(sel, tick)
        return sel.next_tokens[:k], sel.parent_idx[:k]

    def _apply_survivors(
        self,
        group: BeamGroup,
        next_tokens: torch.Tensor,
        parent_idx: Optional[torch.Tensor],
        tick: int,
    ) -> None:
        rows = group.all_rows
        if parent_idx is not None:
            # Final steps skip this: their KV is never read again. Orphans are
            # staged, not freed here -- the set difference must not block launch.
            old_map, new_map = remap_kv_mapping(
                self.req_to_token_pool.req_to_token,
                rows=rows,
                parent_idx=parent_idx,
                prefix_len=group.prompt_len,
                # All rows are synchronized; the leader's committed length
                # covers the KV computed through this step.
                seq_len=group.leader.kv.kv_committed_len,
            )
            group.pending_orphans.append(StagedOrphans(tick, old_map, new_map))
            # SSM state rides the same reparenting: every row whose KV now
            # points at its parent's window must also decode from its parent's
            # recurrent state, or the two histories disagree from here on.
            self._apply_mamba_cow(rows, parent_idx, group.all_rows_cpu)
            # Length placeholder only; the DAG owns history and member rows
            # have no host state.
            group.leader.output_ids.append(0)
        self._stash_next_tokens(rows, next_tokens)

    def _apply_mamba_cow(
        self,
        rows: torch.Tensor,
        parent_idx: torch.Tensor,
        rows_cpu: Optional[Sequence[int]] = None,
    ) -> None:
        """Dispatch the mamba half of one group's prune.

        `rows_cpu` is `rows` on the host (same order). Only the double buffer
        needs it: it arms a snapshot that has to be withdrawn if this group ends
        before a decode consumes it (see BeamMambaCow.forget_rows). The true
        copy holds nothing across the tick, so it ignores it.

        Attention KV is reparented by the `remap_kv_mapping` call above; SSM
        state cannot be, so each pruned row has to be given its parent's state.
        Two ways to do that, picked by --enable-beam-mamba-state-copy:

          - the copy-on-write double buffer (default): the row reads its
            parent's slot for this one forward while writing a slot of its own,
            so the copy is fused into the step that was going to write anyway;
          - the true copy (_apply_mamba_state_copy): the state is copied
            outright, here and now.

        The dispatch sits here rather than in the admission gate because that
        gate asks `tree_cache.supports_mamba()`, which is only a proxy: a pool
        that carries SSM state but reports False there (ChunkCache) still runs
        this method. `replace_mamba_for_rows` is what actually says "this pool
        has per-row SSM state".

        No-op on a pool without the mamba double buffer (a non-hybrid pool).
        """
        pool = self.req_to_token_pool
        replace = getattr(pool, "replace_mamba_for_rows", None)
        if replace is None:
            return
        need = int(rows.numel())
        if need == 0:
            return
        # Check before touching anything: a short allocator must not leave a
        # half-installed read mapping / half-copied slot behind. len() reads a
        # shape, not data, so this stays off the sync path. Both paths need
        # `need` fresh slots, so both pay this check.
        if pool.mamba_allocator.available_size() < need:
            raise RuntimeError(
                f"Beam prune needs {need} fresh mamba slots for the mamba state "
                f"handover but only {pool.mamba_allocator.available_size()} "
                f"are free; raise --max-mamba-cache-size / "
                f"--mamba-full-memory-ratio, lower beam_width, or switch "
                f"--enable-beam-mamba-double-buffer / "
                f"--enable-beam-mamba-state-copy off."
            )
        if self.enable_beam_mamba_state_copy:
            # Nothing above has moved any state, so a raise in here still
            # leaves the group exactly as it was.
            self._apply_mamba_state_copy(rows, parent_idx, replace)
            return
        self._apply_mamba_cow_double_buffer(rows, parent_idx, replace, need, rows_cpu)

    def _apply_mamba_cow_double_buffer(
        self,
        rows: torch.Tensor,
        parent_idx: torch.Tensor,
        replace,
        need: int,
        rows_cpu: Optional[Sequence[int]] = None,
    ) -> None:
        """Write side of the copy-on-write double buffer for one group's prune.

        Each row reads its parent's slot for this one forward while writing a
        slot of its own. That read-then-write *is* the state copy, fused into
        the step that was going to write anyway.

        The three steps are order-critical and pure device work (no D2H):

          1. source slots = the parent rows' slots as they are right now;
          2. `set_beam_cow_read_mapping`, which snapshots the live table --
             it must run BEFORE step 3, or the snapshot would name the fresh
             slots and every row would read the slot it is about to write;
          3. `replace_mamba_for_rows`, which installs the fresh slots, and the
             detached old ones are staged to go back one tick late (the kernel
             is still reading them this forward).

        The cost is a cross-tick contract: the mapping is a snapshot of the
        whole row table that has to survive until the backend consumes it, and
        only the GDN packed decode path knows how to read a source index at
        all. That is the risk the true-copy path exists to price.

        Part of that contract is `rows_cpu`: this group may end in the very
        commit that follows, and then no decode ever consumes the mapping it
        armed here. The retire path withdraws those rows (see
        `_forget_beam_cow_rows`), so the next owner of the row numbers is not
        sent reading slots that were detached and freed in the meantime.
        """
        pool = self.req_to_token_pool
        # `parent_idx` is in `rows` order (the same index space remap_kv_mapping
        # uses), so the parent's slot is the row's own table entry gathered at
        # the parent's position -- read before step 3 overwrites any of them.
        src_slots = pool.req_index_to_mamba_index_mapping[rows][parent_idx]
        pool.set_beam_cow_read_mapping(rows, src_slots, rows_cpu=rows_cpu)
        previous = replace(rows)
        if previous is None:
            # Unreachable after the availability check above, but a silent None
            # here would leave the rows reading their parents' slots with no
            # fresh write target -- a self-read/write aliasing corruption.
            raise RuntimeError(
                f"Beam prune could not allocate {need} fresh mamba slots for the "
                f"copy-on-write double buffer; raise --max-mamba-cache-size or "
                f"lower beam_width."
            )
        if previous.numel():
            pool.stage_beam_cow_source_slots(previous)

    def _apply_mamba_state_copy(
        self,
        rows: torch.Tensor,
        parent_idx: torch.Tensor,
        replace,
    ) -> None:
        """True-copy alternative to the double buffer for one group's prune.

        Same shape as the COW path -- every pruned row ends up on a freshly
        allocated slot holding its parent's state -- but the copy happens here,
        in the prune, instead of being smuggled into the next forward's write:

          1. source slots = the parent rows' slots as they are right now
             (must be read before step 2 overwrites any of them);
          2. `replace_mamba_for_rows` installs a fresh slot per row;
          3. `MambaPool.copy_from` moves conv + temporal state parent -> fresh
             slot. Both sides are translated virtual -> physical first: the
             table holds virtual ids and copy_from addresses the state tensors.
          4. no `set_beam_cow_read_mapping`, so nothing spans the tick: the
             row reads and writes its own slot from here on.

        Why it is safe where the COW path is not: because step 2 allocates, the
        destinations are slots nobody owned a tick ago, hence disjoint from the
        sources by construction. An in-place variant (src and dst drawn from the
        same live slot set) would overlap for a self-parented row, and for two
        rows swapping parents -- and `copy_from`'s fused branch asserts src/dst
        disjoint, so those cases would be a corrupted copy rather than a loud
        failure. Allocating first is what buys the disjointness for free.

        What it costs: one state-sized copy per row per prune, on the critical
        path, whereas the COW path's copy is folded into a write it was going to
        do anyway. It is the slower path; it is here as a correctness baseline
        and a bisection handle.

        What it buys: no snapshot and no backend contract. `src_mamba_cache_indices`
        stays None, so the cuda-graph gates stay open and the KDA / Mamba2 /
        non-packed GDN decode paths -- which raise NotImplementedError on a
        separate source index -- work unchanged.

        The old slots still go back one tick late (step 4 reuses the existing
        staging queue). `copy_from` only enqueues on the stream, so an immediate
        free would let the allocator reissue a slot whose copy has not landed;
        the deferred release is the same protection the COW path needs, and it
        carries none of the defects the read mapping does.
        """
        pool = self.req_to_token_pool
        mamba_pool = pool.mamba_pool
        # Checked before anything moves: copy_from carries only `temporal`, not
        # the ReplaySSM ring, and it resets the destination cursor to 0. A beam
        # prune copies from an *active* decoding slot whose ring is non-empty,
        # so those pending updates would be dropped silently and the state
        # would disagree with the ring the next decode reads.
        if mamba_pool.replayssm_write_pos is not None:
            raise RuntimeError(
                "Beam mamba state copy (--enable-beam-mamba-state-copy) is not "
                "compatible with ReplaySSM: the copy moves temporal state but "
                "not the ring, and a beam prune copies from an active decoding "
                "slot, so its un-flushed ring entries would be dropped "
                "silently. Disable --enable-linear-replayssm / "
                "--enable-linear-replayssm-spec, or run with "
                "--enable-beam-mamba-double-buffer instead."
            )
        src_slots = pool.req_index_to_mamba_index_mapping[rows][parent_idx]
        previous = replace(rows)
        if previous is None:
            raise RuntimeError(
                f"Beam prune could not allocate {int(rows.numel())} fresh mamba "
                f"slots for the state copy; raise --max-mamba-cache-size or "
                f"lower beam_width."
            )
        new_slots = pool.req_index_to_mamba_index_mapping[rows]
        mamba_pool.copy_from(
            pool.translate_mamba_indices(src_slots),
            pool.translate_mamba_indices(new_slots),
        )
        if previous.numel():
            pool.stage_beam_cow_source_slots(previous)

    def _reclaim_orphans(
        self, group: BeamGroup, up_to_tick: Optional[int] = None
    ) -> None:
        # Callers must be past the tick's copy_done sync: collect_orphan_slots
        # synchronizes, which would stall the launch path.
        if not group.pending_orphans:
            return
        if up_to_tick is None:
            # Teardown. Only records the commit half actually validated may be
            # drained here: a launch-half write-back from an uncommitted tick
            # has its old_mapping slots still reachable from the live rows, and
            # free_member_rows is what owns those. Draining them blindly would
            # double-free. Records past last_committed_tick are left to
            # free_member_rows' live-window reclaim.
            staged = [
                e for e in group.pending_orphans if e.tick <= group.last_committed_tick
            ]
            group.pending_orphans = []
        else:
            staged = [e for e in group.pending_orphans if e.tick <= up_to_tick]
            group.pending_orphans = [
                e for e in group.pending_orphans if e.tick > up_to_tick
            ]
        if not staged:
            return
        # Plain free(), never a nested free_group: an inner begin/end pair
        # inside the decode path's group double-frees.
        allocator = self.token_to_kv_pool_allocator
        for entry in staged:
            orphans = collect_orphan_slots(entry.old_mapping, entry.new_mapping)
            if orphans.numel():
                group.slots_freed += orphans.numel()
                allocator.free(orphans)

    def _finish_group(self, group: BeamGroup) -> None:
        # The leader carries the best sequence's finish reason.
        group.final_results = group.finalize()
        top = group.final_results[0]
        leader = group.leader
        if top.matched_token is not None:
            leader.finished_reason = FINISH_MATCHED_TOKEN(matched=top.matched_token)
        else:
            leader.finished_reason = FINISH_LENGTH(length=group.num_committed)
        self._free_member_rows(group)
        self._retire_group(group)

    def _abort_group(self, group: BeamGroup) -> None:
        group.state = BeamGroupState.FINISHED
        group.final_results = []
        leader = group.leader
        leader.finished_reason = leader.to_finish or FINISH_ABORT("Beam group aborted.")
        leader.to_finish = None
        self._free_member_rows(group)
        self._retire_group(group)

    def retire_group(self, req: Req) -> None:
        """Leader ended outside the commit path -- retracted, or aborted while
        still queued. Member rows, if any, were released by the caller."""
        group = req.beam_group
        if group is None or group.retired:
            return
        group.state = BeamGroupState.FINISHED
        group.final_results = []
        self._retire_group(group)

    def _free_member_rows(self, group: BeamGroup) -> None:
        # Order matters: free_member_rows reclaims slots that only the staged
        # orphan records can still name (a write-back whose old_mapping is no
        # longer reachable from any live row), so it must run while the queue
        # is intact. _reclaim_orphans afterwards drains whatever is left --
        # its teardown branch clears the queue unconditionally, which would
        # otherwise hide those records from the reclaim logic entirely.
        free_member_rows(group, self.req_to_token_pool, self.token_to_kv_pool_allocator)
        self._reclaim_orphans(group)

    def _forget_beam_cow_rows(self, group: BeamGroup) -> None:
        """Withdraw this group's rows from an unconsumed mamba COW snapshot.

        Called from `_retire_group` because that is the one place all three
        exits converge -- `_finish_group`, `_abort_group`, and the retract-abort
        path (`retire_group`) -- and it runs while the group is still the owner
        of its rows.

        Timing: on the finish and abort paths `fork.free_member_rows` has
        already handed the member rows back to `free_slots` (it runs just
        before), and on the retract path `retract_decode` freed them together
        with the leader's row. That is still safe: everything between that free
        and this call is straight-line, allocating-nothing code on the scheduler
        thread, so no other request can have been issued these row numbers in
        between. What matters is only that the withdrawal happens before the
        scheduler next admits a request, which it does.

        No-op when the group never armed a snapshot (no spawn, or a spawn that
        rolled back) and on a non-hybrid pool, which has no such method.
        """
        # getattr, not an attribute: duck-typed group stand-ins (tests, and the
        # retract path's partially released group) do not all carry the field.
        rows_cpu = getattr(group, "all_rows_cpu", None)
        if rows_cpu is None:
            return
        forget = getattr(self.req_to_token_pool, "forget_beam_cow_rows", None)
        if forget is None:
            return
        rows = getattr(group, "all_rows", None)
        if rows is None:
            # The rows are already released; only the host copy survived, so
            # rebuild the device index. Teardown path, not the hot one.
            rows = torch.tensor(
                rows_cpu, dtype=torch.int64, device=self.req_to_token_pool.device
            )
        forget(rows, rows_cpu)

    def _retire_group(self, group: BeamGroup) -> None:
        # Exactly once per group, so the O(1) live gate stays accurate.
        if not group.retired:
            # Staged orphans are referenced by no row, so the retract-abort
            # path's direct fork.free_member_rows cannot see them.
            self._reclaim_orphans(group)
            # Before `retired = True`, i.e. while the group still owns its rows:
            # a snapshot armed by this group's last prune would otherwise outlive
            # it and serve the next owner of these rows freed parent slots.
            self._forget_beam_cow_rows(group)
            group.retired = True
            # Drops overshoot selections staged after the terminal commit.
            group._pending_steps.clear()
            # Drop the group's penalty state: it holds a [k, vocab] count table
            # that nothing reads once the group is gone. Only the reference goes
            # away -- no zeroing kernel is enqueued, so this never touches a
            # stream the device may still be running on, and the allocator's own
            # stream bookkeeping is what makes the block reusable afterwards.
            group.penalty_state = None
            self._num_live_groups -= 1

    def _stash_next_tokens(self, rows, tokens) -> None:
        # Accepts GPU tensors (decode path, no D2H) or host lists (prefill).
        device = self.req_to_token_pool.device
        if not torch.is_tensor(rows):
            rows = torch.tensor(rows, dtype=torch.int64, device=device)
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.int64, device=device)
        self.future_map.stash(rows, RelayPayload(bonus_tokens=tokens))
