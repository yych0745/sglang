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
"""Beam search repetition penalties: frequency / presence / repetition.

Behaviour tests, driven through the real selection path (``_rows_topk_logprobs``
+ ``joint_select`` + ``BeamGroup.advance_frontier``), not through mocks.

Scope, and why HF is not the oracle: these three penalties count ONLY the
tokens a branch has GENERATED. The prompt is never counted, so the very first
(prefill) selection sees all-zero counts. HF's default ``repetition_penalty``
is a different animal -- it runs inside the logits-processor list, i.e. after
``log_softmax``, and it counts the prompt -- so it is never used as a reference
here. Every expected value comes from the explicit formula in
``oracle_logprobs`` below.

Order, pinned by tests: raw logits -> fp32 -> frequency (subtract f * count)
-> presence (subtract p where count > 0) -> repetition (seen positions only:
negative logit * r, non-negative logit / r) -> logsumexp -> trie mask (AFTER
the normalisation, unchanged from before).
"""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.beam_search import BeamGroup, materialize_tokens
from sglang.srt.beam_search.coordinator import BeamCoordinator, _rows_topk_logprobs
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.beam_search.penalty_state import (
    BeamPenaltyState,
    make_beam_penalty_state,
)
from sglang.srt.beam_search.trie import BeamTrie
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

VOCAB = 8
EOS = 7

# Frequency / presence / repetition used by most tests.
F, P, R = 0.4, 0.6, 1.5


# --- helpers ----------------------------------------------------------------


def flat_logits(rows, base=-6.0):
    """[len(rows), VOCAB] logits: `base` everywhere except the explicit
    {token: value} entries. The base carries a tiny per-token taper so the
    filler tokens never tie (a tie would make topk's pick an implementation
    detail, and the 2k-th candidate is what decides some of these tests)."""
    out = torch.tensor(
        [[base - 0.001 * v for v in range(VOCAB)] for _ in rows], dtype=torch.float32
    )
    for r, spec in enumerate(rows):
        for tok, val in spec.items():
            out[r, tok] = float(val)
    return out


def _adjust(row, generated, f=0.0, p=0.0, r=1.0, order="raw"):
    """Penalty chain for one row, written out longhand.

    order="raw" is ours: penalties on raw logits, then logsumexp.
    order="hf" is the HF default beam order: log_softmax first, then the
    processors on log-probs. It exists only so a test can show the two differ.
    """
    counts = {}
    for t in generated:
        counts[t] = counts.get(t, 0) + 1
    lse = math.log(sum(math.exp(float(x)) for x in row))
    out = []
    for v, raw in enumerate(row):
        c = counts.get(v, 0)
        x = float(raw)
        if order == "hf":
            x = x - lse
        if c:
            x -= f * c
            x -= p
            x = x * r if x < 0 else x / r
        out.append(x)
    if order == "hf":
        # The processors act on log-probs that are already normalized, so
        # nothing is subtracted a second time.
        return out
    lse = math.log(sum(math.exp(a) for a in out))
    return [a - lse for a in out]


def oracle_logprobs(rows, generated_per_row, f=0.0, p=0.0, r=1.0):
    """Independent oracle: our documented order, one row at a time."""
    return [_adjust(row, gen, f, p, r, "raw") for row, gen in zip(rows, generated_per_row)]


def hf_order_logprobs(rows, generated_per_row, f=0.0, p=0.0, r=1.0):
    """The HF-default beam order (log_softmax, then the processor). Never equal
    to ours; kept so the difference is asserted, not assumed."""
    return [_adjust(row, gen, f, p, r, "hf") for row, gen in zip(rows, generated_per_row)]


def scored(pieces, state=None):
    """Full-vocab logprobs per row through the production scorer."""
    vals, toks = _rows_topk_logprobs(pieces, VOCAB, penalty_state=state)
    out = torch.zeros(vals.shape[0], VOCAB, dtype=torch.float32)
    # num_candidates == VOCAB, so `toks` is a permutation of the vocabulary.
    out.scatter_(1, toks, vals)
    return out


def split_pieces(logits):
    """The decode-path layout: one leader piece, then the member tail."""
    return [logits] if logits.shape[0] == 1 else [logits[:1], logits[1:]]


def make_group(**kw):
    defaults = dict(
        beam_width=2,
        stop_token_ids=[EOS],
        max_new_tokens=8,
        early_stopping=False,
        device="cpu",
    )
    defaults.update(kw)
    return BeamGroup(**defaults)


def launch_step(group, logits, mask=None, tick=0):
    """The coordinator's launch half for one tick: score, joint-select, advance.
    Returns the raw selection so callers can look at the chosen tokens."""
    probs, toks = _rows_topk_logprobs(
        split_pieces(logits),
        group.num_candidates,
        mask=mask,
        penalty_state=group.penalty_state,
    )
    if group.next_step_is_final():
        fsel = select_final_topk(
            group.frontier_cum_logprobs, probs, toks, group.beam_width
        )
        group.advance_final_frontier(fsel, tick)
        return fsel
    sel = joint_select(
        group.frontier_cum_logprobs, probs, toks, group.stop_token_ids, group.beam_width
    )
    group.advance_frontier(sel, tick)
    return sel


def step(group, logits, **kw):
    """Launch half + deferred half, i.e. one complete tick."""
    sel = launch_step(group, logits, **kw)
    group.commit_pending()
    return sel


def history(group):
    """Generated tokens of every frontier row, from the DAG (not from counts)."""
    return [materialize_tokens(leaf) for leaf in group.leaves]


def _coordinator():
    """A real BeamCoordinator with only what the retire/validate paths touch."""
    return BeamCoordinator(
        model_config=SimpleNamespace(vocab_size=1000, is_encoder_decoder=False),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        dllm_enabled=False,
        max_req_len=64,
        req_to_token_pool=SimpleNamespace(size=64, device="cpu"),
        token_to_kv_pool_allocator=None,
        tree_cache=SimpleNamespace(
            supports_swa=lambda: False, supports_mamba=lambda: False
        ),
        future_map=None,
    )


# --- does each penalty actually move the ranking? ----------------------------


class TestPenaltyChangesRanking(CustomTestCase):
    """Each of the three must change WHICH token a real BeamGroup selects."""

    def test_presence_penalty_stops_a_branch_repeating_itself(self):
        # Both rows are shaped so that, unpenalised, row 0 repeats token 1
        # (its own last token) and takes both survivor slots.
        first = flat_logits([{1: 0.0, 2: -0.1}])
        second = flat_logits([{1: 0.0, 2: -0.05}, {2: 0.0, 1: -0.05}])

        plain = make_group()
        step(plain, first)
        step(plain, second)
        self.assertEqual(
            sorted(tuple(h) for h in history(plain)), [(1, 1), (1, 2)]
        )

        penal = make_group(presence_penalty=1.0)
        step(penal, first)
        step(penal, second)
        # Each branch now prefers the OTHER branch's token: no repetition.
        self.assertEqual(
            sorted(tuple(h) for h in history(penal)), [(1, 2), (2, 1)]
        )

    def _greedy(self, spec=None, max_new_tokens=4, **pen):
        """beam_width 1: one branch, the same logits every step. Token 1 is the
        best raw candidate, token 2 the runner-up; `spec` sets the gap."""
        group = make_group(beam_width=1, max_new_tokens=max_new_tokens, **pen)
        logits = flat_logits([spec or {1: 0.0, 2: -0.45}])
        for _ in range(3):
            step(group, logits)
        return history(group)[0]

    def test_frequency_penalty_accumulates_with_the_count(self):
        # f = 0.3: token 1 survives one penalty (-0.3 > -0.45) but not two
        # (-0.6 < -0.45), so it repeats exactly twice and then switches.
        self.assertEqual(self._greedy(frequency_penalty=0.3), [1, 1, 2])

    def test_presence_penalty_is_charged_once_only(self):
        # p = 0.6: one penalty already outweighs the 0.45 gap, and a second
        # occurrence costs nothing extra -- the branch alternates instead.
        self.assertEqual(self._greedy(presence_penalty=0.6), [1, 2, 1])

    def test_repetition_penalty_changes_ranking(self):
        # Both candidates are NEGATIVE, so r = 3.0 multiplies them: token 1
        # goes -0.2 -> -0.6, below the runner-up at -0.5, and the repeat dies
        # on the first offence. (A zero logit is invariant under the scaling --
        # 0 / r == 0 -- which is why this uses negative logits.)
        self.assertEqual(
            self._greedy({1: -0.2, 2: -0.5}, repetition_penalty=3.0), [1, 2, 1]
        )

    def test_negative_penalties_encourage_repetition(self):
        # f/p < 0 are accepted by SamplingParams.verify and act as a bonus:
        # a greedy branch that already repeats keeps repeating.
        self.assertEqual(self._greedy(frequency_penalty=-1.0), [1, 1, 1])
        self.assertEqual(self._greedy(presence_penalty=-1.0), [1, 1, 1])

    def test_repetition_below_one_encourages_repetition(self):
        # r < 1 shrinks the magnitude of a negative logit: -0.2 -> -0.1, so the
        # repeat keeps winning.
        self.assertEqual(
            self._greedy({1: -0.2, 2: -0.5}, repetition_penalty=0.5), [1, 1, 1]
        )


# --- order, scope, and numerical agreement -----------------------------------


class TestPenaltyOrderAndScope(CustomTestCase):
    def test_scores_match_the_independent_oracle(self):
        group = make_group(frequency_penalty=F, presence_penalty=P, repetition_penalty=R)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        logits = flat_logits([{1: 0.0, 2: -0.2, 3: -0.5}, {2: 0.0, 1: -0.2, 4: -0.5}])
        got = scored(split_pieces(logits), group.penalty_state)
        expected = oracle_logprobs(logits.tolist(), [[1], [2]], f=F, p=P, r=R)
        for r in range(logits.shape[0]):
            for v in range(VOCAB):
                self.assertAlmostEqual(float(got[r, v]), expected[r][v], places=5)

    def test_oracles_disagree_so_the_order_is_actually_pinned(self):
        # Guard against a vacuous "matches the oracle" test: ours and the HF
        # default order must not produce the same numbers, or the assertion
        # above would pass for either implementation.
        rows = [[0.0, 1.0, 2.0, -3.0, -4.0, -5.0, -6.0, -7.0]]
        ours = oracle_logprobs(rows, [[1, 1, 2]], f=F, p=P, r=R)[0]
        hf = hf_order_logprobs(rows, [[1, 1, 2]], f=F, p=P, r=R)[0]
        for a, b in zip(ours, hf):
            self.assertNotAlmostEqual(a, b, places=3)

    def test_additive_penalties_run_before_the_scaling_one(self):
        # Token 3 sits at +0.30 raw. f = 1.0 pushes it to -0.70 (across zero),
        # and only THEN does r = 2.0 scale the negative value: -1.40.
        group = make_group(beam_width=1, frequency_penalty=1.0, repetition_penalty=2.0)
        step(group, flat_logits([{3: 0.0, 2: -1.0}]))
        row = flat_logits([{3: 0.30, 2: -1.0}])
        got = scored([row], group.penalty_state)[0]

        adjusted = [float(x) for x in row[0]]
        adjusted[3] = (0.30 - 1.0) * 2.0
        lse = math.log(sum(math.exp(a) for a in adjusted))
        self.assertAlmostEqual(float(got[3]), adjusted[3] - lse, places=5)
        # Not the scaling-first value (+0.15), nor the unscaled additive (-0.70).
        self.assertNotAlmostEqual(float(got[3]), 0.15 - lse, places=3)
        self.assertNotAlmostEqual(float(got[3]), -0.70 - lse, places=3)

    def test_penalties_apply_to_raw_logits_not_to_log_probs(self):
        # A global shift of the raw logits must change our scores (the scaling
        # sees the shifted value) while leaving the HF order untouched.
        group = make_group(beam_width=1, repetition_penalty=2.0)
        step(group, flat_logits([{1: 0.0, 2: -1.0}]))
        low = flat_logits([{1: 1.0, 2: 0.0, 3: 2.0}])
        high = low + 10.0

        got_low = float(scored([low], group.penalty_state)[0][1])
        got_high = float(scored([high], group.penalty_state)[0][1])
        self.assertNotAlmostEqual(got_low, got_high, places=3)

        hf_low = hf_order_logprobs(low.tolist(), [[1]], r=2.0)[0]
        hf_high = hf_order_logprobs(high.tolist(), [[1]], r=2.0)[0]
        self.assertAlmostEqual(hf_low[1], hf_high[1], places=5)
        self.assertNotAlmostEqual(got_low, hf_low[1], places=3)

    def test_the_first_selection_is_unpenalized(self):
        # Counts start empty, so the prefill step is scored raw even with all
        # three penalties maxed out.
        plain = make_group()
        penal = make_group(frequency_penalty=2.0, presence_penalty=2.0, repetition_penalty=2.0)
        logits = flat_logits([{1: 0.0, 2: -0.1}])
        self.assertTrue(
            torch.allclose(
                scored([logits], penal.penalty_state),
                scored([logits], plain.penalty_state),
                atol=0,
                rtol=0,
            )
        )

    def test_the_prompt_is_never_counted(self):
        # A prompt made of five token-1s contributes nothing: after one step
        # that generated token 1, token 1 carries ONE count, not six.
        group = make_group(frequency_penalty=0.5)
        group.leader = SimpleNamespace(origin_input_ids=[1] * 5)
        group.prompt_len = 5
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))

        logits = flat_logits([{1: 0.0, 2: -0.2}, {2: 0.0, 1: -0.2}])
        got = scored(split_pieces(logits), group.penalty_state)
        counted_generated = oracle_logprobs(logits.tolist(), [[1], [2]], f=0.5)
        counted_prompt = oracle_logprobs(
            logits.tolist(), [[1] * 6, [1] * 5 + [2]], f=0.5
        )
        for r in range(2):
            for v in range(VOCAB):
                self.assertAlmostEqual(float(got[r, v]), counted_generated[r][v], places=5)
        self.assertNotAlmostEqual(
            float(got[0, 1]), counted_prompt[0][1], places=3
        )


# --- count mechanics through the real DAG ------------------------------------


class TestPenaltyCountMechanics(CustomTestCase):
    def test_prefill_expands_one_row_to_k(self):
        group = make_group(frequency_penalty=F)
        self.assertIsNone(group.penalty_state.counts)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        counts = group.penalty_state.counts
        self.assertEqual(list(counts.shape), [group.beam_width, VOCAB])
        self.assertEqual(counts.dtype, torch.int32)
        # One count per surviving branch, on the token that branch generated.
        self.assertEqual(int(counts.sum()), group.beam_width)

    def _counts_from_history(self, group):
        expected = []
        for leaf in group.leaves:
            row = [0] * VOCAB
            for t in materialize_tokens(leaf):
                row[t] += 1
            expected.append(row)
        return expected

    def test_counts_follow_the_dag_not_the_frontier_order(self):
        # Three steps with all three penalties on. Every row's counts must be
        # exactly the multiset of tokens on its own DAG path -- which exercises
        # repeated parents, self-parents, swapped parents and branch
        # independence at once, without a host-side survivor count.
        group = make_group(frequency_penalty=F, presence_penalty=P, repetition_penalty=R)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        step(group, flat_logits([{1: 0.0, 2: -0.05}, {2: 0.0, 1: -0.05}]))
        step(group, flat_logits([{3: 0.0, 1: -0.3}, {4: 0.0, 2: -0.3}]))
        # No dummy rows were seated, so leaves are the true per-row histories.
        self.assertEqual(len(group.leaves), group.beam_width)
        self.assertEqual(
            group.penalty_state.counts.tolist(), self._counts_from_history(group)
        )

    def test_branch_counts_stay_independent(self):
        group = make_group(frequency_penalty=1.0)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        counts = group.penalty_state.counts
        # Each branch counted only its own token.
        self.assertEqual(int(counts[0, 1]) + int(counts[0, 2]), 1)
        self.assertEqual(int(counts[1, 1]) + int(counts[1, 2]), 1)
        self.assertEqual(int(counts[0].sum()), 1)
        self.assertEqual(int(counts[1].sum()), 1)
        self.assertNotEqual(counts[0].tolist(), counts[1].tolist())

    def test_finished_stop_candidates_never_enter_the_counts(self):
        group = make_group(frequency_penalty=1.0)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        # EOS is the best candidate of both rows: it finishes, and the second
        # best continues. The stop token must be counted nowhere.
        step(
            group,
            flat_logits(
                [{EOS: 0.0, 3: -0.2, 1: -0.4}, {EOS: 0.0, 4: -0.2, 2: -0.4}]
            ),
        )
        self.assertEqual(int(group.penalty_state.counts[:, EOS].max()), 0)
        self.assertEqual(int(group.penalty_state.counts[:, EOS].sum()), 0)

    def test_the_final_step_does_not_advance_the_counts(self):
        group = make_group(frequency_penalty=F, max_new_tokens=2)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        after_first = group.penalty_state.counts.clone()
        # The next step hits max_new_tokens: it scores with the current counts
        # and then finishes; nothing is ever read from the counts again.
        self.assertTrue(group.next_step_is_final())
        step(group, flat_logits([{3: 0.0, 4: -0.1}, {4: 0.0, 3: -0.1}]))
        self.assertTrue(torch.equal(group.penalty_state.counts, after_first))

    def test_max_new_tokens_one_never_advances(self):
        group = make_group(frequency_penalty=F, max_new_tokens=1)
        # The prefill selection is itself the final one.
        self.assertTrue(group.next_step_is_final())
        launch_step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        counts = group.penalty_state.counts
        self.assertEqual(list(counts.shape), [1, VOCAB])
        self.assertEqual(int(counts.sum()), 0)

    def test_dummy_rows_are_advanced_but_never_promoted(self):
        # Three of the four candidates are stop tokens: one real survivor, so
        # the group pads the frontier with a dummy leaf (-inf cum) instead of
        # finishing. Every row is advanced, dummy included, but the dummy can
        # never win a later selection or reach the result pool.
        group = make_group(
            beam_width=2, stop_token_ids=[5, 6, 7], frequency_penalty=1.0
        )
        sel = launch_step(group, flat_logits([{5: 0.0, 6: -0.1, 7: -0.2, 1: -0.3}]))
        group.commit_pending()
        self.assertEqual(int(sel.num_survivors), 1)
        self.assertEqual(len(group.completed), 3)
        self.assertEqual(len(group.leaves), 2)  # padded to beam_width

        # Next step: the dummy row carries -inf, so both survivors come from
        # the one real row.
        launch_step(group, flat_logits([{1: 0.0, 2: -0.1}, {1: 0.0, 2: -0.1}]))
        group.commit_pending()
        self.assertEqual(
            sorted(tuple(h) for h in history(group)), [(1, 1), (1, 2)]
        )
        # One branch repeated token 1 (and is charged for it: the penalty is
        # why the other branch's token 1 ranked second here), the other went
        # to token 2.
        counts = group.penalty_state.counts.tolist()
        self.assertEqual(
            sorted(counts), [[0, 1, 1, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0]]
        )

    def test_only_the_launch_half_advances_the_counts(self):
        # Overlap: the launch half advances for tick N while the deferred half
        # for tick N-1 commits. A commit must never touch the counts, so an
        # off-by-one tick cannot double-count.
        group = make_group(frequency_penalty=F)
        launch_step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        after_launch = group.penalty_state.counts.clone()
        self.assertEqual(int(after_launch.sum()), group.beam_width)
        group.commit_pending()
        self.assertTrue(torch.equal(group.penalty_state.counts, after_launch))


# --- defaults stay exactly as before -----------------------------------------


class TestNoOpDefaults(CustomTestCase):
    def test_defaults_create_no_state(self):
        self.assertIsNone(make_group().penalty_state)
        self.assertIsNone(
            make_group(
                frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0
            ).penalty_state
        )

    def test_legacy_scores_are_identical_without_a_state(self):
        logits = torch.randn(3, VOCAB)
        group = make_group()
        vals, toks = _rows_topk_logprobs(
            [logits], 4, penalty_state=group.penalty_state
        )
        ref_vals, ref_toks = torch.topk(torch.log_softmax(logits, -1), 4, dim=-1)
        # allclose, not equal: logsumexp and log_softmax differ by an ulp.
        self.assertTrue(torch.allclose(vals, ref_vals, atol=1e-6))
        self.assertTrue(torch.equal(toks, ref_toks))

    def test_default_group_matches_the_unpenalised_oracle(self):
        group = make_group()
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        logits = flat_logits([{1: 0.0, 2: -0.2}, {2: 0.0, 1: -0.2}])
        got = scored(split_pieces(logits), group.penalty_state)
        expected = oracle_logprobs(logits.tolist(), [[1], [2]])
        for r in range(2):
            for v in range(VOCAB):
                self.assertAlmostEqual(float(got[r, v]), expected[r][v], places=5)

    def test_leader_and_tail_pieces_get_consistent_row_offsets(self):
        # The tail piece must index counts from row 1, not row 0, and an empty
        # piece must advance the row cursor by nothing.
        group = make_group(frequency_penalty=1.0)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        logits = flat_logits([{1: 0.0, 2: -0.2}, {2: 0.0, 1: -0.2}])
        split = _rows_topk_logprobs(
            [logits[:1], torch.zeros(0, VOCAB), logits[1:]],
            group.num_candidates,
            penalty_state=group.penalty_state,
        )
        whole = _rows_topk_logprobs(
            [logits], group.num_candidates, penalty_state=group.penalty_state
        )
        self.assertTrue(torch.allclose(split[0], whole[0], atol=1e-6))
        self.assertTrue(torch.equal(split[1], whole[1]))
        # Sanity: the two rows really do carry different counts, so a wrong
        # offset could not pass by accident.
        counts = group.penalty_state.counts
        self.assertNotEqual(counts[0].tolist(), counts[1].tolist())

    def test_scoring_does_not_mutate_the_captured_logits(self):
        group = make_group(frequency_penalty=F, presence_penalty=P, repetition_penalty=R)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        logits = flat_logits([{1: 0.0, 2: -0.1}, {3: 0.0, 1: -0.1}])
        before = logits.clone()
        scored(split_pieces(logits), group.penalty_state)
        self.assertTrue(torch.equal(logits, before))

    def test_fp16_source_matches_fp32(self):
        group = make_group(frequency_penalty=F, repetition_penalty=R)
        step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        logits = flat_logits([{1: 0.0, 2: -0.2}, {2: 0.0, 1: -0.2}])
        from_fp32 = scored(split_pieces(logits), group.penalty_state)
        from_fp16 = scored(
            [p.half() for p in split_pieces(logits)], group.penalty_state
        )
        self.assertTrue(torch.allclose(from_fp32, from_fp16, atol=2e-3))


# --- the state object itself -------------------------------------------------


class TestBeamPenaltyState(CustomTestCase):
    def _state(self, rows=2, **pen):
        state = BeamPenaltyState(**pen)
        # Init the count table the way the selection path does.
        state.apply(torch.zeros(rows, VOCAB))
        return state

    def test_factory_returns_none_only_at_the_defaults(self):
        self.assertIsNone(make_beam_penalty_state())
        self.assertIsNone(make_beam_penalty_state(0.0, 0.0, 1.0))
        for pen in (
            dict(frequency_penalty=0.5),
            dict(presence_penalty=0.5),
            dict(repetition_penalty=1.5),
            dict(repetition_penalty=0.5),
        ):
            state = make_beam_penalty_state(**pen)
            self.assertIsInstance(state, BeamPenaltyState)
            self.assertTrue(state.required)
        self.assertFalse(BeamPenaltyState().required)

    def test_counts_are_int32_on_the_logits_device(self):
        state = BeamPenaltyState(frequency_penalty=0.5)
        self.assertIsNone(state.counts)
        state.apply(torch.zeros(3, VOCAB))
        self.assertEqual(state.counts.dtype, torch.int32)
        self.assertEqual(state.counts.device, torch.zeros(3, VOCAB).device)
        self.assertEqual(list(state.counts.shape), [3, VOCAB])

    def test_apply_returns_a_fresh_fp32_tensor(self):
        state = self._state(frequency_penalty=0.5)
        src = torch.zeros(2, VOCAB)  # already fp32: `.to()` would alias it
        out = state.apply(src)
        self.assertIsNot(out, src)
        self.assertEqual(out.dtype, torch.float32)
        out[0, 0] = 123.0
        self.assertEqual(float(src[0, 0]), 0.0)

    def test_repetition_splits_on_the_sign_of_the_logit(self):
        state = self._state(rows=1, repetition_penalty=2.0)
        for tok in (1, 2, 3):
            state.advance(torch.tensor([0]), torch.tensor([tok]))
        # One row, three seen tokens: +0.8 -> 0.8 / 2 = 0.4; -0.6 -> -1.2;
        # 0.0 -> 0.0 (0 / r == 0, the non-negative branch).
        row = torch.zeros(1, VOCAB)
        row[0, 1] = 0.8
        row[0, 2] = -0.6
        row[0, 3] = 0.0
        out = state.apply(row)[0]
        self.assertAlmostEqual(float(out[1]), 0.4, places=6)
        self.assertAlmostEqual(float(out[2]), -1.2, places=6)
        self.assertAlmostEqual(float(out[3]), 0.0, places=6)
        # Unseen tokens are untouched.
        self.assertAlmostEqual(float(out[4]), 0.0, places=6)

    def test_presence_is_charged_once_and_frequency_per_count(self):
        # Token 1 generated twice on row 0.
        state = self._state(frequency_penalty=0.25, presence_penalty=1.0)
        state.advance(torch.tensor([0, 0]), torch.tensor([1, 1]))
        state.advance(torch.tensor([0, 0]), torch.tensor([1, 2]))
        row = torch.zeros(2, VOCAB)
        row[0, 1] = 1.0
        out = state.apply(row)[0]
        # 1.0 - 0.25*2 - 1.0 = -0.5: frequency charged per count (0.50),
        # presence once (1.00), not per count (-1.5) and not skipped (0.5).
        self.assertAlmostEqual(float(out[1]), -0.5, places=6)
        self.assertNotAlmostEqual(float(out[1]), -1.5, places=3)
        self.assertNotAlmostEqual(float(out[1]), 0.5, places=3)

    def test_negative_penalties_add(self):
        state = self._state(frequency_penalty=-0.5, presence_penalty=-0.25)
        state.advance(torch.tensor([0]), torch.tensor([1]))
        row = torch.zeros(1, VOCAB)
        row[0, 1] = 1.0
        out = state.apply(row)[0]
        self.assertAlmostEqual(float(out[1]), 1.75, places=6)

    def test_swapped_parents_do_not_alias(self):
        state = self._state(rows=2, frequency_penalty=0.5)
        state.advance(torch.tensor([0, 0]), torch.tensor([5, 6]))
        # Rows swap parents: an in-place row loop would have row 0 read the
        # row 1 it just overwrote.
        state.advance(torch.tensor([1, 0]), torch.tensor([7, 3]))
        counts = state.counts.tolist()
        self.assertEqual(counts[0][5], 0)
        self.assertEqual(counts[0][6], 1)  # inherited from old row 1
        self.assertEqual(counts[0][7], 1)  # its own token
        self.assertEqual(counts[1][5], 1)  # inherited from old row 0
        self.assertEqual(counts[1][6], 0)
        self.assertEqual(counts[1][3], 1)

    def test_repeated_parent_copies_without_double_counting(self):
        state = self._state(rows=3, frequency_penalty=0.5)
        state.advance(torch.tensor([0, 0, 0]), torch.tensor([3, 3, 3]))
        counts = state.counts.tolist()
        for row in counts:
            self.assertEqual(row[3], 1, "both children of one parent count once")

    def test_parent_rotation(self):
        # A 3-cycle: every row inherits from a different row.
        state = self._state(rows=3, frequency_penalty=0.5)
        state.advance(torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]))
        state.advance(torch.tensor([1, 2, 0]), torch.tensor([4, 5, 6]))
        counts = state.counts.tolist()
        self.assertEqual([counts[0][2], counts[0][4]], [1, 1])
        self.assertEqual([counts[1][3], counts[1][5]], [1, 1])
        self.assertEqual([counts[2][1], counts[2][6]], [1, 1])

    def test_advance_before_the_first_apply_is_a_no_op(self):
        # A group that never scored (e.g. aborted at admission) has no table.
        state = BeamPenaltyState(frequency_penalty=0.5)
        state.advance(torch.tensor([0]), torch.tensor([1]))
        self.assertIsNone(state.counts)


# --- a legal tiny repetition_penalty must not become a divide by zero --------


# Small enough that raw / r (100 at r = 1e-8) stays far inside fp32, so any
# disagreement below is about the scaling and not about overflow.
SMALL_RAW = 1e-6
# Both are legal (SamplingParams.verify accepts them), and both are small
# enough that `seen * (r - 1) + 1` misbehaves in fp32: it cancels to exactly
# 0.0 at 1e-8, and to a ~19%-off 1.19e-7 at 1e-7.
TINY_R = (1e-8, 1e-7)

# A real constrained-decoding dictionary: root -> {1, 4}; 1 -> {2, 3}; 4 -> {5}.
TRIE_PATHS = [[1, 2], [1, 3], [4, 5]]


def tiny_r_adjust(raw, seen, r):
    """Independent longhand for the repetition term at a tiny r.

    r is used DIRECTLY -- never rebuilt as (r - 1) + 1, which is what cancels
    to 0.0 in fp32. Seen: negative logit * r, non-negative logit / r (so a
    zero logit stays exactly 0). Unseen: the logit itself (a scaling of 1.0).
    """
    if not seen:
        return raw
    return raw * r if raw < 0 else raw / r


def penalized_row(raw_row, counts_row, f, r):
    """The documented chain (frequency, then repetition; presence is off) for
    one row, straight off the count table the real state built."""
    out = []
    for v, raw in enumerate(raw_row):
        x = raw - f * counts_row[v]
        out.append(tiny_r_adjust(x, counts_row[v] > 0, r))
    return out


def logsumexp_list(xs):
    """fp64 log-sum-exp, shifted by the max so no magnitude can overflow."""
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class TestTinyRepetitionPenalty(CustomTestCase):
    """r = 1e-8 / 1e-7: finite, and scaled by r itself, over the real state."""

    def _state(self, r):
        state = BeamPenaltyState(repetition_penalty=r)
        # Init the count table the way the selection path does, then generate
        # three tokens on the single row: 1, 2, 3 are seen; 4, 5 are not.
        state.apply(torch.zeros(1, VOCAB))
        for tok in (1, 2, 3):
            state.advance(torch.tensor([0]), torch.tensor([tok]))
        return state

    def _row(self):
        # +SMALL_RAW seen, -SMALL_RAW seen, +0.0 seen, +SMALL_RAW unseen,
        # -SMALL_RAW unseen, over a low filler.
        row = torch.full((1, VOCAB), -6.0)
        row[0, 1] = SMALL_RAW
        row[0, 2] = -SMALL_RAW
        row[0, 3] = 0.0
        row[0, 4] = SMALL_RAW
        row[0, 5] = -SMALL_RAW
        return row

    def test_apply_scales_by_r_itself_and_stays_finite(self):
        for r in TINY_R:
            row = self._row()
            out = self._state(r).apply(row)
            self.assertTrue(
                bool(torch.isfinite(out).all()),
                msg=f"r={r} produced non-finite logits: {out.tolist()}",
            )
            for tok, seen in ((1, True), (2, True), (3, True), (4, False), (5, False)):
                want = tiny_r_adjust(float(row[0, tok]), seen, r)
                self.assertTrue(
                    math.isclose(float(out[0, tok]), want, rel_tol=1e-5),
                    msg=f"r={r} token {tok}: got {float(out[0, tok])}, want {want}",
                )
            # The scaling really bit: the seen positive logit sits ~raw / r
            # above its unseen twin (100 at r = 1e-8, 10 at 1e-7), not 0 above
            # it, which is what a silently-neutral scaling would give.
            self.assertGreater(float(out[0, 1]) - float(out[0, 4]), 1.0)

    def test_apply_leaves_the_source_row_untouched(self):
        for r in TINY_R:
            row = self._row()
            before = row.clone()
            self._state(r).apply(row)
            self.assertTrue(torch.equal(row, before))

    def test_the_scored_row_matches_the_oracle_over_the_full_vocab(self):
        for r in TINY_R:
            state = self._state(r)
            row = self._row()
            got = scored([row], state)[0]
            self.assertTrue(
                bool(torch.isfinite(got).all()), msg=f"r={r}: {got.tolist()}"
            )
            penalized = [
                tiny_r_adjust(float(row[0, v]), v in (1, 2, 3), r)
                for v in range(VOCAB)
            ]
            lse = logsumexp_list(penalized)
            for v in range(VOCAB):
                want = penalized[v] - lse
                self.assertTrue(
                    math.isclose(float(got[v]), want, rel_tol=1e-4, abs_tol=1e-4),
                    msg=f"r={r} token {v}: got {float(got[v])}, want {want}",
                )


class TestTinyPenaltyWithTrie(CustomTestCase):
    """A penalized step under a real BeamTrie: the mask is built before the
    selection, and the normalizer still runs over the penalized FULL vocab."""

    def _group(self, **pen):
        return BeamGroup(
            beam_width=2,
            stop_token_ids=[EOS],
            max_new_tokens=8,
            early_stopping=False,
            device="cpu",
            trie=BeamTrie.from_paths(TRIE_PATHS, device="cpu", eos_token_id=EOS),
            **pen,
        )

    def test_masked_step_stays_legal_and_uses_the_full_vocab_lse(self):
        group = self._group(frequency_penalty=0.25, repetition_penalty=1e-8)
        # Step 1 (prefill): counts are empty, so it is scored raw. The root
        # releases {1, 4} only, and the mask is built BEFORE the selection.
        mask1 = group.trie.build_mask(group.trie_cursor, VOCAB)
        self.assertEqual([v for v in range(VOCAB) if not bool(mask1[0, v])], [1, 4])
        probs1, toks1 = _rows_topk_logprobs(
            split_pieces(flat_logits([{1: 0.0, 4: -0.1}])),
            group.num_candidates,
            mask=mask1,
            penalty_state=group.penalty_state,
        )
        sel1 = joint_select(
            group.frontier_cum_logprobs,
            probs1,
            toks1,
            group.stop_token_ids,
            group.beam_width,
        )
        group.advance_frontier(sel1, 0)
        group.commit_pending()
        self.assertEqual(int(sel1.num_survivors), 2)
        self.assertEqual(sorted(int(t) for t in sel1.next_tokens[:2]), [1, 4])

        # Step 2: each row has generated one token, and that token is FORBIDDEN
        # here -- so it is penalized (raw / 1e-8) and then masked out, which is
        # exactly the case where a 0 scaling would poison the normalizer.
        mask2 = group.trie.build_mask(group.trie_cursor, VOCAB)
        self.assertEqual([v for v in range(VOCAB) if not bool(mask2[0, v])], [2, 3])
        self.assertEqual([v for v in range(VOCAB) if not bool(mask2[1, v])], [5])
        logits = flat_logits([{1: 1.0, 2: -1.0, 3: -2.0}, {4: 1.0, 5: -1.0}])
        probs, toks = _rows_topk_logprobs(
            split_pieces(logits),
            group.num_candidates,
            mask=mask2,
            penalty_state=group.penalty_state,
        )
        # No NaN anywhere: a poisoned normalizer would make every candidate
        # NaN. (-inf is legitimate here -- it is what a masked-out candidate
        # carries, and joint_select reads that as "not a real extension".)
        self.assertFalse(
            bool(torch.isnan(probs).any()), msg=f"NaN logprobs: {probs}"
        )
        sel = joint_select(
            group.frontier_cum_logprobs,
            probs,
            toks,
            group.stop_token_ids,
            group.beam_width,
        )
        # Every survivor is legal for the row it extends.
        for i in range(int(sel.num_survivors)):
            parent, token = int(sel.parent_idx[i]), int(sel.next_tokens[i])
            self.assertFalse(
                bool(mask2[parent, token]),
                msg=f"row {parent} selected forbidden token {token}",
            )
        # The normalizer is the lse of the PENALIZED row over the WHOLE vocab,
        # not a renormalization over the allowed set (masking runs after it).
        counts = group.penalty_state.counts.tolist()
        for row in range(logits.shape[0]):
            penalized = penalized_row(
                [float(x) for x in logits[row]], counts[row], 0.25, 1e-8
            )
            lse_full = logsumexp_list(penalized)
            lse_allowed = logsumexp_list(
                [p for v, p in enumerate(penalized) if not bool(mask2[row, v])]
            )
            # The forbidden-but-penalized token dominates, so the two really
            # differ: this check is not vacuous.
            self.assertGreater(abs(lse_full - lse_allowed), 1.0)
            top = int(toks[row, 0])
            self.assertTrue(math.isfinite(float(probs[row, 0])))
            self.assertTrue(
                math.isclose(
                    float(probs[row, 0]), penalized[top] - lse_full, rel_tol=1e-5
                ),
                msg=f"row {row}: got {float(probs[row, 0])}, want "
                f"{penalized[top] - lse_full}",
            )
            self.assertFalse(
                math.isclose(
                    float(probs[row, 0]), penalized[top] - lse_allowed, rel_tol=1e-3
                )
            )


class TestPenaltyOverlapAndRetire(CustomTestCase):
    """Two launched ticks, one commit: the counts must not move twice."""

    def test_two_ticks_commit_once_and_retire_drops_the_counts(self):
        group = make_group(frequency_penalty=F, presence_penalty=P, repetition_penalty=R)
        launch_step(group, flat_logits([{1: 0.0, 2: -0.1}]), tick=0)
        after_tick0 = group.penalty_state.counts.clone()
        self.assertEqual(int(after_tick0.sum()), group.beam_width)
        # Overlap: the launch half runs tick 1 before tick 0's deferred half.
        launch_step(
            group, flat_logits([{1: 0.0, 2: -0.05}, {2: 0.0, 1: -0.05}]), tick=1
        )
        after_tick1 = group.penalty_state.counts.clone()
        self.assertEqual(int(after_tick1.sum()), 2 * group.beam_width)
        self.assertFalse(torch.equal(after_tick0, after_tick1))

        # Only tick 0 reaches the DAG.
        group.commit_pending(up_to_tick=0)
        self.assertEqual(group.last_committed_tick, 0)
        self.assertTrue(torch.equal(group.penalty_state.counts, after_tick1))
        # ... and tick 1 follows: neither commit touched the counts.
        group.commit_pending()
        self.assertEqual(group.last_committed_tick, 1)
        self.assertTrue(torch.equal(group.penalty_state.counts, after_tick1))

        # The table is exactly the per-row multiset of the two generated
        # tokens: nothing was counted twice by the launches or the commits.
        expected = []
        for leaf in group.leaves:
            row = [0] * VOCAB
            for tok in materialize_tokens(leaf):
                row[tok] += 1
            expected.append(row)
        self.assertEqual(group.penalty_state.counts.tolist(), expected)

        coord = _coordinator()
        coord._num_live_groups += 1
        coord._retire_group(group)
        self.assertIsNone(group.penalty_state)


# --- wiring and lifecycle ----------------------------------------------------


class TestWiringAndLifecycle(CustomTestCase):
    def _validate(self, **pen):
        coord = _coordinator()
        params = SamplingParams(beam_width=2, max_new_tokens=8, n=1, **pen)
        req = SimpleNamespace(
            sampling_params=params,
            session_id=None,
            lora_id=None,
            origin_input_ids=[1, 2, 3],
            eos_token_ids=[2],
            tokenizer=None,
        )
        recv_req = SimpleNamespace(
            return_logprob=False,
            return_hidden_states=False,
            return_sampling_mask=False,
            return_routed_experts=False,
            session_params=None,
        )
        # The pre-SWA/mamba gates read the published runtime context; a unit
        # test has none, so pin them to their neutral values.
        with (
            mock.patch(
                "sglang.srt.beam_search.coordinator.get_disagg",
                lambda: SimpleNamespace(disaggregation_mode="null"),
            ),
            mock.patch(
                "sglang.srt.beam_search.coordinator.get_schedule",
                lambda: SimpleNamespace(page_size=1),
            ),
            mock.patch(
                "sglang.srt.beam_search.coordinator.get_memory",
                lambda: SimpleNamespace(
                    enable_hisparse=False, enable_hierarchical_cache=False
                ),
            ),
            mock.patch(
                "sglang.srt.beam_search.coordinator.get_parallel",
                lambda: SimpleNamespace(pp_size=1),
            ),
            mock.patch(
                "sglang.srt.beam_search.coordinator.is_dp_attention_enabled",
                lambda: False,
            ),
        ):
            error = coord.validate_and_init(req, recv_req)
        self.assertIsNone(error)
        return req

    def test_validate_and_init_keeps_the_user_penalties(self):
        req = self._validate(frequency_penalty=0.5, presence_penalty=0.25, repetition_penalty=1.75)
        state = req.beam_group.penalty_state
        self.assertIsNotNone(state)
        self.assertEqual(state.frequency_penalty, 0.5)
        self.assertEqual(state.presence_penalty, 0.25)
        self.assertEqual(state.repetition_penalty, 1.75)

    def test_the_leader_row_params_stay_neutral(self):
        req = self._validate(frequency_penalty=0.5, presence_penalty=0.25, repetition_penalty=1.75)
        # The leader's own sampler reads the neutral placeholder: it must never
        # apply the penalties itself.
        self.assertEqual(req.sampling_params.frequency_penalty, 0.0)
        self.assertEqual(req.sampling_params.presence_penalty, 0.0)
        self.assertEqual(req.sampling_params.repetition_penalty, 1.0)
        self.assertTrue(req.sampling_params.ignore_eos)

    def test_validate_and_init_without_penalties_creates_no_state(self):
        req = self._validate()
        self.assertIsNone(req.beam_group.penalty_state)

    def test_retire_drops_the_penalty_state(self):
        coord = _coordinator()
        group = make_group(frequency_penalty=F)
        launch_step(group, flat_logits([{1: 0.0, 2: -0.1}]))
        self.assertIsNotNone(group.penalty_state.counts)
        coord._num_live_groups += 1
        coord._retire_group(group)
        self.assertIsNone(group.penalty_state)

    def test_two_groups_never_share_counts(self):
        a = make_group(frequency_penalty=1.0)
        b = make_group(repetition_penalty=2.0)
        first = flat_logits([{1: 0.0, 2: -0.1}])
        step(a, first)
        # a's step left b's state untouched -- no shared table.
        self.assertIsNone(b.penalty_state.counts)
        step(b, first)
        self.assertIsNot(a.penalty_state.counts, b.penalty_state.counts)

        logits = flat_logits([{1: 0.0, 2: -0.2}, {2: 0.0, 1: -0.2}])
        scored_a = scored(split_pieces(logits), a.penalty_state)
        scored_b = scored(split_pieces(logits), b.penalty_state)
        self.assertFalse(torch.allclose(scored_a, scored_b, atol=1e-4))
        for group, pen in ((a, dict(f=1.0)), (b, dict(r=2.0))):
            expected = oracle_logprobs(logits.tolist(), [[1], [2]], **pen)
            for r in range(2):
                for v in range(VOCAB):
                    self.assertAlmostEqual(
                        float(scored(split_pieces(logits), group.penalty_state)[r, v]),
                        expected[r][v],
                        places=5,
                    )


if __name__ == "__main__":
    unittest.main()
