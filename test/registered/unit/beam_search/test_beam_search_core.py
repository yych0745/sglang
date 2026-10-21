"""Golden and differential tests for the beam search core.

Covers the pure selection functions (joint_select / select_final_topk), the
backpointer history DAG, and the BeamGroup lifecycle. The differential oracle is
a naive walk-in-order loop written here, not the implementation under test.
"""

import math
import random
import unittest

import msgspec
import torch

from sglang.srt.beam_search import (
    BeamGroup,
    BeamNode,
    CompletedBeam,
    joint_select,
    materialize_tokens,
    select_final_topk,
)
from sglang.srt.beam_search.beam_group import BeamGroupState
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def T(data, dtype):
    return torch.tensor(data, dtype=dtype)


def run_select(cum, logprobs, tokens, stop_ids, k):
    return joint_select(
        T(cum, torch.float32),
        T(logprobs, torch.float32),
        T(tokens, torch.int64),
        T(sorted(stop_ids), torch.int64),
        k,
    )


def reference_select(cum, logprobs, tokens, stop_ids, k):
    """Naive port of the original walk-in-order expansion loop."""
    num_candidates = len(logprobs[0])
    cands = [
        (cum[r] + logprobs[r][c], r, tokens[r][c])
        for r in range(len(cum))
        for c in range(num_candidates)
    ]
    cands.sort(key=lambda x: -x[0])
    cands = cands[:num_candidates]

    survivors, finished = [], []
    for score, row, token in cands:
        if token in stop_ids:
            finished.append((score, row, token))
        else:
            survivors.append((score, row, token))
            if len(survivors) == k:
                break
    return survivors, finished


def decode_sampling_params(**overrides) -> SamplingParams:
    """Decode a `SamplingParams` the way `/generate` does: it is
    `array_like=True`, so the wire form is a positional array. Start from the
    default's builtins (a full-length list) and set the slots we care about."""
    fields = list(SamplingParams.__struct_fields__)
    wire = msgspec.to_builtins(SamplingParams())
    for name, value in overrides.items():
        wire[fields.index(name)] = value
    return msgspec.convert(wire, SamplingParams)


def unpack(sel):
    ns, nf = int(sel.num_survivors), int(sel.num_finished)
    survivors = list(
        zip(
            sel.new_cum_logprobs[:ns].tolist(),
            sel.parent_idx[:ns].tolist(),
            sel.next_tokens[:ns].tolist(),
        )
    )
    finished = list(
        zip(
            sel.fin_cum_logprobs[:nf].tolist(),
            sel.fin_parent_idx[:nf].tolist(),
            sel.fin_tokens[:nf].tolist(),
        )
    )
    return survivors, finished


class TestJointSelectGolden(CustomTestCase):
    CUM = [0.0, -1.0]
    LOGPROBS = [[-0.1, -0.2, -0.3, -0.4], [-0.05, -0.5, -0.6, -0.7]]
    TOKENS = [[10, 11, 12, 13], [20, 21, 22, 23]]

    def assert_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for (a_score, a_row, a_tok), (e_score, e_row, e_tok) in zip(actual, expected):
            self.assertAlmostEqual(a_score, e_score, places=5)
            self.assertEqual((a_row, a_tok), (e_row, e_tok))

    def test_no_stop_fast_path(self):
        sel = run_select(self.CUM, self.LOGPROBS, self.TOKENS, set(), 2)
        survivors, finished = unpack(sel)
        self.assert_close(survivors, [(-0.1, 0, 10), (-0.2, 0, 11)])
        self.assertEqual(finished, [])

    def test_stop_routing(self):
        # Sorted walk: 10 survives, 11 is a stop (examined), 12 survives -> stop.
        # 13 and every row-1 candidate fall outside the examined window.
        sel = run_select(self.CUM, self.LOGPROBS, self.TOKENS, {11, 20}, 2)
        survivors, finished = unpack(sel)
        self.assert_close(survivors, [(-0.1, 0, 10), (-0.3, 0, 12)])
        self.assert_close(finished, [(-0.2, 0, 11)])

    def test_insufficient_survivors(self):
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2
        )
        survivors, finished = unpack(sel)
        self.assert_close(survivors, [(-0.4, 0, 8)])
        self.assert_close(finished, [(-0.1, 0, 5), (-0.2, 0, 6), (-0.3, 0, 7)])

    def test_all_stop(self):
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7, 8}, 2
        )
        survivors, finished = unpack(sel)
        self.assertEqual(survivors, [])
        self.assertEqual(len(finished), 4)

    def test_select_final_topk(self):
        sel = select_final_topk(
            T(self.CUM, torch.float32),
            T(self.LOGPROBS, torch.float32),
            T(self.TOKENS, torch.int64),
            2,
        )
        self.assertEqual(sel.tokens.tolist(), [10, 11])
        self.assertEqual(sel.parent_idx.tolist(), [0, 0])
        for actual, expected in zip(sel.cum_logprobs.tolist(), [-0.1, -0.2]):
            self.assertAlmostEqual(actual, expected, places=5)


class TestJointSelectDifferential(CustomTestCase):
    def test_random_vs_reference(self):
        rng = random.Random(42)
        torch.manual_seed(42)
        for trial in range(200):
            k = rng.choice([1, 2, 3, 5])
            num_rows = rng.choice([1, k])
            num_candidates = 2 * k
            vocab = list(range(100))

            logprobs = torch.randn(num_rows, num_candidates)
            tokens = [rng.sample(vocab, num_candidates) for _ in range(num_rows)]
            cum = [rng.uniform(-5, 0) for _ in range(num_rows)]
            stop_density = rng.choice([0.0, 0.3, 0.9, 1.0])
            stop_ids = {t for t in vocab if rng.random() < stop_density}

            sel = joint_select(
                T(cum, torch.float32),
                logprobs,
                T(tokens, torch.int64),
                T(sorted(stop_ids), torch.int64),
                k,
            )
            survivors, finished = unpack(sel)
            ref_survivors, ref_finished = reference_select(
                cum, logprobs.tolist(), tokens, stop_ids, k
            )

            msg = f"trial={trial} k={k} rows={num_rows} density={stop_density}"
            self.assertEqual(len(survivors), len(ref_survivors), msg)
            self.assertEqual(len(finished), len(ref_finished), msg)
            for actual, expected in zip(
                survivors + finished, ref_survivors + ref_finished
            ):
                self.assertAlmostEqual(actual[0], expected[0], places=4, msg=msg)
                self.assertEqual(actual[1:], expected[1:], msg)


class TestHistory(CustomTestCase):
    def test_materialize(self):
        a = BeamNode(1)
        b = BeamNode(2, a)
        c = BeamNode(3, a)  # reparent: sibling branch off the same prefix
        self.assertEqual(materialize_tokens(b), [1, 2])
        self.assertEqual(materialize_tokens(c), [1, 3])
        self.assertEqual(materialize_tokens(None), [])


class TestBeamGroup(CustomTestCase):
    def _make_group(self, **kwargs):
        defaults = dict(beam_width=2, stop_token_ids=[99], max_new_tokens=3)
        defaults.update(kwargs)
        return BeamGroup(**defaults)

    def test_lifecycle_eos_and_length_finish(self):
        group = self._make_group()

        # Prefill selection: single pseudo-row frontier.
        sel = run_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], {99}, 2)
        self.assertFalse(group.advance(sel))
        self.assertEqual(
            [materialize_tokens(leaf) for leaf in group.leaves], [[1], [2]]
        )

        # The best candidate is a stop token; two survivors remain.
        sel = run_select(
            [-0.1, -0.2],
            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[99, 5, 6, 7], [8, 9, 10, 11]],
            {99},
            2,
        )
        self.assertFalse(group.advance(sel))
        self.assertEqual(len(group.completed), 1)
        self.assertEqual(
            [materialize_tokens(leaf) for leaf in group.leaves], [[1, 5], [2, 8]]
        )

        # max_new_tokens reached: host decides, final top-k all finish.
        self.assertTrue(group.next_step_is_final())
        fsel = select_final_topk(
            group.frontier_cum_logprobs,
            T([[-0.1, -0.9], [-0.05, -0.9]], torch.float32),
            T([[12, 13], [14, 15]], torch.int64),
            2,
        )
        self.assertTrue(group.advance_final(fsel))

        results = group.finalize()
        self.assertEqual(len(results), 2)
        # EOS beam: cum -0.15 over 2 tokens -> score -0.075, the best.
        self.assertEqual(results[0].tokens, [1, 99])
        self.assertEqual(results[0].matched_token, 99)
        self.assertAlmostEqual(results[0].beam_score, -0.075, places=5)
        # Runner-up: [2, 8, 14] with cum -0.36 over 3 tokens -> -0.12.
        self.assertEqual(results[1].tokens, [2, 8, 14])
        self.assertIsNone(results[1].matched_token)
        self.assertAlmostEqual(results[1].beam_score, -0.12, places=5)

    def test_insufficient_survivors_finishes_group(self):
        group = self._make_group(stop_token_ids=[5, 6, 7])
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2
        )
        self.assertTrue(group.advance(sel))
        results = group.finalize()
        # Pool: three stop-finished beams + the folded-in partial survivor.
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].tokens, [5])
        self.assertAlmostEqual(results[0].beam_score, -0.1, places=5)

    def test_length_penalty_ordering(self):
        group = self._make_group(length_penalty=0.0)
        # penalty 0: score == cum_logprob regardless of length.
        self.assertAlmostEqual(group.beam_score(-0.3, 2), -0.3, places=6)
        group2 = self._make_group(length_penalty=1.0)
        self.assertAlmostEqual(group2.beam_score(-0.3, 2), -0.15, places=6)


class TestDummyBeamPadding(CustomTestCase):
    """Short steps keep the group alive with dummy leaves instead of finishing.

    Constrained decoding (trie) routinely has fewer than beam_width legal
    children at a layer; ending the request there would truncate every output.
    """

    def _make_group(self, **kwargs):
        defaults = dict(
            beam_width=2,
            stop_token_ids=[5, 6, 7],
            max_new_tokens=8,
            early_stopping=False,  # isolate padding from the EOS-stop shortcut
        )
        defaults.update(kwargs)
        return BeamGroup(**defaults)

    def test_one_survivor_pads_to_beam_width_and_keeps_decoding(self):
        group = self._make_group()
        # Three of four candidates are stop tokens: exactly one real survivor.
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2
        )
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        # Shape stays fixed: the dummy aliases the one real leaf.
        self.assertEqual(len(group.leaves), 2)
        self.assertEqual([materialize_tokens(l) for l in group.leaves], [[8], [8]])
        # Only the real EOS beams reached the pool; the dummy never does.
        self.assertEqual(len(group.completed), 3)
        self.assertTrue(all(b.matched_token is not None for b in group.completed))

    def test_dummy_carries_minus_inf_not_zero(self):
        # A padded slot must not look like a perfect score. -inf, not 0.0, is
        # what keeps a dummy from outranking every real candidate.
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2
        )
        scores = sel.new_cum_logprobs.tolist()
        self.assertAlmostEqual(scores[0], -0.4, places=5)
        self.assertEqual(scores[1], float("-inf"))

    def test_dummy_row_stop_candidate_is_not_published(self):
        # A dummy row's candidates all carry -inf, stop tokens included. When the
        # real row is itself partly masked (a short trie layer), the candidate
        # pool does contain those -inf stops and the examined window stays open,
        # so without a finiteness guard they would be published as finished
        # beams scored -inf.
        stop = {5, 6, 7}
        sel = joint_select(
            T([-0.4, float("-inf")], torch.float32),
            # Row 0: one real survivor plus two masked slots.
            # Row 1: a dummy row -- every candidate is -inf.
            T([[-0.6, float("-inf"), float("-inf")], [-0.1, -0.2, -0.3]], torch.float32),
            T([[20, 5, 6], [5, 6, 7]], torch.int64),
            T(sorted(stop), torch.int64),
            3,
        )
        self.assertEqual(int(sel.num_survivors), 1)
        self.assertEqual(
            int(sel.num_finished),
            0,
            "a -inf stop candidate from a dummy row must not count as finished",
        )

    def test_all_stop_still_finishes_the_group(self):
        # Zero survivors is the one case that must end the group: nothing legal
        # is left to extend.
        group = self._make_group()
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7, 8}, 2
        )
        self.assertTrue(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.FINISHED)

    def test_dummy_never_enters_the_returned_beams(self):
        # Drive a short step, then a final step whose top-k is padded by the
        # dummy. The -inf extension must be dropped from the result pool.
        group = self._make_group(max_new_tokens=2)
        group.advance(
            run_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 7, 8]], {5, 6, 7}, 2)
        )
        final = select_final_topk(
            group.frontier_cum_logprobs,
            T([[-0.1, -0.2, -0.3, -0.4], [-0.1, -0.2, -0.3, -0.4]], torch.float32),
            T([[10, 11, 12, 13], [10, 11, 12, 13]], torch.int64),
            2,
        )
        self.assertTrue(group.advance_final(final))
        results = group.finalize()
        # No padding artifact may surface as a scored beam.
        for r in results:
            self.assertTrue(
                math.isfinite(r.cum_logprob),
                f"dummy-derived beam leaked into results: {r}",
            )
        # The best real sequence wins; it is the EOS beam, not the dummy's [8, 10]
        # path reusing the survivor's row.
        self.assertEqual(results[0].tokens, [5])

    def test_early_stopping_counts_only_real_eos(self):
        # Padding must not inflate the completed counter. With beam_width=3 and
        # only two examined stop candidates, one short step is not enough to stop.
        group = BeamGroup(beam_width=3, stop_token_ids=[5, 6], max_new_tokens=8)
        sel = run_select(
            [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[5, 6, 9, 10]], {5, 6}, 3
        )
        self.assertEqual(int(sel.num_survivors), 2)  # two survivors, one dummy
        self.assertEqual(int(sel.num_finished), 2)
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.completed), 2)  # two EOS beams, no padding
        self.assertEqual(len(group.leaves), 3)  # shape held at beam_width


class TestLengthPenaltyWiring(CustomTestCase):
    """The API-facing half of length penalty: the request-level
    ``length_penalty`` must survive SamplingParams normalization and change which
    candidate BeamGroup.finalize ranks first.

    This pins the two ends of the wire (SamplingParams in, finalize ordering
    out). The coordinator's hand-off between them is a single keyword argument
    and is covered end-to-end by ``send_beam_official.py --length-penalty``
    against a live server, where the score shows up as ``sequence_score``.
    """

    def test_default_is_length_normalized(self):
        self.assertEqual(SamplingParams().length_penalty, 1.0)

    def test_explicit_values_survive_normalization(self):
        for lp in (0.0, 0.6, 1.0, 1.7, 3.0):
            self.assertEqual(SamplingParams(length_penalty=lp).length_penalty, lp)

    def test_decodes_from_a_request_dict(self):
        # /generate hands these over positionally (SamplingParams is
        # array_like), so decode a full-length array with our slots set.
        params = decode_sampling_params(beam_width=3, length_penalty=0.7)
        self.assertEqual(params.length_penalty, 0.7)
        self.assertEqual(params.beam_width, 3)

    def test_verify_accepts_the_whole_penalty_range(self):
        for lp in (0.0, 0.5, 1.0, 1.5, 3.0):
            SamplingParams(length_penalty=lp).verify(1024)

    def test_verify_rejects_non_finite_penalty(self):
        for lp in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                SamplingParams(length_penalty=lp).verify(1024)

    def _finished_group(self, length_penalty: float) -> BeamGroup:
        """A finished group holding two candidates: a short one with the better
        cum_logprob (-1.0 over 2 tokens) and a long one with the better
        per-token score (-1.2 over 4 tokens)."""
        group = BeamGroup(
            beam_width=2,
            length_penalty=length_penalty,
            stop_token_ids=(99,),
            max_new_tokens=8,
            device="cpu",
        )
        group.completed = [
            CompletedBeam(BeamNode(1, BeamNode(0)), -1.0, 2, 99),
            CompletedBeam(
                BeamNode(13, BeamNode(12, BeamNode(11, BeamNode(10)))), -1.2, 4, 99
            ),
        ]
        group.state = BeamGroupState.FINISHED
        return group

    def _top_length(self, length_penalty: float) -> int:
        return len(self._finished_group(length_penalty).finalize()[0].tokens)

    def test_no_penalty_ranks_the_shorter_beam_first(self):
        results = self._finished_group(0.0).finalize()
        self.assertEqual([len(r.tokens) for r in results], [2, 4])
        self.assertAlmostEqual(results[0].beam_score, -1.0, places=6)
        self.assertAlmostEqual(results[1].beam_score, -1.2, places=6)

    def test_unit_penalty_ranks_the_longer_beam_first(self):
        results = self._finished_group(1.0).finalize()
        self.assertEqual([len(r.tokens) for r in results], [4, 2])
        self.assertAlmostEqual(results[0].beam_score, -1.2 / 4, places=6)
        self.assertAlmostEqual(results[1].beam_score, -1.0 / 2, places=6)

    def test_ranking_flips_at_the_predicted_penalty(self):
        # Short wins while 1 / 2**lp < 1.2 / 4**lp, i.e. below lp = ln(1.2)/ln(2).
        flip = math.log(1.2) / math.log(2.0)
        self.assertEqual(self._top_length(0.0), 2)
        self.assertEqual(self._top_length(flip - 0.02), 2)
        self.assertEqual(self._top_length(flip + 0.02), 4)
        self.assertEqual(self._top_length(1.0), 4)
        self.assertEqual(self._top_length(2.0), 4)


class TestEarlyStoppingWiring(CustomTestCase):
    """The API-facing half of early stopping: the request-level ``early_stopping``
    must survive SamplingParams normalization and make BeamGroup stop as soon as
    the completed pool reaches beam_width, without running to max_new_tokens.

    Pins both ends of the wire (SamplingParams in, commit-side stop out). The
    coordinator hand-off is a single keyword argument, covered end-to-end by
    ``send_beam_official.py --no-early-stopping`` against a live server.
    """

    def test_default_is_on(self):
        self.assertIs(SamplingParams().early_stopping, True)

    def test_none_normalizes_to_on(self):
        # Non-optional field: an explicit null must fall back to the default.
        self.assertIs(SamplingParams(early_stopping=None).early_stopping, True)

    def test_explicit_values_survive_normalization(self):
        self.assertIs(SamplingParams(early_stopping=True).early_stopping, True)
        self.assertIs(SamplingParams(early_stopping=False).early_stopping, False)

    def test_decodes_from_a_request_dict(self):
        # /generate hands these over positionally (SamplingParams is
        # array_like), so decode a full-length array with our slots set.
        params = decode_sampling_params(beam_width=3, early_stopping=False)
        self.assertIs(params.early_stopping, False)
        self.assertEqual(params.beam_width, 3)

    def _group_ready_to_stop(self, early_stopping: bool) -> BeamGroup:
        """A live group one stop away from having a full completed pool."""
        group = BeamGroup(
            beam_width=2,
            stop_token_ids=(99,),
            max_new_tokens=8,
            early_stopping=early_stopping,
            device="cpu",
        )
        # Prefill: two live survivors, nothing completed yet.
        sel = run_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], {99}, 2)
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.completed, [])
        return group

    def _step_with_stop(self, group: BeamGroup) -> bool:
        # Best candidate per row is a stop token; both survivors keep decoding.
        sel = run_select(
            [-0.1, -0.2],
            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[99, 5, 6, 7], [99, 9, 10, 11]],
            {99},
            2,
        )
        return group.advance(sel)

    def test_stops_once_completed_pool_reaches_beam_width(self):
        group = self._group_ready_to_stop(early_stopping=True)
        # Two stop-finished beams land at once -> completed == beam_width -> stop.
        self.assertTrue(self._step_with_stop(group))
        self.assertEqual(group.state, BeamGroupState.FINISHED)
        self.assertIn(99, [r.matched_token for r in group.finalize()])

    def test_disabled_runs_on_past_the_completed_pool(self):
        group = self._group_ready_to_stop(early_stopping=False)
        # Same step, but early stopping is off: the group keeps decoding.
        self.assertFalse(self._step_with_stop(group))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.completed), 2)

    def test_fills_remaining_slots_before_stopping(self):
        # Only one stop finishes per step, so the pool is one short of k: the
        # judge must wait rather than stop on a partially filled pool.
        group = BeamGroup(
            beam_width=2,
            stop_token_ids=(99,),
            max_new_tokens=8,
            early_stopping=True,
            device="cpu",
        )
        sel = run_select([0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], {99}, 2)
        self.assertFalse(group.advance(sel))
        # One stop + two survivors: completed == 1 < beam_width -> keep going.
        sel = run_select(
            [-0.1, -0.2],
            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[99, 5, 6, 7], [8, 9, 10, 11]],
            {99},
            2,
        )
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.completed), 1)
        # Second stop fills the pool -> stop.
        sel = run_select(
            [
                group.frontier_cum_logprobs[0].item(),
                group.frontier_cum_logprobs[1].item(),
            ],
            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[99, 5, 6, 7], [8, 9, 10, 11]],
            {99},
            2,
        )
        self.assertTrue(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.FINISHED)
        self.assertGreaterEqual(len(group.completed), 2)


if __name__ == "__main__":
    unittest.main()
