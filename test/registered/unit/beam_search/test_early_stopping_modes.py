"""Beam search early stopping: the four modes, wired end to end.

`early_stopping` used to be a bool ("stop as soon as the completed pool fills,
and fold the live beams in as truncated results"). It is now one of
``"off" | "false" | "true" | "never"``:

  off    never stop early;
  true   stop once beam_width beams finished naturally (transformers
         `early_stopping=True`) -- and DROP the live beams;
  false  stop when the best live beam cannot beat the worst returned candidate
         scored at the length it has right now;
  never  the same bound taken at max_new_tokens while length_penalty > 0, so a
         beam that could still win is never cut.

The modes are pinned against a group whose completed pool is already full while
live beams remain -- the only situation where the judge has a decision to make.
"""

import unittest

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
from sglang.srt.beam_search.joint_select import SelectResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EOS = 99
ALL_MODES = ("off", "false", "true", "never")


def run_select(cum, logprobs, tokens, stop_ids, k):
    return joint_select(
        torch.tensor(cum, dtype=torch.float32),
        torch.tensor(logprobs, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.int64),
        torch.tensor(sorted(stop_ids), dtype=torch.int64),
        k,
    )


def _group(**kwargs) -> BeamGroup:
    defaults = dict(
        beam_width=2,
        stop_token_ids=(EOS,),
        max_new_tokens=8,
        length_penalty=1.0,
        device="cpu",
    )
    defaults.update(kwargs)
    return BeamGroup(**defaults)


def _prefill(group: BeamGroup) -> BeamGroup:
    """One unconstrained prefill selection: beam_width live beams, empty pool."""
    sel = run_select(
        [0.0], [[-0.1, -0.2, -0.3, -0.4]], [[1, 2, 3, 4]], {EOS}, group.beam_width
    )
    assert not group.advance(sel), "prefill must leave the group decoding"
    return group


def _sel(survivors, finished=(), k=2, num_candidates=4) -> SelectResult:
    """A hand-built SelectResult, for exact control of the candidate scores.

    Unused slots carry -inf exactly like joint_select's output, so a judge that
    reads a padded slot as "the best achievable continuation" is caught here.
    """
    toks = torch.zeros(k, dtype=torch.int64)
    pars = torch.zeros(k, dtype=torch.int64)
    cums = torch.full((k,), float("-inf"), dtype=torch.float32)
    for i, (cum, parent, tok) in enumerate(survivors):
        cums[i], pars[i], toks[i] = cum, parent, tok
    ftoks = torch.zeros(num_candidates, dtype=torch.int64)
    fpars = torch.zeros(num_candidates, dtype=torch.int64)
    fcums = torch.full((num_candidates,), float("-inf"), dtype=torch.float32)
    for i, (cum, parent, tok) in enumerate(finished):
        fcums[i], fpars[i], ftoks[i] = cum, parent, tok
    return SelectResult(
        next_tokens=toks,
        parent_idx=pars,
        new_cum_logprobs=cums,
        num_survivors=torch.tensor(len(survivors), dtype=torch.int64),
        fin_tokens=ftoks,
        fin_parent_idx=fpars,
        fin_cum_logprobs=fcums,
        num_finished=torch.tensor(len(finished), dtype=torch.int64),
    )


def _pool_full_step(group: BeamGroup) -> bool:
    """A real joint_select step that finishes beam_width EOS beams while two
    live beams (cum -0.3 and -0.6) survive it.

    With lp = 1 and length 2 the pool scores -0.075 and -0.155 (worst -0.155)
    and the best live beam scores -0.3 / 2 = -0.15 -- i.e. the heuristic modes
    are one hair away from stopping, which is the interesting case.
    """
    return group.advance(
        run_select(
            [-0.1, -0.2],
            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
            [[EOS, 5, 6, 7], [EOS, 9, 10, 11]],
            {EOS},
            2,
        )
    )


def _heuristic_group(mode, *, pool=(-1.0, -2.0), pool_len=4, **kwargs) -> BeamGroup:
    """A prefilled group whose completed pool is hand-set.

    `pool` holds the cum_logprobs of the pool entries, each `pool_len` tokens
    long. The defaults give scores -0.25 and -0.5 at lp = 1, so the bar any
    live beam has to clear (the beam_width-th best) is -0.5.
    """
    group = _prefill(_group(early_stopping=mode, **kwargs))
    group.completed = [
        CompletedBeam(BeamNode(1 + i, BeamNode(0)), cum, pool_len, EOS)
        for i, cum in enumerate(pool)
    ]
    return group


def _live_step(group: BeamGroup, best_cum: float) -> bool:
    """One step whose best live beam carries `best_cum`; generated length 2."""
    return group.advance(
        _sel([(best_cum, 0, 7), (best_cum - 0.5, 1, 8)], k=group.beam_width)
    )


def _step(group: BeamGroup, survivors, finished=()) -> bool:
    """One step with hand-set survivor scores and hand-set stop candidates.

    `finished` entries are branches that hit EOS this step, so they land in the
    completed pool *and* -- under the aligned judge -- feed `best_possible`.
    """
    return group.advance(
        _sel(
            survivors,
            finished=finished,
            k=group.beam_width,
            num_candidates=group.num_candidates,
        )
    )


class TestFourModes(CustomTestCase):
    """The pooled-full-with-live-beams scenario, once per mode."""

    def test_off_never_stops(self):
        group = _prefill(_group(early_stopping="off"))
        self.assertFalse(_pool_full_step(group))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.completed), 2)
        self.assertEqual(len(group.leaves), 2)

    def test_true_stops(self):
        group = _prefill(_group(early_stopping="true"))
        self.assertTrue(_pool_full_step(group))
        self.assertEqual(group.state, BeamGroupState.FINISHED)

    def test_heuristic_modes_do_not_stop_a_hair_short(self):
        # best live -0.3 / 2 = -0.15 vs worst finished -0.155: the live beam can
        # still take a slot, so both heuristic modes keep decoding.
        for mode in ("false", "never"):
            with self.subTest(mode=mode):
                group = _prefill(_group(early_stopping=mode))
                self.assertFalse(_pool_full_step(group))
                self.assertEqual(group.state, BeamGroupState.DECODING)


class TestTrueDiscardsLiveBeams(CustomTestCase):
    """Core regression: the old judge folded the live beams into `completed`.

    Early stopping asserts that what we already hold cannot be improved upon, so
    publishing an unfinished beam there would seed the result set with a
    truncated sequence.
    """

    def test_completed_is_exactly_the_pool_before_the_step(self):
        group = _prefill(_group(early_stopping="true"))
        self.assertEqual(group.completed, [])
        self.assertTrue(_pool_full_step(group))
        # Two EOS beams, no third entry from the two live beams.
        self.assertEqual(len(group.completed), 2)
        self.assertEqual([b.matched_token for b in group.completed], [EOS, EOS])
        self.assertEqual(group.leaves, [])

    def test_no_live_beam_reaches_the_results(self):
        group = _prefill(_group(early_stopping="true"))
        self.assertTrue(_pool_full_step(group))
        results = group.finalize()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.matched_token, EOS)
            self.assertEqual(r.tokens[-1], EOS)
        # The survivors of the step were tokens 5 and 6; neither may appear as
        # the last token of a returned sequence.
        self.assertNotIn(5, [r.tokens[-1] for r in results])
        self.assertNotIn(6, [r.tokens[-1] for r in results])

    def test_off_keeps_the_live_beams_out_of_the_pool_too(self):
        # Same step, no early stop: the survivors stay live, not completed.
        group = _prefill(_group(early_stopping="off"))
        self.assertFalse(_pool_full_step(group))
        self.assertEqual([b.matched_token for b in group.completed], [EOS, EOS])
        self.assertEqual([materialize_tokens(l)[-1] for l in group.leaves], [5, 6])


class TestHeuristicBound(CustomTestCase):
    """`best_possible <= worst_finished`, with `bound_len` split by mode."""

    def test_false_stops_when_the_bound_is_met_exactly(self):
        # worst finished -0.5; best live -1.0 at length 2 -> -0.5 == -0.5.
        self.assertTrue(_live_step(_heuristic_group("false"), -1.0))

    def test_false_keeps_going_a_hair_below_the_bound(self):
        # -0.998 / 2 = -0.499 > -0.5: the live beam can still win.
        self.assertFalse(_live_step(_heuristic_group("false"), -0.998))

    def test_never_and_false_disagree_only_on_the_bound(self):
        # Same numbers, different bound: "never" scores the live beam at
        # max_new_tokens (8) instead of the current length (2), so -1.0 / 8
        # still beats -0.5 and only a far worse frontier stops it.
        self.assertTrue(_live_step(_heuristic_group("false"), -1.0))
        self.assertFalse(_live_step(_heuristic_group("never"), -1.0))
        # -4.0 / 8 == -0.5: even the longest possible beam cannot win.
        self.assertTrue(_live_step(_heuristic_group("never"), -4.0))
        self.assertFalse(_live_step(_heuristic_group("never"), -3.99))

    def test_never_degenerates_to_false_when_penalty_is_not_positive(self):
        # lp = 0: score == cum_logprob, so longer is never better and the
        # current length is already the upper bound. worst = -2.0 here.
        for mode in ("false", "never"):
            with self.subTest(mode=mode):
                self.assertTrue(
                    _live_step(_heuristic_group(mode, length_penalty=0.0), -2.0)
                )
                self.assertFalse(
                    _live_step(_heuristic_group(mode, length_penalty=0.0), -1.9)
                )

    def test_judge_reuses_the_length_penalty_of_the_final_ranking(self):
        # The judge must score with `beam_score`, the same length penalty that
        # orders the final results. With lp = 2 the bar is -2/4**2 = -0.125 and
        # a live beam of length 2 is divided by 4: -0.75 / 4 = -0.1875 <= -0.125
        # -> stop. The very same numbers under lp = 1 (bar -0.5, divisor 2) give
        # -0.375 > -0.5 -> keep going, which is what pins the sharing.
        self.assertTrue(
            _live_step(_heuristic_group("false", length_penalty=2.0), -0.75)
        )
        self.assertFalse(
            _live_step(_heuristic_group("false", length_penalty=2.0), -0.4)
        )
        self.assertFalse(
            _live_step(_heuristic_group("false", length_penalty=1.0), -0.75)
        )


class TestBestPossibleIsTheWholeStep(CustomTestCase):
    """`best_possible` is the best candidate of the *whole* step -- the ones
    that hit EOS included -- not just the best survivor.

    The internal judge scores `expanded_cum_logprobs[0]`, the head of the
    descending topk over every candidate the step expanded. A step's
    highest-scoring candidate is very often an EOS: that branch already ended,
    so letting it raise `best_possible` makes the bound looser and the stop
    strictly later than a survivor-only bound would make it. Looser is the
    alignment target here; every case below turns on a stop candidate that
    outranks every survivor.
    """

    def test_a_top_scoring_stop_candidate_raises_the_bound(self):
        # Pool (-1.0, -2.0) at length 4 -> scores -0.25 / -0.5. This step's EOS
        # candidate (-0.1 at length 2, score -0.05) joins the pool and pushes
        # -0.5 out of the top beam_width, so the bar is -0.25.
        #
        #   survivor-only bound:  -1.0 / 2 = -0.5  <= -0.25 -> stop
        #   whole-step bound:     -0.1 / 2 = -0.05 >  -0.25 -> keep decoding
        #
        # The second is the internal behaviour, and it is what we assert.
        group = _heuristic_group("false")
        self.assertFalse(_step(group, [(-1.0, 0, 7), (-1.5, 1, 8)], [(-0.1, 0, EOS)]))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.completed), 3)
        self.assertEqual(len(group.leaves), 2)  # live beams survive the step

    def test_without_that_stop_candidate_the_same_step_stops(self):
        # Contrast for the case above: drop the high-scoring EOS and the bar is
        # back to -0.5 with nothing to raise the bound, so the best survivor
        # (-1.0 / 2 = -0.5) meets it exactly. This is the survivor-only
        # verdict -- it is what the whole-step judge must *not* produce.
        self.assertTrue(_step(_heuristic_group("false"), [(-1.0, 0, 7), (-1.5, 1, 8)]))

    def test_a_stop_candidate_below_the_best_survivor_changes_nothing(self):
        # The step's best candidate is a survivor, so folding the stop side in
        # cannot move `best_possible`; both directions of the verdict must be
        # exactly what they were with no stop candidate at all.
        for finished in ((), ((-1.0, 0, EOS),)):
            with self.subTest(finished=finished):
                group = _heuristic_group("false")
                self.assertFalse(
                    _step(group, [(-0.1, 0, 7), (-0.6, 1, 8)], finished)
                )
                self.assertEqual(group.state, BeamGroupState.DECODING)
        for finished in ((), ((-2.0, 0, EOS),)):
            with self.subTest(finished=finished):
                self.assertTrue(
                    _step(_heuristic_group("false"), [(-1.0, 0, 7), (-1.5, 1, 8)], finished)
                )

    def test_dummy_stop_candidates_carry_no_score(self):
        # A dummy row's stop candidate is -inf: joint_select never publishes it
        # as a finished beam, and every unused finished slot is padded -inf.
        # It must neither raise the bound nor fill a pool slot.
        sel = _sel([(-0.5, 0, 7), (-1.0, 1, 8)], [(float("-inf"), 0, EOS)])
        self.assertEqual(float(sel.fin_cum_logprobs.max()), float("-inf"))
        group = _heuristic_group("false")
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.completed), 3)
        self.assertEqual(len(group._completed_beam_scores()), 2)  # -inf dropped
        # Bar stays -0.5 and the best survivor scores -0.25: same verdict as a
        # step with no stop candidate whatsoever.
        self.assertFalse(_step(_heuristic_group("false"), [(-0.5, 0, 7), (-1.0, 1, 8)]))

    def test_a_step_with_no_stop_candidate_degenerates_to_the_survivor_bound(self):
        # num_finished == 0: the whole stop side is -inf padding, so the max
        # over it is the best survivor and the verdict is unchanged -- both
        # across the threshold and one hair short of it.
        self.assertTrue(_step(_heuristic_group("false"), [(-1.0, 0, 7), (-1.5, 1, 8)]))
        self.assertFalse(
            _step(_heuristic_group("false"), [(-0.998, 0, 7), (-1.5, 1, 8)])
        )

    def test_false_and_never_still_differ_only_in_the_bound(self):
        # The step's best candidate is the EOS, so both modes score the *same*
        # cum_logprob and only `bound_len` separates them -- which is what pins
        # where the value came from.

        def step(mode, stop_cum, surv_cum):
            return _step(
                _heuristic_group(mode),
                [(surv_cum, 0, 7), (surv_cum - 0.5, 1, 8)],
                [(stop_cum, 0, EOS)],
            )

        # Best candidate EOS -1.0, best survivor -2.0, bar -0.5:
        #   "false" -1.0 / 2 = -0.5      <= -0.5 -> stop
        #   "never" -1.0 / 8 = -0.125    >  -0.5 -> keep going
        self.assertTrue(step("false", -1.0, -2.0))
        self.assertFalse(step("never", -1.0, -2.0))
        # One hair from the "never" bar: -3.99 / 8 = -0.49875 > -0.5, so no
        # stop -- while a survivor-only bound (-8.0 / 8 = -1.0) would have
        # stopped. -4.0 / 8 = -0.5 meets it exactly.
        self.assertFalse(step("never", -3.99, -8.0))
        self.assertTrue(step("never", -4.0, -8.0))


class TestJudgePreconditions(CustomTestCase):
    def test_no_mode_stops_while_the_pool_is_short(self):
        # One EOS beam only: a live beam can still take the empty slot, so not
        # even "true" stops here.
        for mode in ALL_MODES:
            with self.subTest(mode=mode):
                group = _prefill(_group(early_stopping=mode))
                self.assertFalse(
                    group.advance(
                        run_select(
                            [-0.1, -0.2],
                            [[-0.05, -0.2, -0.5, -0.9], [-0.11, -0.4, -0.8, -1.2]],
                            [[EOS, 5, 6, 7], [8, 9, 10, 11]],
                            {EOS},
                            2,
                        )
                    )
                )
                self.assertEqual(group.state, BeamGroupState.DECODING)
                self.assertEqual(len(group.completed), 1)

    def test_pool_entries_without_a_score_do_not_fill_a_slot(self):
        # Two pool entries, one of them carrying no usable score: the threshold
        # has a single real entry, short of beam_width.
        for mode in ("false", "never"):
            with self.subTest(mode=mode):
                group = _heuristic_group(mode, pool=(-1.0, float("-inf")))
                self.assertEqual(len(group.completed), 2)
                self.assertFalse(_live_step(group, -1.0))
                self.assertEqual(group.state, BeamGroupState.DECODING)

    def test_a_scoreless_entry_does_not_disturb_a_real_threshold(self):
        # A pool of three where one entry has no score: the bar is still the
        # beam_width-th best of the real ones, so the stop still fires.
        group = _heuristic_group("false", pool=(-1.0, -2.0, float("-inf")))
        self.assertEqual(len(group.completed), 3)
        self.assertTrue(_live_step(group, -1.0))


class TestPaddingIsNotAContinuableBranch(CustomTestCase):
    """A dummy-padded slot is -inf; reading it must not stop the group."""

    def test_padded_slot_is_not_the_best_continuation(self):
        # worst finished -1.0; the real survivor (-0.5 / 2 = -0.25) still beats
        # it, while a -inf slot would score -inf and stop any group.
        group = _heuristic_group("false", pool=(-4.0, -4.0))
        sel = _sel([(-0.5, 0, 7)], k=2)
        self.assertEqual(float(sel.new_cum_logprobs[1]), float("-inf"))
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.DECODING)
        self.assertEqual(len(group.leaves), 2)  # dummy seated to hold the shape

    def test_short_trie_layer_keeps_the_group_alive(self):
        # Constrained layer with a single legal child: row 1 is a dummy seated
        # to keep the batch shape and every candidate it expands is -inf.
        group = _group(early_stopping="false")
        self.assertFalse(
            group.advance(
                joint_select(
                    torch.tensor([0.0], dtype=torch.float32),
                    torch.tensor(
                        [[-0.4, float("-inf"), float("-inf")]], dtype=torch.float32
                    ),
                    torch.tensor([[20, EOS, EOS + 1]], dtype=torch.int64),
                    torch.tensor([EOS], dtype=torch.int64),
                    2,
                )
            )
        )
        self.assertEqual(len(group.leaves), 2)

        group.completed = [
            CompletedBeam(BeamNode(1, BeamNode(0)), -8.0, 4, EOS),
            CompletedBeam(BeamNode(2, BeamNode(0)), -8.0, 4, EOS),
        ]
        sel = joint_select(
            group.frontier_cum_logprobs,
            torch.tensor(
                [[-0.6, float("-inf"), float("-inf")], [-0.1, -0.2, -0.3]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[20, EOS, EOS + 1], [EOS, EOS + 1, EOS + 2]], dtype=torch.int64
            ),
            torch.tensor([EOS], dtype=torch.int64),
            2,
        )
        self.assertEqual(int(sel.num_survivors), 1)
        self.assertEqual(float(sel.new_cum_logprobs[1]), float("-inf"))
        # -1.0 / 2 = -0.5 still beats the -2.0 bar: no stop.
        self.assertFalse(group.advance(sel))
        self.assertEqual(group.state, BeamGroupState.DECODING)


class TestLengthTruncationIsUnaffected(CustomTestCase):
    """The two terminal paths must stay distinct: truncation folds survivors in,
    early stopping drops them. Both are pinned here under the same mode."""

    def test_final_step_folds_its_survivors_in(self):
        group = _group(early_stopping="true", max_new_tokens=2)
        _prefill(group)
        self.assertTrue(
            group.advance_final(
                select_final_topk(
                    group.frontier_cum_logprobs,
                    torch.tensor([[-0.1, -0.9], [-0.05, -0.9]], dtype=torch.float32),
                    torch.tensor([[12, 13], [14, 15]], dtype=torch.int64),
                    2,
                )
            )
        )
        self.assertEqual(len(group.completed), 2)
        self.assertTrue(all(b.matched_token is None for b in group.completed))
        results = group.finalize()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.matched_token is None for r in results))

    def test_the_same_mode_drops_them_on_the_early_stop_path(self):
        group = _prefill(_group(early_stopping="true", max_new_tokens=2))
        self.assertTrue(_pool_full_step(group))
        self.assertEqual(len(group.completed), 2)
        self.assertTrue(all(b.matched_token == EOS for b in group.completed))


class TestWireCompatibility(CustomTestCase):
    def test_bools_still_map_to_modes(self):
        self.assertEqual(SamplingParams(early_stopping=True).early_stopping, "true")
        self.assertEqual(SamplingParams(early_stopping=False).early_stopping, "off")
        self.assertEqual(
            _group(early_stopping=True).early_stopping,
            "true",
        )
        self.assertEqual(
            _group(early_stopping=False).early_stopping,
            "off",
        )

    def test_bools_still_behave_as_they_did(self):
        self.assertTrue(_pool_full_step(_prefill(_group(early_stopping=True))))
        self.assertFalse(_pool_full_step(_prefill(_group(early_stopping=False))))

    def test_modes_are_case_and_space_insensitive(self):
        self.assertEqual(SamplingParams(early_stopping=" NEVER ").early_stopping, "never")
        self.assertEqual(_group(early_stopping="True").early_stopping, "true")

    def test_unknown_value_is_rejected(self):
        for bad in ("sometimes", "yes", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    SamplingParams(early_stopping=bad)
                with self.assertRaises(ValueError):
                    _group(early_stopping=bad)
        # The group is a plain class, so it also sees non-string callers.
        for bad in (3, 1.0, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _group(early_stopping=bad)


class TestDefaultIsOff(CustomTestCase):
    """BREAKING CHANGE: the default used to be True (== "true")."""

    def test_sampling_params_default(self):
        self.assertEqual(SamplingParams().early_stopping, "off")

    def test_none_normalizes_to_the_default(self):
        self.assertEqual(SamplingParams(early_stopping=None).early_stopping, "off")

    def test_group_default(self):
        self.assertEqual(_group().early_stopping, "off")

    def test_default_runs_a_full_pool_to_max_new_tokens(self):
        group = _prefill(_group())
        self.assertEqual(group.early_stopping, "off")
        self.assertFalse(_pool_full_step(group))
        self.assertEqual(group.state, BeamGroupState.DECODING)


if __name__ == "__main__":
    unittest.main()
