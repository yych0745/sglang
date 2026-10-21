"""Unit tests for the write side of the beam mamba copy-on-write double buffer.

The read side (pool -> backend -> kernel) and the state machine are covered
elsewhere; this file covers what had no caller before: the beam prune arming
the COW, and the two guards that keep it from being silently dropped.

  - order:    the prune must set the read mapping BEFORE it hands the rows
              fresh slots, or every row reads the slot it is about to write;
  - late free: the detached old slots come back one tick later, never in the
              tick that detached them;
  - admission: mamba still refuses beam search without the switch, SWA always
              refuses it;
  - graph:    a batch carrying beam rows never replays a cuda graph, and if a
              mapping ever does reach a replay the backend says so loudly.

Pure CPU tensor bookkeeping: the pool is built with only the mamba accounting
the methods touch, so the real method bodies run.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_member_row_mamba import _CpuHybridPool

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _coordinator(pool, **kwargs):
    """A real BeamCoordinator with only what the mamba path touches."""
    return BeamCoordinator(
        model_config=SimpleNamespace(vocab_size=1000, is_encoder_decoder=False),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        dllm_enabled=False,
        max_req_len=64,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=None,
        tree_cache=SimpleNamespace(supports_swa=lambda: False, supports_mamba=lambda: False),
        future_map=SimpleNamespace(stash=lambda *a, **k: None),
        **kwargs,
    )


class TestPruneCowOrder(CustomTestCase):
    """set -> replace -> stage, in that order, with no host sync in between."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=8, mamba_size=8)
        # A beam_width=4 group: the leader's row plus three member rows.
        self.rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.owned = self.pool.alloc_mamba_for_rows(self.rows)
        self.coord = _coordinator(self.pool)

    def test_rows_own_consecutive_slots_before_the_prune(self):
        # Fixture sanity: slots come off the front of the free list, so the
        # expectations below can name them.
        self.assertEqual(self.owned.tolist(), [1, 2, 3, 4])

    def test_read_mapping_names_the_parents_old_slots(self):
        # Survivor i continues from row parent_idx[i]; the parent's slot at
        # prune time is the row's slot BEFORE any of them were replaced.
        parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        read = self.pool.get_beam_cow_read_mapping(self.rows)
        self.assertEqual(read.tolist(), [2, 2, 1, 3])

    def test_rows_write_slots_they_did_not_read(self):
        # The whole point of the double buffer. If the read mapping were armed
        # after the replacement, `read` would equal `new` and the kernel would
        # read the slot it is writing -- a silent no-op copy, i.e. every beam
        # would restart from empty state.
        parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        read = self.pool.get_beam_cow_read_mapping(self.rows)
        new = self.pool.get_mamba_indices(self.rows)
        self.assertEqual(new.tolist(), [5, 6, 7, 8])
        # Disjoint, so no row reads its own write target.
        self.assertEqual(set(read.tolist()) & set(new.tolist()), set())
        self.assertTrue(bool((read != new).all()))

    def test_the_leader_row_is_replaced_too(self):
        # Row 0 is the leader. Rows 1..3 all continue from it, so they read the
        # leader's slot: if the leader kept that slot it would write it in the
        # very launch that reads it, and three beams would race it. The leader
        # therefore needs a fresh slot exactly like a member row.
        parent_idx = torch.tensor([1, 0, 0, 0], dtype=torch.int64)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        new = self.pool.get_mamba_indices(self.rows)
        self.assertNotEqual(int(new[0]), int(self.owned[0]))
        # ...and the three survivors really do read the leader's old slot.
        read = self.pool.get_beam_cow_read_mapping(self.rows)
        self.assertEqual(read[1:].tolist(), [1, 1, 1])

    def test_a_self_parented_row_still_moves(self):
        # parent_idx[i] == i means "this beam keeps its own history" -- but its
        # slot still has to be replaced, because another row may name it as a
        # parent. Reading your own OLD slot while writing a new one is exactly
        # the self-copy.
        parent_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        read = self.pool.get_beam_cow_read_mapping(self.rows)
        new = self.pool.get_mamba_indices(self.rows)
        self.assertEqual(read.tolist(), self.owned.tolist())
        self.assertEqual(set(read.tolist()) & set(new.tolist()), set())

    def test_two_groups_pruning_in_one_tick_share_one_snapshot(self):
        # The second group's set must reuse the snapshot the first one made.
        # Re-snapshotting would pick up the first group's freshly installed
        # slots and hand them back as read sources.
        rows_a = torch.tensor([1, 2], dtype=torch.int64)
        rows_b = torch.tensor([3, 4], dtype=torch.int64)

        self.coord._apply_mamba_cow(rows_a, torch.tensor([1, 0], dtype=torch.int64))
        self.coord._apply_mamba_cow(rows_b, torch.tensor([1, 0], dtype=torch.int64))

        read = self.pool.get_beam_cow_read_mapping(self.rows)
        self.assertEqual(read.tolist(), [2, 1, 4, 3])

    def test_a_non_hybrid_pool_is_a_no_op(self):
        # A plain pool has no mamba table at all; the prune must not care.
        coord = _coordinator(object())

        self.assertIsNone(
            coord._apply_mamba_cow(self.rows, torch.tensor([0, 0, 0, 0], dtype=torch.int64))
        )

    def test_no_rows_does_nothing(self):
        empty = torch.zeros(0, dtype=torch.int64)

        self.coord._apply_mamba_cow(empty, empty)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertEqual(self.pool.mamba_allocator.available_size(), 4)


class TestSourceSlotsComeBackOneTickLate(CustomTestCase):
    """Staging is not freeing: the kernel still reads those slots this tick."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=8, mamba_size=8)
        self.rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.owned = self.pool.alloc_mamba_for_rows(self.rows)
        self.coord = _coordinator(self.pool)
        self.parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)

    def test_the_detached_slots_do_not_return_in_the_pruning_tick(self):
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)

        # 8 slots, 4 owned before, 4 more handed out by the replacement: the
        # four detached ones are staged, so nothing is available yet. Handing
        # them back now would let the next allocation reissue a slot this
        # forward is still reading.
        self.assertEqual(self.pool.mamba_allocator.available_size(), 0)
        self.assertNotIn(1, self.pool.mamba_allocator.free_slots.tolist())

    def test_the_relay_hook_returns_them_on_the_next_forward(self):
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)

        # The relay hook runs once per forward, after the forward that read the
        # staged slots has been launched -- which is the earliest safe point.
        self.coord.maybe_select_and_relay(None, None)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 4)
        self.assertEqual(sorted(self.pool.mamba_allocator.free_slots.tolist()), [1, 2, 3, 4])
        # The rows keep the fresh slots they were given.
        self.assertEqual(self.pool.get_mamba_indices(self.rows).tolist(), [5, 6, 7, 8])

    def test_the_relay_hook_runs_the_flush_even_with_no_live_group(self):
        # The last prune of a group that has since been retired must not leak:
        # the flush sits above the `_num_live_groups == 0` early return.
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)
        self.coord._num_live_groups = 0

        self.coord.maybe_select_and_relay(None, None)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 4)

    def test_a_second_flush_frees_nothing_more(self):
        # Take-once: returning the same ids twice would let the allocator issue
        # one slot to two requests.
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)
        self.coord.maybe_select_and_relay(None, None)
        free_after_first = self.pool.mamba_allocator.free_slots.tolist()

        self.coord.maybe_select_and_relay(None, None)

        self.assertEqual(self.pool.mamba_allocator.free_slots.tolist(), free_after_first)
        self.assertNotIn(0, self.pool.mamba_allocator.free_slots.tolist())

    def test_every_slot_is_back_after_a_prune_cycle(self):
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)
        self.coord.maybe_select_and_relay(None, None)

        self.pool.free_mamba_for_rows(self.rows)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 8)
        self.assertEqual(
            sorted(self.pool.mamba_allocator.free_slots.tolist()), list(range(1, 9))
        )


class TestPruneRefusesAShortAllocator(CustomTestCase):
    """A prune that cannot get its fresh slots must fail loudly, and must not
    leave a half-armed read mapping behind."""

    def setUp(self):
        self.rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    def test_short_allocator_raises_before_arming_anything(self):
        pool = _CpuHybridPool(size=8, mamba_size=5)
        pool.alloc_mamba_for_rows(self.rows)
        coord = _coordinator(pool)

        with self.assertRaises(RuntimeError):
            coord._apply_mamba_cow(self.rows, torch.tensor([0, 0, 0, 0], dtype=torch.int64))

        # No mapping was armed, so nothing can be silently mis-consumed later.
        self.assertFalse(pool.beam_cow_read_mapping_pending())
        # And the rows keep the slots they had, not a silent 0.
        self.assertEqual(pool.get_mamba_indices(self.rows).tolist(), [1, 2, 3, 4])

    def test_the_message_names_the_switch(self):
        pool = _CpuHybridPool(size=8, mamba_size=5)
        pool.alloc_mamba_for_rows(self.rows)
        coord = _coordinator(pool)

        with self.assertRaises(RuntimeError) as ctx:
            coord._apply_mamba_cow(self.rows, torch.tensor([0, 0, 0, 0], dtype=torch.int64))

        self.assertIn("--enable-beam-mamba-double-buffer", str(ctx.exception))


class TestMambaAdmissionGate(CustomTestCase):
    """A: SWA always refuses; mamba refuses unless the switch is on."""

    def _req(self):
        params = SamplingParams(beam_width=4, max_new_tokens=8, n=1)
        params.stop_strs = []
        params.stop_regex_strs = []
        return (
            SimpleNamespace(
                sampling_params=params,
                session_id=None,
                lora_id=None,
                origin_input_ids=[1, 2, 3],
                eos_token_ids=[2],
                tokenizer=None,
            ),
            SimpleNamespace(
                return_logprob=False,
                return_hidden_states=False,
                return_sampling_mask=False,
                return_routed_experts=False,
                session_params=None,
            ),
        )

    def _validate(self, *, swa, mamba, double_buffer=False):
        coord = BeamCoordinator(
            model_config=SimpleNamespace(vocab_size=1000, is_encoder_decoder=False),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            dllm_enabled=False,
            max_req_len=64,
            req_to_token_pool=SimpleNamespace(size=64, device="cpu"),
            token_to_kv_pool_allocator=None,
            tree_cache=SimpleNamespace(
                supports_swa=lambda: swa, supports_mamba=lambda: mamba
            ),
            future_map=None,
            enable_beam_mamba_double_buffer=double_buffer,
        )
        req, recv_req = self._req()
        # The checks before the SWA/mamba gate read the published runtime
        # context; a unit test has none, so pin them to their neutral values.
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
            return coord.validate_and_init(req, recv_req)

    def test_mamba_is_refused_when_the_switch_is_off(self):
        error = self._validate(swa=False, mamba=True)

        self.assertIsNotNone(error)
        self.assertIn("mamba", error)
        self.assertIn("--enable-beam-mamba-double-buffer", error)

    def test_mamba_is_allowed_when_the_switch_is_on(self):
        self.assertIsNone(self._validate(swa=False, mamba=True, double_buffer=True))

    def test_swa_is_refused_even_with_the_switch_on(self):
        error = self._validate(swa=True, mamba=False, double_buffer=True)

        self.assertIsNotNone(error)
        self.assertIn("SWA", error)

    def test_swa_wins_over_mamba_when_both_are_reported(self):
        error = self._validate(swa=True, mamba=True, double_buffer=True)

        self.assertIsNotNone(error)
        self.assertIn("SWA", error)

    def test_a_plain_cache_is_unaffected(self):
        self.assertIsNone(self._validate(swa=False, mamba=False))
        self.assertIsNone(self._validate(swa=False, mamba=False, double_buffer=True))


class TestDecodeGraphRefusesBeamBatches(CustomTestCase):
    """C: a beam decode batch must never replay a captured graph."""

    @staticmethod
    def _runner():
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.ragged_verify_mode = False
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = False
        runner.is_encoder_decoder = False
        runner.enable_two_batch_overlap = False
        runner.disable_padding = False
        runner.max_bs = 64
        runner.enable_pdmux = False
        runner.record_nolora_graph = False
        runner.attention_graph_variants = None
        runner.captured_req_width = 1
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
        )
        return runner

    @staticmethod
    def _forward_batch(beam_num_tail_rows):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return SimpleNamespace(
            replace_embeds=None,
            beam_num_tail_rows=beam_num_tail_rows,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            batch_size=4,
            input_ids=None,
            lora_ids=None,
            can_run_tbo=False,
            can_run_decode_cuda_graph=True,
            encoder_lens=None,
            capture_hidden_mode=0,
        )

    def test_a_batch_carrying_beam_rows_is_refused(self):
        # The captured/replayed mamba metadata has no src_mamba_cache_indices,
        # so replaying would drop the prune's COW and decode from unwritten
        # state. Refuse rather than replay wrong.
        self.assertFalse(self._runner().can_run_graph(self._forward_batch(3)))

    def test_a_batch_without_beam_rows_is_still_eligible(self):
        # The gate must not turn into a blanket cuda-graph disable.
        self.assertTrue(self._runner().can_run_graph(self._forward_batch(0)))

    def test_a_pending_cow_mapping_makes_the_backend_refuse_a_replay(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        pool = _CpuHybridPool(size=8, mamba_size=8)
        rows = torch.tensor([1, 2], dtype=torch.int64)
        pool.alloc_mamba_for_rows(rows)
        pool.set_beam_cow_read_mapping(rows, torch.tensor([1, 1], dtype=torch.int32))
        backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        backend.req_to_token_pool = pool

        with self.assertRaises(RuntimeError):
            backend._assert_no_beam_cow_for_graph(ForwardMode.DECODE)

    def test_non_decode_graph_paths_are_not_blocked(self):
        # A prefill carrying an armed mapping is normal (the mapping is armed
        # for the group's next decode), so only the decode replay checks.
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        pool = _CpuHybridPool(size=8, mamba_size=8)
        rows = torch.tensor([1, 2], dtype=torch.int64)
        pool.alloc_mamba_for_rows(rows)
        pool.set_beam_cow_read_mapping(rows, torch.tensor([1, 1], dtype=torch.int32))
        backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        backend.req_to_token_pool = pool

        self.assertIsNone(backend._assert_no_beam_cow_for_graph(ForwardMode.EXTEND))
        self.assertIsNone(backend._assert_no_beam_cow_for_graph(ForwardMode.IDLE))

    def test_the_pool_reports_a_pending_mapping(self):
        pool = _CpuHybridPool(size=8, mamba_size=8)
        rows = torch.tensor([1, 2], dtype=torch.int64)
        pool.alloc_mamba_for_rows(rows)

        self.assertFalse(pool.beam_cow_read_mapping_pending())
        pool.set_beam_cow_read_mapping(rows, torch.tensor([1, 1], dtype=torch.int32))
        self.assertTrue(pool.beam_cow_read_mapping_pending())


if __name__ == "__main__":
    unittest.main()
