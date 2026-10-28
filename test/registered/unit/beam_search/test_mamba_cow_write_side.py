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
              mapping ever does reach a replay the backend says so loudly;
  - copy:     the --enable-beam-mamba-state-copy alternative, which copies the
              state outright instead of arming a read mapping.

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

    def _validate(self, *, swa, mamba, double_buffer=False, state_copy=False):
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
            enable_beam_mamba_state_copy=state_copy,
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

    def test_mamba_is_allowed_with_only_the_state_copy_switch(self):
        # The true-copy path is a complete opt-in on its own: it never arms a
        # read mapping, so it needs nothing from the double buffer.
        self.assertIsNone(self._validate(swa=False, mamba=True, state_copy=True))

    def test_mamba_is_allowed_with_both_switches(self):
        # Both on: admission does not care which one the prune will pick.
        self.assertIsNone(
            self._validate(swa=False, mamba=True, double_buffer=True, state_copy=True)
        )

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
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
            # can_run_graph also asks the pool whether a prune mapping is
            # pending: beam_width=1 prunes without producing member rows, so
            # the row count alone does not catch it.
            req_to_token_pool=SimpleNamespace(
                beam_cow_read_mapping_pending=lambda: False
            ),
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


class _CpuMambaPool:
    """MambaPool stand-in carrying observable state.

    Only what the state-copy prune touches: a `temporal` tensor indexed the way
    `MambaPool.copy_from` indexes it ([layers, slots, dim]), the ReplaySSM ring
    cursor, and a `copy_from` that performs the copy and records its arguments.

    It also enforces the src/dst disjointness the real fused copy asserts. That
    is the whole reason the prune allocates fresh slots first: an in-place copy
    would hand overlapping sets here on a self-parented row.
    """

    def __init__(self, size: int, state_dim: int = 3, replayssm: bool = False):
        self.size = size
        self.temporal = torch.zeros(1, size + 1, state_dim)
        self.replayssm_write_pos = (
            torch.zeros(size + 1, dtype=torch.int64) if replayssm else None
        )
        self.calls = []

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        src = src_indices.tolist()
        dst = dst_indices.tolist()
        overlap = set(src) & set(dst)
        assert not overlap, (
            "copy_from requires disjoint src/dst slots; "
            f"overlap={sorted(overlap)}"
        )
        self.calls.append((src, dst))
        self.temporal[:, dst] = self.temporal[:, src]


class _CpuStateCopyPool(_CpuHybridPool):
    """_CpuHybridPool with a mamba_pool the copy path can actually copy from."""

    def __init__(
        self, size: int, mamba_size: int, state_dim: int = 3, replayssm: bool = False
    ):
        super().__init__(size=size, mamba_size=mamba_size, replayssm=replayssm)
        self.mamba_pool = _CpuMambaPool(
            mamba_size, state_dim=state_dim, replayssm=replayssm
        )


class _CpuVirtualSlotPool(_CpuStateCopyPool):
    """A pool whose mamba ids are virtual: the physical one is offset.

    `UnifiedHybridReqToTokenPool` hands out virtual ids; `copy_from` addresses
    the state tensors physically, so the prune must translate both sides. The
    offset makes "did it translate?" observable.
    """

    OFFSET = 100

    def __init__(self, size: int, mamba_size: int):
        super().__init__(size=size, mamba_size=mamba_size)
        self.mamba_pool.size = mamba_size + self.OFFSET
        self.mamba_pool.temporal = torch.zeros(
            1, mamba_size + self.OFFSET + 1, self.mamba_pool.temporal.shape[2]
        )

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        return mamba_indices + self.OFFSET


class TestStateCopyIsArmed(CustomTestCase):
    """--enable-beam-mamba-state-copy: the prune copies state instead of
    arming a read mapping, so nothing spans the tick."""

    def setUp(self):
        self.pool = _CpuStateCopyPool(size=8, mamba_size=8)
        self.rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.owned = self.pool.alloc_mamba_for_rows(self.rows)
        self.coord = _coordinator(self.pool, enable_beam_mamba_state_copy=True)

    @staticmethod
    def _seed(pool, slots):
        """Give every slot a distinct state: slot s holds the value s."""
        for s in slots.tolist():
            pool.mamba_pool.temporal[:, s] = float(s)

    @staticmethod
    def _state(pool, slots):
        return [float(pool.mamba_pool.temporal[0, s, 0]) for s in slots.tolist()]

    def test_no_read_mapping_is_left_pending(self):
        # The whole point: without a snapshot there is nothing for the backend
        # to consume, nothing for the graph gates to trip on, and no stale
        # mapping that could outlive its tick.
        self.coord._apply_mamba_cow(self.rows, torch.tensor([1, 1, 0, 2]))

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertIsNone(self.pool.get_beam_cow_read_mapping(self.rows))

    def test_the_rows_move_to_fresh_slots(self):
        self.coord._apply_mamba_cow(self.rows, torch.tensor([1, 1, 0, 2]))

        new = self.pool.get_mamba_indices(self.rows)
        self.assertEqual(new.tolist(), [5, 6, 7, 8])
        # Fresh, hence disjoint from every source slot by construction.
        self.assertEqual(set(new.tolist()) & set(self.owned.tolist()), set())

    def test_the_new_slots_hold_the_parents_state(self):
        # Not just the indices: the state itself has to have moved.
        parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)
        self._seed(self.pool, self.owned)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        new = self.pool.get_mamba_indices(self.rows)
        # Slot s was seeded with s, and row i's parent owned owned[parent_idx[i]].
        self.assertEqual(
            self._state(self.pool, new), self.owned[parent_idx].tolist()
        )
        # And the copy really ran, with src/dst the way the design says.
        self.assertEqual(
            self.pool.mamba_pool.calls,
            [(self.owned[parent_idx].tolist(), new.tolist())],
        )

    def test_a_self_parented_row_copies_its_own_state(self):
        # parent_idx[i] == i: the row keeps its own history, which is still a
        # copy -- into a fresh slot, so src and dst stay disjoint.
        parent_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self._seed(self.pool, self.owned)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        new = self.pool.get_mamba_indices(self.rows)
        self.assertEqual(self._state(self.pool, new), self.owned.tolist())
        self.assertEqual(set(self.owned.tolist()) & set(new.tolist()), set())

    def test_two_rows_swapping_parents(self):
        # The case an in-place copy gets wrong: with dst drawn from the live
        # slot set, row 0's dst IS row 1's src, so one of them loses. Allocating
        # fresh slots first cannot alias.
        parent_idx = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
        self._seed(self.pool, self.owned)

        self.coord._apply_mamba_cow(self.rows, parent_idx)

        new = self.pool.get_mamba_indices(self.rows)
        self.assertEqual(
            self._state(self.pool, new), self.owned[parent_idx].tolist()
        )
        self.assertEqual(
            self._state(self.pool, new), [2.0, 1.0, 4.0, 3.0]
        )

    def test_the_copy_addresses_physical_slots(self):
        # The table holds virtual ids; copy_from addresses the state tensors
        # physically. On a unified pool those differ, so an untranslated call
        # would copy garbage from an unrelated slot.
        pool = _CpuVirtualSlotPool(size=8, mamba_size=8)
        rows = torch.tensor([1, 2], dtype=torch.int64)
        owned = pool.alloc_mamba_for_rows(rows)
        coord = _coordinator(pool, enable_beam_mamba_state_copy=True)
        parent_idx = torch.tensor([1, 0], dtype=torch.int64)

        coord._apply_mamba_cow(rows, parent_idx)

        new = pool.get_mamba_indices(rows)
        self.assertEqual(
            pool.mamba_pool.calls,
            [
                (
                    [int(owned[1]) + pool.OFFSET, int(owned[0]) + pool.OFFSET],
                    [int(new[0]) + pool.OFFSET, int(new[1]) + pool.OFFSET],
                )
            ],
        )

    def test_the_state_copy_wins_when_both_switches_are_on(self):
        # Documented precedence: the true copy holds no state across a tick, so
        # it is the safer of the two when a user sets both.
        coord = _coordinator(
            self.pool,
            enable_beam_mamba_double_buffer=True,
            enable_beam_mamba_state_copy=True,
        )

        coord._apply_mamba_cow(self.rows, torch.tensor([1, 1, 0, 2]))

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertEqual(len(self.pool.mamba_pool.calls), 1)

    def test_a_graph_replay_is_allowed_after_the_prune(self):
        # Unlike the COW path, whose pending mapping makes the backend refuse a
        # DECODE replay: nothing is pending, so the gate is open.
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            MambaAttnBackendBase,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.coord._apply_mamba_cow(self.rows, torch.tensor([1, 1, 0, 2]))
        backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
        backend.req_to_token_pool = self.pool

        self.assertIsNone(backend._assert_no_beam_cow_for_graph(ForwardMode.DECODE))

    def test_a_non_hybrid_pool_is_a_no_op(self):
        coord = _coordinator(object(), enable_beam_mamba_state_copy=True)

        self.assertIsNone(
            coord._apply_mamba_cow(self.rows, torch.tensor([0, 0, 0, 0]))
        )

    def test_no_rows_does_nothing(self):
        empty = torch.zeros(0, dtype=torch.int64)

        self.coord._apply_mamba_cow(empty, empty)

        self.assertEqual(self.pool.mamba_pool.calls, [])
        self.assertEqual(self.pool.mamba_allocator.available_size(), 4)


class TestStateCopyRefusesWhatItCannotDo(CustomTestCase):
    """Both guards fire before anything moves, so a refused prune leaves the
    group exactly as it was."""

    def setUp(self):
        self.rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)

    def test_replayssm_is_refused(self):
        pool = _CpuStateCopyPool(size=8, mamba_size=8, replayssm=True)
        owned = pool.alloc_mamba_for_rows(self.rows)
        coord = _coordinator(pool, enable_beam_mamba_state_copy=True)

        with self.assertRaises(RuntimeError) as ctx:
            coord._apply_mamba_cow(self.rows, self.parent_idx)

        # The reason has to be in the message: copy_from moves temporal only,
        # so an active source's ring would be dropped silently.
        self.assertIn("--enable-linear-replayssm", str(ctx.exception))

    def test_replayssm_leaves_the_rows_untouched(self):
        pool = _CpuStateCopyPool(size=8, mamba_size=8, replayssm=True)
        owned = pool.alloc_mamba_for_rows(self.rows)
        coord = _coordinator(pool, enable_beam_mamba_state_copy=True)

        with self.assertRaises(RuntimeError):
            coord._apply_mamba_cow(self.rows, self.parent_idx)

        self.assertEqual(pool.get_mamba_indices(self.rows).tolist(), owned.tolist())
        self.assertFalse(pool.beam_cow_read_mapping_pending())
        self.assertEqual(pool.mamba_pool.calls, [])
        self.assertEqual(pool.mamba_allocator.available_size(), 4)

    def test_a_short_allocator_raises(self):
        # Same as the COW path: 4 rows, only 1 slot left.
        pool = _CpuStateCopyPool(size=8, mamba_size=5)
        pool.alloc_mamba_for_rows(self.rows)
        coord = _coordinator(pool, enable_beam_mamba_state_copy=True)

        with self.assertRaises(RuntimeError) as ctx:
            coord._apply_mamba_cow(self.rows, self.parent_idx)

        self.assertIn("--enable-beam-mamba-state-copy", str(ctx.exception))

    def test_a_short_allocator_leaves_the_rows_untouched(self):
        pool = _CpuStateCopyPool(size=8, mamba_size=5)
        pool.alloc_mamba_for_rows(self.rows)
        coord = _coordinator(pool, enable_beam_mamba_state_copy=True)

        with self.assertRaises(RuntimeError):
            coord._apply_mamba_cow(self.rows, self.parent_idx)

        self.assertEqual(pool.get_mamba_indices(self.rows).tolist(), [1, 2, 3, 4])
        self.assertFalse(pool.beam_cow_read_mapping_pending())
        self.assertEqual(pool.mamba_pool.calls, [])
        self.assertEqual(pool.mamba_allocator.available_size(), 1)


class TestStateCopyFreesTheOldSlotsOneTickLate(CustomTestCase):
    """copy_from only enqueues on the stream: the detached slots still need the
    deferred release, exactly as on the COW path."""

    def setUp(self):
        self.pool = _CpuStateCopyPool(size=8, mamba_size=8)
        self.rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        self.owned = self.pool.alloc_mamba_for_rows(self.rows)
        self.coord = _coordinator(self.pool, enable_beam_mamba_state_copy=True)
        self.parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)

    def test_the_detached_slots_do_not_return_in_the_pruning_tick(self):
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 0)
        self.assertNotIn(1, self.pool.mamba_allocator.free_slots.tolist())

    def test_the_relay_hook_returns_them_on_the_next_forward(self):
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)

        self.coord.maybe_select_and_relay(None, None)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 4)
        self.assertEqual(
            sorted(self.pool.mamba_allocator.free_slots.tolist()), [1, 2, 3, 4]
        )
        # The rows keep the slots the state was copied into.
        self.assertEqual(self.pool.get_mamba_indices(self.rows).tolist(), [5, 6, 7, 8])

    def test_every_slot_is_back_after_a_prune_cycle(self):
        self.coord._apply_mamba_cow(self.rows, self.parent_idx)
        self.coord.maybe_select_and_relay(None, None)

        self.pool.free_mamba_for_rows(self.rows)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 8)


if __name__ == "__main__":
    unittest.main()
