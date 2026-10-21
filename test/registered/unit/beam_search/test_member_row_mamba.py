"""Unit tests for mamba slot ownership of beam member rows.

Beam member rows are bare req_to_token rows (no Req), so `alloc()` never runs
for them and, on a hybrid pool, their row -> mamba-slot entry stayed 0 -- slot
0, the reserved padding slot. These tests pin the three ownership transitions:

  - spawn:  every member row owns a distinct non-zero slot, or the spawn fails
            loudly (never a silent 0);
  - fork:   a pruned row gets a fresh slot and hands its previous one back one
            tick late;
  - free:   the slots go back to the allocator, so nothing leaks.

Plus the invariant the fix must not break: the non-beam `alloc()` path is
untouched.

Pure tensor bookkeeping on CPU: the pool is built with only the mamba
accounting it needs, so the real method bodies run.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.fork import free_member_rows
from sglang.srt.beam_search.mamba_cow import BeamMambaCow
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _CpuHybridPool(HybridReqToTokenPool):
    """HybridReqToTokenPool with the real mamba accounting on CPU.

    Bypasses HybridReqToTokenPool.__init__ (which builds GPU state pools) and
    installs exactly what the three ownership methods touch: the row -> slot
    table, the slot free-list, and the optional ReplaySSM ring cursor.
    """

    def __init__(self, size: int, mamba_size: int, replayssm: bool = False):
        ReqToTokenPool.__init__(
            self,
            size=size,
            max_context_len=4,
            device="cpu",
            enable_memory_saver=False,
        )
        self.mamba_allocator = MambaSlotAllocator(size=mamba_size, device="cpu")
        self.req_index_to_mamba_index_mapping = torch.zeros(
            size + 1, dtype=torch.int32
        )
        self.beam_mamba_cow = BeamMambaCow()
        self.enable_mamba_extra_buffer = False
        self.mamba_pool = SimpleNamespace(
            size=mamba_size,
            replayssm_write_pos=(
                torch.full((mamba_size + 1,), 7, dtype=torch.int64)
                if replayssm
                else None
            ),
        )


class TestMemberRowAlloc(CustomTestCase):
    def setUp(self):
        self.pool = _CpuHybridPool(size=8, mamba_size=8)
        # A beam_width=4 group's member rows.
        self.rows = torch.tensor([2, 3, 4], dtype=torch.int64)

    def test_each_row_owns_a_distinct_nonzero_slot(self):
        slots = self.pool.alloc_mamba_for_rows(self.rows)

        self.assertEqual(slots.tolist(), self.pool.get_mamba_indices(self.rows).tolist())
        # Not 0: slot 0 is the reserved padding slot every member would
        # otherwise share and corrupt.
        self.assertTrue(bool((slots != 0).all()))
        # Distinct: two members on one slot would write through each other.
        self.assertEqual(len(set(slots.tolist())), self.rows.numel())

    def test_slots_come_out_of_the_allocator(self):
        before = self.pool.mamba_allocator.available_size()

        self.pool.alloc_mamba_for_rows(self.rows)

        self.assertEqual(
            self.pool.mamba_allocator.available_size(), before - self.rows.numel()
        )

    def test_no_rows_allocates_nothing(self):
        slots = self.pool.alloc_mamba_for_rows(torch.zeros(0, dtype=torch.int64))

        self.assertEqual(slots.numel(), 0)
        self.assertEqual(
            self.pool.mamba_allocator.available_size(), self.pool.mamba_allocator.size
        )

    def test_exhausted_allocator_fails_loudly(self):
        # 2 slots, 3 members: no partial batch, and never a silent 0.
        pool = _CpuHybridPool(size=8, mamba_size=2)

        self.assertIsNone(pool.alloc_mamba_for_rows(self.rows))
        # A failure must consume nothing, or a retry would still be short.
        self.assertEqual(pool.mamba_allocator.available_size(), 2)
        self.assertTrue(
            torch.equal(pool.get_mamba_indices(self.rows), torch.zeros(3, dtype=torch.int32))
        )

    def test_a_short_allocator_still_serves_a_smaller_beam(self):
        pool = _CpuHybridPool(size=8, mamba_size=2)
        self.assertIsNone(pool.alloc_mamba_for_rows(self.rows))

        two = torch.tensor([2, 3], dtype=torch.int64)
        slots = pool.alloc_mamba_for_rows(two)

        self.assertIsNotNone(slots)
        self.assertTrue(bool((slots != 0).all()))

    def test_exact_fit_succeeds(self):
        pool = _CpuHybridPool(size=8, mamba_size=3)

        self.assertIsNotNone(pool.alloc_mamba_for_rows(self.rows))
        self.assertEqual(pool.mamba_allocator.available_size(), 0)

    def test_replayssm_ring_starts_empty(self):
        pool = _CpuHybridPool(size=8, mamba_size=8, replayssm=True)

        slots = pool.alloc_mamba_for_rows(self.rows)

        # write_pos=0 means "ring empty", so the decode kernel ignores ring
        # contents. A recycled slot would otherwise inherit a stranger's ring.
        self.assertEqual(pool.mamba_pool.replayssm_write_pos[slots].tolist(), [0, 0, 0])
        # A slot nobody was issued keeps its stale cursor: only the issued ones
        # are reset, not the whole ring table.
        self.assertEqual(int(pool.mamba_pool.replayssm_write_pos[8]), 7)

    def test_extra_buffer_refuses_instead_of_half_owning(self):
        self.pool.enable_mamba_extra_buffer = True

        # The ping-pong cursor lives on Req.kv, which a member row has not.
        with self.assertRaises(RuntimeError):
            self.pool.alloc_mamba_for_rows(self.rows)


class TestMemberRowFree(CustomTestCase):
    def setUp(self):
        self.pool = _CpuHybridPool(size=8, mamba_size=8)
        self.rows = torch.tensor([2, 3, 4], dtype=torch.int64)
        self.slots = self.pool.alloc_mamba_for_rows(self.rows)

    def test_free_returns_the_slots_to_the_allocator(self):
        self.assertEqual(self.pool.mamba_allocator.available_size(), 5)

        self.pool.free_mamba_for_rows(self.rows)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 8)

    def test_freed_slots_are_reissued(self):
        # mamba_size == beam: a leak shows up as the next beam being refused,
        # which is exactly how a leaked member row would surface in production.
        pool = _CpuHybridPool(size=8, mamba_size=3)
        self.assertIsNotNone(pool.alloc_mamba_for_rows(self.rows))

        pool.free_mamba_for_rows(self.rows)

        self.assertIsNotNone(pool.alloc_mamba_for_rows(self.rows))

    def test_free_zeroes_the_table_entry(self):
        self.pool.free_mamba_for_rows(self.rows)

        self.assertTrue(
            torch.equal(
                self.pool.get_mamba_indices(self.rows), torch.zeros(3, dtype=torch.int32)
            )
        )

    def test_double_free_cannot_inject_slot_zero(self):
        self.pool.free_mamba_for_rows(self.rows)
        self.pool.free_mamba_for_rows(self.rows)

        free = self.pool.mamba_allocator.free_slots.tolist()
        self.assertEqual(len(free), 8)
        self.assertNotIn(0, free)
        self.assertEqual(sorted(free), list(range(1, 9)))

    def test_freeing_unowned_rows_frees_nothing(self):
        # Rows 5/6 never went through alloc_mamba_for_rows: their entry is 0,
        # the reserved padding slot, which must never reach the allocator.
        self.pool.free_mamba_for_rows(torch.tensor([5, 6], dtype=torch.int64))

        self.assertEqual(self.pool.mamba_allocator.available_size(), 5)

    def test_freeing_no_rows_is_a_no_op(self):
        self.pool.free_mamba_for_rows(torch.zeros(0, dtype=torch.int64))

        self.assertEqual(self.pool.mamba_allocator.available_size(), 5)

    def test_members_and_leader_do_not_share_a_slot(self):
        leader_row = torch.tensor([1], dtype=torch.int64)
        leader_slot = self.pool.alloc_mamba_for_rows(leader_row)

        all_slots = torch.cat([leader_slot, self.slots])
        self.assertEqual(len(set(all_slots.tolist())), all_slots.numel())


class TestMemberRowFork(CustomTestCase):
    def setUp(self):
        self.pool = _CpuHybridPool(size=8, mamba_size=8)
        self.rows = torch.tensor([2, 3, 4], dtype=torch.int64)
        self.owned = self.pool.alloc_mamba_for_rows(self.rows)

    def test_fork_gives_a_fresh_slot_per_row(self):
        previous = self.pool.replace_mamba_for_rows(self.rows)

        fresh = self.pool.get_mamba_indices(self.rows)
        self.assertTrue(torch.equal(previous, self.owned))
        # No row keeps the slot it had: two survivors of one parent would write
        # through the same state otherwise.
        self.assertEqual(len(set(fresh.tolist()) & set(previous.tolist())), 0)
        self.assertTrue(bool((fresh != 0).all()))

    def test_previous_slots_are_released_one_tick_late(self):
        previous = self.pool.replace_mamba_for_rows(self.rows)

        # Staged, not freed: the kernel is still reading them this tick. The 3
        # fresh slots are already out, so only 2 of the 8 remain.
        self.pool.stage_beam_cow_source_slots(previous)
        self.assertEqual(self.pool.mamba_allocator.available_size(), 2)

        self.pool.flush_beam_cow_source_slots()

        # The previous slots are back; the fresh ones stay owned by the rows.
        self.assertEqual(self.pool.mamba_allocator.available_size(), 5)
        self.assertEqual(
            sorted(self.pool.mamba_allocator.free_slots.tolist()), [1, 2, 3, 7, 8]
        )

    def test_unowned_rows_are_omitted_from_the_handback(self):
        # Row 5 owns nothing. Returning its 0 would stage slot 0 and free the
        # reserved padding slot on the next flush.
        rows = torch.tensor([2, 5], dtype=torch.int64)

        previous = self.pool.replace_mamba_for_rows(rows)

        self.assertEqual(previous.numel(), 1)
        self.assertNotIn(0, previous.tolist())

    def test_short_allocator_leaves_ownership_intact(self):
        pool = _CpuHybridPool(size=8, mamba_size=3)
        owned = pool.alloc_mamba_for_rows(self.rows)

        self.assertIsNone(pool.replace_mamba_for_rows(self.rows))
        # The rows must keep the slots they had, not be left at 0.
        self.assertTrue(torch.equal(pool.get_mamba_indices(self.rows), owned))
        self.assertEqual(pool.mamba_allocator.available_size(), 0)

    def test_fork_then_free_releases_every_slot(self):
        self.pool.replace_mamba_for_rows(self.rows)
        self.pool.stage_beam_cow_source_slots(self.owned)
        self.pool.flush_beam_cow_source_slots()

        self.pool.free_mamba_for_rows(self.rows)

        self.assertEqual(self.pool.mamba_allocator.available_size(), 8)

    def test_no_rows_forks_to_nothing(self):
        previous = self.pool.replace_mamba_for_rows(torch.zeros(0, dtype=torch.int64))

        self.assertEqual(previous.numel(), 0)
        self.assertEqual(self.pool.mamba_allocator.available_size(), 5)


class TestMemberRowReleaseThroughTheBeamPath(CustomTestCase):
    """The pool API is only half the fix: fork.free_member_rows has to call it,
    or a member row's slot outlives the group."""

    def _group(self, pool, num_members):
        """A group whose leader + member rows really came out of the pool."""
        rows = pool.alloc_rows(num_members + 1)
        members = torch.tensor(rows[1:], dtype=torch.int64)
        leader = SimpleNamespace(kv=ReqKvInfo(kv_allocated_len=2, kv_committed_len=2))
        return SimpleNamespace(
            member_rows=members,
            member_rows_cpu=members.clone(),
            all_rows=torch.tensor(rows, dtype=torch.int64),
            prompt_len=2,
            leader=leader,
        )

    def test_teardown_returns_the_member_slots(self):
        pool = _CpuHybridPool(size=8, mamba_size=8)
        group = self._group(pool, 2)
        pool.alloc_mamba_for_rows(group.member_rows)
        self.assertEqual(pool.mamba_allocator.available_size(), 6)

        free_member_rows(group, pool, SimpleNamespace(free=lambda slots: None))

        self.assertEqual(pool.mamba_allocator.available_size(), 8)
        # The member rows go back too (only the leader's row stays out), and the
        # table entry is cleared so a reissued row cannot inherit a freed slot.
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3, 4, 5, 7, 8])
        self.assertTrue(
            torch.equal(
                pool.get_mamba_indices(torch.tensor([7, 8])), torch.zeros(2, dtype=torch.int32)
            )
        )

    def test_teardown_without_a_spawn_owns_nothing(self):
        # beam_width == 1: no member rows were ever spawned.
        pool = _CpuHybridPool(size=8, mamba_size=8)
        group = self._group(pool, 1)
        group.all_rows = group.all_rows[:1]

        free_member_rows(group, pool, SimpleNamespace(free=lambda slots: None))

        self.assertEqual(pool.mamba_allocator.available_size(), 8)

    def test_teardown_is_idempotent(self):
        pool = _CpuHybridPool(size=8, mamba_size=8)
        group = self._group(pool, 2)
        pool.alloc_mamba_for_rows(group.member_rows)

        free_member_rows(group, pool, SimpleNamespace(free=lambda slots: None))
        free_member_rows(group, pool, SimpleNamespace(free=lambda slots: None))

        self.assertEqual(pool.mamba_allocator.available_size(), 8)
        self.assertNotIn(0, pool.mamba_allocator.free_slots.tolist())

    def test_failed_spawn_returns_the_req_rows(self):
        # The req rows are already out of free_slots when the mamba alloc is
        # attempted, and the admission gate does not reserve mamba slots for
        # members yet -- so this path is reachable on a full pool. Leaking the
        # rows here would wedge the pool a few requests later.
        from sglang.srt.beam_search.coordinator import BeamCoordinator

        pool = _CpuHybridPool(size=8, mamba_size=1)
        leader = SimpleNamespace(
            kv=ReqKvInfo(kv_allocated_len=2, kv_committed_len=2, req_pool_idx=1)
        )
        pool.alloc_rows(1)  # the leader's own row
        group = SimpleNamespace(beam_width=4, prompt_len=2)
        coord = SimpleNamespace(req_to_token_pool=pool)

        rows_before = sorted(pool.free_slots)
        with self.assertRaises(AssertionError):
            BeamCoordinator._spawn_member_rows(coord, group, leader)

        self.assertEqual(sorted(pool.free_slots), rows_before)
        self.assertIsNone(group.member_rows)

    def test_non_hybrid_pool_is_skipped(self):
        # A plain pool has no member-row API; free_member_rows must not care.
        freed = []

        class _PlainPool:
            def free_rows(self, indices):
                freed.extend(indices)

        group = SimpleNamespace(
            member_rows=torch.tensor([6, 7], dtype=torch.int64),
            member_rows_cpu=torch.tensor([6, 7], dtype=torch.int64),
            all_rows=torch.tensor([5, 6, 7], dtype=torch.int64),
            prompt_len=2,
            leader=SimpleNamespace(
                kv=ReqKvInfo(kv_allocated_len=2, kv_committed_len=2)
            ),
        )
        free_member_rows(group, _PlainPool(), SimpleNamespace(free=lambda slots: None))

        self.assertEqual(freed, [6, 7])

    def test_a_full_beam_cycle_leaks_nothing(self):
        pool = _CpuHybridPool(size=8, mamba_size=1)
        # One mamba slot in the whole pool: three consecutive groups can only
        # all fit if each released its member slot on teardown.
        for _ in range(3):
            group = self._group(pool, 1)
            self.assertIsNotNone(pool.alloc_mamba_for_rows(group.member_rows))
            free_member_rows(group, pool, SimpleNamespace(free=lambda slots: None))
        self.assertEqual(pool.mamba_allocator.available_size(), 1)


class TestNonBeamAllocUnaffected(CustomTestCase):
    """The fix must not move the non-beam path: ordinary requests still own
    exactly one slot each, issued by alloc() and nothing else."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=8, mamba_size=8)
        self.reqs = [SimpleNamespace(kv=ReqKvInfo()) for _ in range(3)]

    def test_alloc_rows_is_not_overridden_for_mamba(self):
        # Member-row ownership is an explicit interface precisely because
        # alloc() routes through alloc_rows(): allocating there too would
        # double-allocate every ordinary request.
        self.assertIs(
            HybridReqToTokenPool.alloc_rows, ReqToTokenPool.alloc_rows
        )

    def test_plain_pool_has_no_member_row_api(self):
        # So the beam side's capability probe skips it entirely.
        pool = ReqToTokenPool(
            size=4, max_context_len=4, device="cpu", enable_memory_saver=False
        )
        self.assertFalse(hasattr(pool, "alloc_mamba_for_rows"))
        self.assertFalse(hasattr(pool, "free_mamba_for_rows"))

    def test_alloc_owns_one_slot_per_request(self):
        before = self.pool.mamba_allocator.available_size()

        rows = self.pool.alloc(self.reqs)

        self.assertEqual(len(rows), 3)
        # One slot per request, not two (alloc_rows must stay mamba-free).
        self.assertEqual(self.pool.mamba_allocator.available_size(), before - 3)

    def test_alloc_slots_match_the_requests_and_the_table(self):
        rows = self.pool.alloc(self.reqs)

        slots = self.pool.get_mamba_indices(torch.tensor(rows, dtype=torch.int64))
        for req, slot in zip(self.reqs, slots.tolist()):
            self.assertEqual(int(req.kv.mamba_pool_idx), slot)
            self.assertNotEqual(slot, 0)
            self.assertTrue(req.kv.mamba_needs_clear)
        self.assertEqual(len(set(slots.tolist())), 3)

    def test_alloc_then_member_rows_stay_disjoint(self):
        rows = self.pool.alloc(self.reqs)
        # alloc_rows pops from the tail, so rows are [6, 7, 8] and these are
        # still free -- member rows live in the same index space as the leader.
        members = torch.tensor([1, 2], dtype=torch.int64)

        member_slots = self.pool.alloc_mamba_for_rows(members)

        leader_slots = self.pool.get_mamba_indices(torch.tensor(rows, dtype=torch.int64))
        self.assertEqual(
            len(set(member_slots.tolist()) & set(leader_slots.tolist())), 0
        )


if __name__ == "__main__":
    unittest.main()
