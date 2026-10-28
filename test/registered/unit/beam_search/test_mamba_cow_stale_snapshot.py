"""Regression: a beam group must not leave its mamba COW read mapping behind.

The read mapping is armed by a prune and is meant to be consumed by the very
next decode (hybrid_linear_attn_backend reads it). But a group can end in the
commit that follows the arming -- early stopping, abort, retraction -- and then
no decode ever comes. The mapping outlived its group: the rows went back to
free_slots and the next request admitted got them, and its first decode read
through the leftover mapping, i.e. the parent slots the dead group had detached
and freed. A use-after-free on state that now belongs to somebody else.

The fix is a withdrawal on retire (BeamMambaCow.forget_rows, called from
BeamCoordinator._retire_group). These tests pin it down:

  - a group that arms and then retires takes its rows out of the snapshot, and
    the snapshot itself goes away once nobody is waiting for it;
  - a second group that pruned in the same tick is *not* collateral damage --
    it shares the snapshot and keeps reading through it;
  - every path that cannot have armed (no spawn, failed spawn, the true-copy
    branch, a non-hybrid pool, a snapshot already consumed) is a no-op;
  - the host-side row list the prune hands down is the same rows, in the same
    order, as the device tensor it armed -- the two must never disagree, or the
    withdrawal would name different rows than the arming did.

Pure CPU tensor bookkeeping on the fake hybrid pool from
test_member_row_mamba / test_mamba_cow_write_side, so the real method bodies
run.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_mamba_cow_write_side import _CpuStateCopyPool, _coordinator
from test_member_row_mamba import _CpuHybridPool

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PROMPT_LEN = 2


def _live_group(pool, beam_width=4, leader_output=None):
    """A real BeamGroup wired the way the coordinator wires it at spawn.

    Rows come out of the pool's free list, every row owns a mamba slot, and
    both the device tensor and the host list are populated -- the pair the
    retire-time withdrawal depends on.
    """
    rows = pool.alloc_rows(beam_width)
    group = BeamGroup(beam_width=beam_width, max_new_tokens=8)
    group.prompt_len = PROMPT_LEN
    group.leader = SimpleNamespace(
        kv=ReqKvInfo(
            kv_allocated_len=PROMPT_LEN,
            kv_committed_len=PROMPT_LEN,
            req_pool_idx=rows[0],
        ),
        output_ids=[] if leader_output is None else leader_output,
        to_finish=None,
    )
    members = torch.tensor(rows[1:], dtype=torch.int64)
    group.member_rows = members
    group.member_rows_cpu = members.clone()
    group.all_rows = torch.tensor(rows, dtype=torch.int64)
    group.all_rows_cpu = list(rows)
    pool.alloc_mamba_for_rows(group.all_rows)
    return group


def _arm(coord, group, parent_idx):
    """One group's prune, exactly as _apply_survivors issues it."""
    coord._apply_mamba_cow(
        group.all_rows,
        torch.tensor(parent_idx, dtype=torch.int64),
        group.all_rows_cpu,
    )


class TestRetireWithdrawsTheSnapshot(CustomTestCase):
    """The reported bug: arm, then finish in the same commit, no consumer."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=12, mamba_size=16)
        self.coord = _coordinator(self.pool)
        self.coord._num_live_groups = 1
        self.group = _live_group(self.pool)
        self.rows = self.group.all_rows

    def test_the_armed_snapshot_is_there_before_the_retire(self):
        # Fixture sanity: the prune really did arm one, so the assertions below
        # are about the retire and not about an arming that never happened.
        _arm(self.coord, self.group, [1, 1, 0, 2])

        self.assertTrue(self.pool.beam_cow_read_mapping_pending())
        self.assertIsNotNone(self.pool.get_beam_cow_read_mapping(self.rows))

    def test_retiring_drops_the_snapshot_nobody_will_consume(self):
        _arm(self.coord, self.group, [1, 1, 0, 2])

        # The group ended in the same commit: no forward ever reads this.
        self.coord._retire_group(self.group)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        # None means "read == write" to the backend, i.e. the live table --
        # not the parents' slots the group detached and freed.
        self.assertIsNone(self.pool.get_beam_cow_read_mapping(self.rows))

    def test_the_next_owner_of_these_rows_reads_live(self):
        # The reported sequence: request 1 arms, ends; request 2 is admitted
        # onto the same row numbers and decodes. Before the fix its first
        # decode found request 1's mapping still armed for those rows.
        _arm(self.coord, self.group, [1, 1, 0, 2])
        self.coord._retire_group(self.group)
        stale_src = self.pool.get_beam_cow_read_mapping(self.rows)

        # Request 2: same rows, fresh slots, no prune of its own yet.
        group2 = _live_group(self.pool)
        rows2 = group2.all_rows

        self.assertIsNone(stale_src)
        # None is the backend's cue to read and write the same slot, i.e. the
        # live table: these rows get their own freshly allocated state, not
        # request 1's detached and freed parents.
        self.assertIsNone(self.pool.get_beam_cow_read_mapping(rows2))
        self.assertEqual(
            self.pool.get_mamba_indices(rows2).tolist(),
            self.pool.req_index_to_mamba_index_mapping[rows2].tolist(),
        )

    def test_retiring_twice_is_a_no_op(self):
        # _retire_group is guarded, and forget_rows on an absent snapshot
        # returns early rather than touching a None mask.
        _arm(self.coord, self.group, [1, 1, 0, 2])
        self.coord._retire_group(self.group)

        self.coord._retire_group(self.group)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())

    def test_the_abort_path_withdraws_after_the_rows_are_gone(self):
        # The realistic shape: _abort_group frees the member rows first, and
        # fork.free_member_rows nulls all_rows / member_rows_cpu on the way
        # out. Only the host list survives, so the withdrawal has to run off
        # that -- otherwise the finish and abort paths, the common ones, would
        # silently skip it.
        _arm(self.coord, self.group, [1, 1, 0, 2])
        self.assertTrue(self.pool.beam_cow_read_mapping_pending())

        self.coord._abort_group(self.group)

        self.assertIsNone(self.group.all_rows)
        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertIsNone(
            self.pool.get_beam_cow_read_mapping(
                torch.tensor(self.group.all_rows_cpu, dtype=torch.int64)
            )
        )


class TestWithdrawalIsScopedToOneGroup(CustomTestCase):
    """Two groups prune in one tick, so they share a single snapshot."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=12, mamba_size=16)
        self.coord = _coordinator(self.pool)
        self.coord._num_live_groups = 2
        self.group1 = _live_group(self.pool)
        self.group2 = _live_group(self.pool)
        # The parents' slots each group's rows are redirected to, captured
        # before either prune moves anything.
        self.src1 = (
            self.pool.get_mamba_indices(self.group1.all_rows)[
                torch.tensor([1, 1, 0, 2], dtype=torch.int64)
            ]
            .clone()
            .tolist()
        )
        _arm(self.coord, self.group1, [1, 1, 0, 2])
        self.src2 = (
            self.pool.get_mamba_indices(self.group2.all_rows)[
                torch.tensor([2, 0, 1, 1], dtype=torch.int64)
            ]
            .clone()
            .tolist()
        )
        _arm(self.coord, self.group2, [2, 0, 1, 1])

    def test_both_groups_are_armed(self):
        self.assertTrue(self.pool.beam_cow_read_mapping_pending())
        self.assertEqual(
            self.pool.get_beam_cow_read_mapping(self.group1.all_rows).tolist(),
            self.src1,
        )
        self.assertEqual(
            self.pool.get_beam_cow_read_mapping(self.group2.all_rows).tolist(),
            self.src2,
        )

    def test_the_surviving_group_keeps_reading_through_the_snapshot(self):
        self.coord._retire_group(self.group1)

        # Still armed: group 2 is waiting for its decode.
        self.assertTrue(self.pool.beam_cow_read_mapping_pending())
        self.assertEqual(
            self.pool.get_beam_cow_read_mapping(self.group2.all_rows).tolist(),
            self.src2,
        )

    def test_the_retired_groups_rows_fall_back_to_live(self):
        self.coord._retire_group(self.group1)

        # Not an error and not a stale read: these rows now read the live
        # table, exactly like any row nobody pruned.
        self.assertEqual(
            self.pool.get_beam_cow_read_mapping(self.group1.all_rows).tolist(),
            self.pool.get_mamba_indices(self.group1.all_rows).tolist(),
        )

    def test_the_snapshot_goes_away_with_the_last_group(self):
        self.coord._retire_group(self.group1)
        self.coord._retire_group(self.group2)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertIsNone(self.pool.get_beam_cow_read_mapping(self.group2.all_rows))


class TestWithdrawalIsANoOpWhereNothingWasArmed(CustomTestCase):
    """Retire runs on every exit, most of which never touched the snapshot."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=12, mamba_size=16)
        self.coord = _coordinator(self.pool)
        self.coord._num_live_groups = 1

    def test_a_group_that_never_pruned_retires_cleanly(self):
        group = _live_group(self.pool)
        self.assertFalse(self.pool.beam_cow_read_mapping_pending())

        self.coord._retire_group(group)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())

    def test_a_group_that_never_spawned_retires_cleanly(self):
        # max_new_tokens == 1 terminates at prefill: no member rows, no prune,
        # and all_rows / all_rows_cpu are still None.
        group = BeamGroup(beam_width=4, max_new_tokens=1)
        group.leader = SimpleNamespace(
            kv=ReqKvInfo(kv_allocated_len=PROMPT_LEN, kv_committed_len=PROMPT_LEN),
            output_ids=[],
            to_finish=None,
        )

        self.coord._retire_group(group)

        self.assertIsNone(group.all_rows_cpu)
        self.assertFalse(self.pool.beam_cow_read_mapping_pending())

    def test_the_consumed_snapshot_needs_no_withdrawal(self):
        # The normal path: the backend reads the mapping and clears it. The
        # retire afterwards must not trip over the absent snapshot.
        group = _live_group(self.pool)
        _arm(self.coord, group, [1, 1, 0, 2])
        self.pool.get_beam_cow_read_mapping(group.all_rows)  # backend consumes
        self.pool.clear_beam_cow_read_mapping()

        self.coord._retire_group(group)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())

    def test_a_non_hybrid_pool_is_a_no_op(self):
        # No mamba table, no read mapping, and no forget method at all: the
        # pool getattr resolves to None and retire moves on.
        coord = _coordinator(object())
        coord._num_live_groups = 1
        group = _live_group(_CpuHybridPool(size=12, mamba_size=16))

        coord._retire_group(group)

        self.assertTrue(group.retired)

    def test_the_rows_cpu_fallback_still_arms_and_withdraws(self):
        # set_read_mapping without rows_cpu falls back to reading the device
        # tensor back (documented as test/cold-path only): the host set is
        # still populated, so a later withdrawal works on it.
        group = _live_group(self.pool)
        self.pool.set_beam_cow_read_mapping(
            group.all_rows, self.pool.get_mamba_indices(group.all_rows)
        )

        self.assertTrue(self.pool.beam_cow_read_mapping_pending())
        self.assertEqual(
            self.pool.beam_mamba_cow._armed_rows, set(group.all_rows.tolist())
        )

        self.pool.forget_beam_cow_rows(group.all_rows, group.all_rows_cpu)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())


class TestStateCopyRetiresCleanly(CustomTestCase):
    """--enable-beam-mamba-state-copy arms nothing, so nothing can go stale."""

    def setUp(self):
        self.pool = _CpuStateCopyPool(size=12, mamba_size=16)
        self.coord = _coordinator(self.pool, enable_beam_mamba_state_copy=True)
        self.coord._num_live_groups = 1
        self.group = _live_group(self.pool)

    def test_the_copy_branch_leaves_no_snapshot_to_withdraw(self):
        _arm(self.coord, self.group, [1, 1, 0, 2])

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertEqual(self.pool.beam_mamba_cow._armed_rows, set())

    def test_retire_after_the_copy_branch_is_safe(self):
        _arm(self.coord, self.group, [1, 1, 0, 2])

        self.coord._retire_group(self.group)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.assertTrue(self.group.retired)


class TestPruneHandsDownTheMatchingHostRows(CustomTestCase):
    """rows_cpu must be `rows` on the host, same order, or the withdrawal
    would name different rows than the arming did."""

    def setUp(self):
        self.pool = _CpuHybridPool(size=12, mamba_size=16)
        self.coord = _coordinator(self.pool)
        self.group = _live_group(self.pool)
        # req_to_token is [size + 1, max_context_len]; the remap walks
        # [prompt_len, kv_committed_len) only.
        self.tokens = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
        self.parent_idx = torch.tensor([1, 1, 0, 2], dtype=torch.int64)
        self.seen = {}

    def test_apply_survivors_passes_all_rows_in_frontier_order(self):
        real = BeamCoordinator._apply_mamba_cow

        def spy(coord, rows, parent_idx, rows_cpu=None):
            self.seen["rows"] = rows
            self.seen["rows_cpu"] = rows_cpu
            self.seen["parent_idx"] = parent_idx
            return real(coord, rows, parent_idx, rows_cpu)

        # On the class: a BeamCoordinator is a msgspec Struct, so a per-instance
        # attribute would be rejected as read-only.
        with mock.patch.object(BeamCoordinator, "_apply_mamba_cow", spy):
            self.coord._apply_survivors(self.group, self.tokens, self.parent_idx, 0)

        # The leader's row first, then the members -- the same order as
        # group.all_rows, which is the order remap_kv_mapping used.
        self.assertEqual(self.seen["rows_cpu"], self.group.all_rows.tolist())
        self.assertIs(self.seen["rows"], self.group.all_rows)
        self.assertTrue(torch.equal(self.seen["parent_idx"], self.parent_idx))

    def test_the_armed_rows_are_the_groups_rows(self):
        self.coord._apply_survivors(self.group, self.tokens, self.parent_idx, 0)

        self.assertEqual(
            self.pool.beam_mamba_cow._armed_rows, set(self.group.all_rows.tolist())
        )
        # ...and retiring withdraws exactly those.
        self.coord._retire_group(self.group)
        self.assertEqual(self.pool.beam_mamba_cow._armed_rows, set())

    def test_the_final_step_arms_nothing(self):
        # parent_idx is None on a length-terminated step: no row moves, so no
        # snapshot either -- and the retire has nothing to withdraw.
        self.coord._apply_survivors(self.group, self.tokens, None, 0)

        self.assertFalse(self.pool.beam_cow_read_mapping_pending())
        self.coord._retire_group(self.group)
        self.assertFalse(self.pool.beam_cow_read_mapping_pending())


class TestFailedSpawnLeavesNoRowsToWithdraw(CustomTestCase):
    """A rolled-back spawn must not leave a host row list behind: those rows
    went back to the pool and never armed anything."""

    def test_the_rollback_clears_the_host_rows(self):
        pool = _CpuHybridPool(size=8, mamba_size=1)
        pool.alloc_rows(1)  # the leader's own row
        leader = SimpleNamespace(
            kv=ReqKvInfo(
                kv_allocated_len=PROMPT_LEN,
                kv_committed_len=PROMPT_LEN,
                req_pool_idx=1,
            )
        )
        group = BeamGroup(beam_width=4, max_new_tokens=8)
        group.prompt_len = PROMPT_LEN
        coord = SimpleNamespace(req_to_token_pool=pool)

        with self.assertRaises(AssertionError):
            BeamCoordinator._spawn_member_rows(coord, group, leader)

        self.assertIsNone(group.all_rows)
        self.assertIsNone(group.all_rows_cpu)

    def test_a_spawned_group_caches_the_host_rows(self):
        pool = _CpuHybridPool(size=8, mamba_size=8)
        # The spawn forks the leader's state into the members; what the copy
        # does is not this file's subject, only that the rows were recorded.
        pool.mamba_pool = SimpleNamespace(
            replayssm_write_pos=None, copy_from=lambda src, dst: None
        )
        pool.alloc_rows(1)
        leader = SimpleNamespace(
            kv=ReqKvInfo(
                kv_allocated_len=PROMPT_LEN,
                kv_committed_len=PROMPT_LEN,
                req_pool_idx=1,
            )
        )
        group = BeamGroup(beam_width=4, max_new_tokens=8)
        group.prompt_len = PROMPT_LEN
        coord = SimpleNamespace(req_to_token_pool=pool)

        BeamCoordinator._spawn_member_rows(coord, group, leader)

        self.assertEqual(group.all_rows_cpu, group.all_rows.tolist())
        # Leader first, then the members -- the frontier order.
        self.assertEqual(group.all_rows_cpu[0], 1)


if __name__ == "__main__":
    unittest.main()
