"""Unit tests for the beam spawn's SSM fork.

A beam spawn gives each member row a mamba slot of its own
(``alloc_mamba_for_rows``) -- but an allocator-issued slot is only *ownership*:
what it holds is whatever the previous tenant left, so a member's first decode
tick used to run on a state that had never seen the prompt. ``fork_mamba_from_row``
is the second half: it fills the freshly owned slots with the leader's
post-prefill state.

Covered here:

  - content:     every member slot really holds the leader's state, not just an
                 index pointing somewhere;
  - ownership:   the member slots are distinct from each other and from the
                 leader's -- a copy, not an alias of the leader's slot;
  - isolation:   writing a member's slot leaves the leader's alone;
  - addressing:  one repeated source, virtual -> physical translation on both
                 sides, and the leader accepted as an int or a 0-d tensor;
  - no-op:       a non-hybrid pool spawns without mamba at all, and
                 beam_width == 1 spawns no members;
  - rollback:    a short allocator still raises before any fork runs;
  - ReplaySSM:   the flushed-source guard fires only in debug mode.

Pure CPU tensor bookkeeping: the pool is the CPU fake from the sibling test
modules, so the real ``_spawn_member_rows`` / ``fork_mamba_from_row`` bodies run.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_mamba_cow_write_side import _CpuStateCopyPool, _CpuVirtualSlotPool
from test_member_row_mamba import _CpuHybridPool

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

LEADER_ROW = 1
PROMPT_LEN = 2


def _leader(leader_row: int = LEADER_ROW) -> SimpleNamespace:
    return SimpleNamespace(
        kv=ReqKvInfo(
            kv_allocated_len=PROMPT_LEN,
            kv_committed_len=PROMPT_LEN,
            req_pool_idx=leader_row,
        )
    )


def _spawn(
    pool,
    *,
    beam_width: int = 4,
    leader_row: int = LEADER_ROW,
    alloc_leader_slot: bool = True,
):
    """Run the real spawn against `pool`, the leader owning a slot beforehand."""
    leader_slot = None
    if alloc_leader_slot:
        leader_slot = pool.alloc_mamba_for_rows(
            torch.tensor([leader_row], dtype=torch.int64)
        )
    group = SimpleNamespace(beam_width=beam_width, prompt_len=PROMPT_LEN)
    # _spawn_member_rows only ever touches req_to_token_pool on self.
    BeamCoordinator._spawn_member_rows(
        SimpleNamespace(req_to_token_pool=pool), group, _leader(leader_row)
    )
    return group, leader_slot


def _seed(pool):
    """Every slot gets a distinct full-width state: slot s holds 10*s + dim."""
    temporal = pool.mamba_pool.temporal
    dims = torch.arange(temporal.shape[2], dtype=temporal.dtype)
    for s in range(temporal.shape[1]):
        temporal[:, s, :] = float(s) * 10.0 + dims


def _seeded_state(pool, slot: int) -> torch.Tensor:
    """What `_seed` put in `slot`, i.e. what a fork-free spawn would leave."""
    temporal = pool.mamba_pool.temporal
    dims = torch.arange(temporal.shape[2], dtype=temporal.dtype)
    return (float(slot) * 10.0 + dims).expand(1, temporal.shape[2])


class TestSpawnForksTheLeadersState(CustomTestCase):
    """The state itself has to move, not just an index."""

    def setUp(self):
        self.pool = _CpuStateCopyPool(size=8, mamba_size=8)
        # Seeded before the spawn: whatever the allocator hands the members is
        # stale, so a spawn that only allocates leaves visibly wrong state.
        _seed(self.pool)
        self.group, self.leader_slot = _spawn(self.pool)
        self.members = self.pool.get_mamba_indices(self.group.member_rows)

    def test_every_member_slot_holds_the_leaders_state(self):
        leader_state = self.pool.mamba_pool.temporal[:, int(self.leader_slot)]
        for slot in self.members.tolist():
            self.assertTrue(
                torch.equal(self.pool.mamba_pool.temporal[:, slot], leader_state),
                f"member slot {slot} does not hold the leader's state",
            )

    def test_the_member_state_is_not_the_stale_content_of_its_own_slot(self):
        # Slot s was seeded with 10*s + dim. A spawn that allocated without
        # forking leaves each member on exactly that value.
        for slot in self.members.tolist():
            self.assertFalse(
                torch.equal(
                    self.pool.mamba_pool.temporal[:, slot],
                    _seeded_state(self.pool, slot),
                )
            )

    def test_members_own_distinct_slots_none_of_them_the_leaders(self):
        # Independent ownership: two members on one slot would write through
        # each other, and a member on the leader's slot would corrupt it.
        slots = self.members.tolist()
        self.assertEqual(len(set(slots)), len(slots))
        self.assertNotIn(int(self.leader_slot), slots)
        self.assertNotIn(0, slots)

    def test_writing_a_member_does_not_move_the_leader(self):
        # A copy, not an alias: if the fork had installed a shared view, this
        # write would show up in the leader's slot too.
        before = self.pool.mamba_pool.temporal[:, int(self.leader_slot)].clone()

        self.pool.mamba_pool.temporal[:, int(self.members[0])] = -1234.0

        self.assertTrue(
            torch.equal(
                self.pool.mamba_pool.temporal[:, int(self.leader_slot)], before
            )
        )

    def test_the_copy_reads_one_source_for_every_destination(self):
        # k-1 destinations, one repeated (read-only) source.
        src, dst = self.pool.mamba_pool.calls[-1]
        self.assertEqual(src, [int(self.leader_slot)] * self.members.numel())
        self.assertEqual(dst, self.members.tolist())

    def test_src_and_dst_are_disjoint(self):
        # What buys the fused copy's correctness: the destinations were
        # allocated a moment earlier, so none of them is the leader's slot.
        src, dst = self.pool.mamba_pool.calls[-1]
        self.assertEqual(set(src) & set(dst), set())


class TestSpawnForkAddressing(CustomTestCase):
    """Which slots the copy names, and how it names them."""

    def test_the_copy_addresses_physical_slots(self):
        # The table holds virtual ids; copy_from addresses the state tensors
        # physically. On a unified pool those differ, so an untranslated call
        # would silently copy from an unrelated slot.
        pool = _CpuVirtualSlotPool(size=8, mamba_size=8)
        _seed(pool)
        group, leader_slot = _spawn(pool)

        members = pool.get_mamba_indices(group.member_rows)
        src, dst = pool.mamba_pool.calls[-1]
        self.assertEqual(src, [int(leader_slot) + pool.OFFSET] * members.numel())
        self.assertEqual(dst, [int(s) + pool.OFFSET for s in members.tolist()])
        # And the state really landed at the translated slots.
        leader_state = pool.mamba_pool.temporal[:, int(leader_slot) + pool.OFFSET]
        for slot in members.tolist():
            self.assertTrue(
                torch.equal(
                    pool.mamba_pool.temporal[:, slot + pool.OFFSET], leader_state
                )
            )

    def test_the_leader_can_be_given_as_an_int_or_a_zero_dim_tensor(self):
        # The coordinator passes req_pool_idx (an int); the pool method also
        # accepts a 0-d index tensor -- both must name the same slot.
        for src_row in (LEADER_ROW, torch.tensor(LEADER_ROW)):
            with self.subTest(src_row=src_row):
                pool = _CpuStateCopyPool(size=8, mamba_size=8)
                _seed(pool)
                leader_slot = pool.alloc_mamba_for_rows(
                    torch.tensor([LEADER_ROW], dtype=torch.int64)
                )
                rows = torch.tensor([2, 3], dtype=torch.int64)
                member_slots = pool.alloc_mamba_for_rows(rows)

                pool.fork_mamba_from_row(src_row, rows)

                leader_state = pool.mamba_pool.temporal[:, int(leader_slot)]
                for slot in member_slots.tolist():
                    self.assertTrue(
                        torch.equal(pool.mamba_pool.temporal[:, slot], leader_state)
                    )

    def test_no_member_rows_forks_nothing(self):
        # beam_width == 1: alloc_rows(0) hands out no row, and an empty fork
        # must not even call copy_from.
        pool = _CpuStateCopyPool(size=8, mamba_size=8)
        _seed(pool)

        group, leader_slot = _spawn(pool, beam_width=1)

        self.assertEqual(int(group.member_rows.numel()), 0)
        self.assertEqual(pool.mamba_pool.calls, [])
        # Only the leader's slot is out.
        self.assertEqual(pool.mamba_allocator.available_size(), 7)

    def test_a_repeated_source_is_safe_for_every_destination(self):
        # The fused kernel reads src[i] and writes dst[i] in one program per
        # pair: a repeated source is a read-only gather, so all k-1 members
        # really do get identical state.
        pool = _CpuStateCopyPool(size=8, mamba_size=8)
        _seed(pool)
        group, leader_slot = _spawn(pool, beam_width=5)

        members = pool.get_mamba_indices(group.member_rows)
        states = [
            pool.mamba_pool.temporal[:, int(s)].clone() for s in members.tolist()
        ]
        for state in states[1:]:
            self.assertTrue(torch.equal(state, states[0]))


class TestSpawnForkOnOtherPools(CustomTestCase):
    """What a pool without the hybrid mamba API does."""

    def test_a_non_hybrid_pool_spawns_without_mamba(self):
        # A plain pool has neither alloc_mamba_for_rows nor the fork: the spawn
        # must still hand out rows and alias the prompt KV.
        pool = ReqToTokenPool(
            size=8, max_context_len=4, device="cpu", enable_memory_saver=False
        )

        group, _ = _spawn(pool, alloc_leader_slot=False)

        self.assertEqual(group.member_rows.tolist(), [6, 7, 8])
        self.assertEqual(group.all_rows.tolist(), [LEADER_ROW, 6, 7, 8])

    def test_the_probe_skips_a_plain_pool(self):
        self.assertTrue(
            hasattr(_CpuHybridPool(size=4, mamba_size=4), "fork_mamba_from_row")
        )
        self.assertFalse(
            hasattr(
                ReqToTokenPool(
                    size=4, max_context_len=4, device="cpu", enable_memory_saver=False
                ),
                "fork_mamba_from_row",
            )
        )


class TestSpawnForkRollback(CustomTestCase):
    """A short allocator must fail exactly as it did before the fork existed."""

    def test_alloc_failure_raises_before_any_fork(self):
        pool = _CpuStateCopyPool(size=8, mamba_size=2)
        pool.alloc_rows(1)  # the leader's own row
        pool.alloc_mamba_for_rows(torch.tensor([LEADER_ROW], dtype=torch.int64))
        group = SimpleNamespace(beam_width=4, prompt_len=PROMPT_LEN)
        rows_before = sorted(pool.free_slots)

        with self.assertRaises(AssertionError) as ctx:
            BeamCoordinator._spawn_member_rows(
                SimpleNamespace(req_to_token_pool=pool), group, _leader()
            )

        self.assertIn("mamba slots", str(ctx.exception))
        # Nothing was copied: there is nothing to inherit into.
        self.assertEqual(pool.mamba_pool.calls, [])
        # Unchanged rollback: the req rows go back and the group keeps no rows.
        self.assertEqual(sorted(pool.free_slots), rows_before)
        self.assertIsNone(group.member_rows)
        self.assertIsNone(group.member_rows_cpu)
        self.assertIsNone(group.all_rows)


class TestSpawnForkReplaySSM(CustomTestCase):
    """The flushed-source guard: debug-only, because it syncs.

    It is an assert and not a RuntimeError because the invariant holds by
    construction -- a slot starts at write_pos 0 and only decode advances the
    cursor, so a leader that has just prefilled has an empty ring. The prune
    path (``_apply_mamba_state_copy``) refuses ReplaySSM outright; the spawn has
    no such reason, so it checks only in debug mode.
    """

    def setUp(self):
        self.pool = _CpuStateCopyPool(size=8, mamba_size=8, replayssm=True)

    def test_a_flushed_source_is_accepted_without_a_host_sync(self):
        # write_pos == 0 for a freshly issued slot: the normal case, and the
        # one that must not grow a host sync in production.
        self.assertFalse(getattr(self.pool.mamba_pool, "debug_memory_pool", False))

        group, leader_slot = _spawn(self.pool)

        self.assertEqual(int(self.pool.mamba_pool.replayssm_write_pos[leader_slot]), 0)
        self.assertEqual(len(self.pool.mamba_pool.calls), 1)

    def test_an_unflushed_source_is_caught_in_debug_mode(self):
        self.pool.mamba_pool.debug_memory_pool = True
        leader_slot = self.pool.alloc_mamba_for_rows(
            torch.tensor([LEADER_ROW], dtype=torch.int64)
        )
        self.pool.mamba_pool.replayssm_write_pos[leader_slot] = 3
        group = SimpleNamespace(beam_width=4, prompt_len=PROMPT_LEN)

        with self.assertRaises(AssertionError) as ctx:
            BeamCoordinator._spawn_member_rows(
                SimpleNamespace(req_to_token_pool=self.pool), group, _leader()
            )

        self.assertIn("ReplaySSM", str(ctx.exception))
        # Reported rather than silently copied: no state was moved.
        self.assertEqual(self.pool.mamba_pool.calls, [])

    def test_debug_mode_accepts_a_flushed_source(self):
        # The guard is not a blanket refusal -- it fires on the cursor value.
        self.pool.mamba_pool.debug_memory_pool = True

        group, leader_slot = _spawn(self.pool)

        self.assertEqual(len(self.pool.mamba_pool.calls), 1)


if __name__ == "__main__":
    unittest.main()
