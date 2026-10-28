"""Unit tests for the mamba/SSM copy-on-write double buffer used by beam search.

These cover the state machine only (pure tensor bookkeeping, CPU): the read
mapping's one-shot lifetime, the snapshot-vs-alias question, and the one-tick
delay on freeing source slots.
"""

import unittest

import torch

from sglang.srt.beam_search.mamba_cow import BeamMambaCow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReadMapping(CustomTestCase):
    def setUp(self):
        # req row i currently owns mamba slot 10 + i.
        self.base = torch.arange(10, 16, dtype=torch.int32)
        self.cow = BeamMambaCow()

    def test_absent_until_set(self):
        # A tick that never prunes must read and write the same slot, which the
        # backend expresses by falling back to the pool's own mapping.
        self.assertIsNone(self.cow.read_mapping_for(torch.tensor([0, 1]), self.base))

    def test_set_then_read_returns_source_slots(self):
        rows = torch.tensor([2, 3], dtype=torch.int64)
        src = torch.tensor([11, 11], dtype=torch.int32)

        self.cow.set_read_mapping(self.base, rows, src)

        # The pruned rows read their parent's slot...
        got = self.cow.read_mapping_for(rows, self.base)
        self.assertTrue(torch.equal(got, src))
        # ...and every other row still reads its own (from the live table).
        untouched = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
        self.assertTrue(
            torch.equal(self.cow.read_mapping_for(untouched, self.base), self.base[untouched])
        )

    def test_read_mapping_is_one_shot(self):
        rows = torch.tensor([0], dtype=torch.int64)
        self.cow.set_read_mapping(self.base, rows, torch.tensor([13], dtype=torch.int32))
        self.assertIsNotNone(self.cow.read_mapping_for(rows, self.base))

        self.cow.clear_read_mapping()

        # The backend clears after consuming it. If the mapping survived into
        # the next forward, a tick that did not prune would read a stale parent
        # slot instead of the state it just wrote.
        self.assertIsNone(self.cow.read_mapping_for(rows, self.base))

    def test_rows_admitted_after_the_snapshot_read_live(self):
        """A row that changed slot after the snapshot must not use it.

        The snapshot copies the whole table mid-prune. Under concurrency a
        request admitted later in the same tick gets a fresh slot, and serving
        it from the snapshot would point it at the slot it held before -- some
        other request's state. This showed up end to end as beams degenerating
        into repeated tokens whenever several groups ran together.
        """
        pruned = torch.tensor([1], dtype=torch.int64)
        self.cow.set_read_mapping(self.base, pruned, torch.tensor([10], dtype=torch.int32))

        latecomer = torch.tensor([4], dtype=torch.int64)
        self.base[latecomer] = 99  # admitted after the snapshot was taken

        self.assertEqual(int(self.cow.read_mapping_for(latecomer, self.base)[0]), 99)
        # the pruned row still reads its parent
        self.assertEqual(int(self.cow.read_mapping_for(pruned, self.base)[0]), 10)

    def test_set_snapshots_the_base_mapping(self):
        rows = torch.tensor([2], dtype=torch.int64)
        self.cow.set_read_mapping(self.base, rows, torch.tensor([10], dtype=torch.int32))

        # The caller reassigns the pruned rows to their fresh slots right after
        # setting the read mapping. That write must not reach back into the
        # read mapping, or the kernel would read the slot it is about to write.
        self.base[rows] = 99

        self.assertEqual(int(self.cow.read_mapping_for(rows, self.base)[0]), 10)

    def test_second_set_reuses_the_existing_snapshot(self):
        first = torch.tensor([1], dtype=torch.int64)
        second = torch.tensor([4], dtype=torch.int64)
        self.cow.set_read_mapping(self.base, first, torch.tensor([10], dtype=torch.int32))
        self.base[second] = 42
        self.cow.set_read_mapping(self.base, second, torch.tensor([12], dtype=torch.int32))

        # Two groups pruned in the same tick both land in one mapping, and the
        # second set must not re-snapshot a base that the first already moved.
        self.assertEqual(int(self.cow.read_mapping_for(first, self.base)[0]), 10)
        self.assertEqual(int(self.cow.read_mapping_for(second, self.base)[0]), 12)


class TestSourceSlots(CustomTestCase):
    def setUp(self):
        self.cow = BeamMambaCow()

    def test_nothing_staged_yields_nothing(self):
        self.assertIsNone(self.cow.take_source_slots())

    def test_staged_slots_come_back_once(self):
        slots = torch.tensor([4, 5], dtype=torch.int64)
        self.cow.stage_source_slots(slots)

        self.assertTrue(torch.equal(self.cow.take_source_slots(), slots))
        # Taking twice would hand the allocator the same slot twice; the second
        # caller would then overwrite a live request's state.
        self.assertIsNone(self.cow.take_source_slots())

    def test_stages_accumulate_within_a_tick(self):
        self.cow.stage_source_slots(torch.tensor([1], dtype=torch.int64))
        self.cow.stage_source_slots(torch.tensor([2, 3], dtype=torch.int64))

        self.assertTrue(
            torch.equal(
                self.cow.take_source_slots(), torch.tensor([1, 2, 3], dtype=torch.int64)
            )
        )

    def test_empty_stage_is_ignored(self):
        self.cow.stage_source_slots(torch.tensor([], dtype=torch.int64))
        self.assertIsNone(self.cow.take_source_slots())

    def test_stage_casts_to_the_allocator_dtype(self):
        self.cow.stage_source_slots(
            torch.tensor([7], dtype=torch.int32), dtype=torch.int64
        )
        self.assertEqual(self.cow.take_source_slots().dtype, torch.int64)


class TestReset(CustomTestCase):
    def test_reset_drops_both_halves(self):
        cow = BeamMambaCow()
        base = torch.arange(4, dtype=torch.int32)
        cow.set_read_mapping(base, torch.tensor([0], dtype=torch.int64),
                             torch.tensor([3], dtype=torch.int32))
        cow.stage_source_slots(torch.tensor([2], dtype=torch.int64))

        cow.reset()

        self.assertIsNone(cow.read_mapping_for(torch.tensor([0], dtype=torch.int64), base))
        self.assertIsNone(cow.take_source_slots())


if __name__ == "__main__":
    unittest.main()
