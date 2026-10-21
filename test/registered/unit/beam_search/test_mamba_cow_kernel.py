"""GPU tests for the conv1d copy-on-write read/write split.

The state machine tests cover the bookkeeping; these cover the thing that
bookkeeping is for: that the kernel actually READS the parent slot and WRITES
the child slot, so a reparented beam continues from its parent's state without
a separate copy.

The oracle is the plain kernel: running COW from parent P into child C must
produce exactly what running the normal path on a slot preloaded with P's state
produces.

One subtlety drives the test structure. Passing src forces the triton path,
because only triton implements the read/write split (see causal_conv1d.py); the
default path uses the fused CUDA kernel when it is available. So an oracle built
with src=None may run a *different* kernel, and the two agree only to ~5e-7, not
bitwise. Oracles here therefore pass an identity src to stay on triton, and the
one test that deliberately crosses backends uses a tolerance.
"""

import unittest

import torch

from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_update
from sglang.test.test_utils import CustomTestCase

_HAS_CUDA = torch.cuda.is_available()


def _rand_state(num_slots, dim, state_len, device, dtype):
    return torch.randn(num_slots, dim, state_len, device=device, dtype=dtype)


@unittest.skipUnless(_HAS_CUDA, "needs CUDA")
class TestConv1dCowSplit(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"
        self.dtype = torch.float32
        self.dim = 64
        self.width = 4
        self.state_len = self.width - 1
        self.num_slots = 8
        self.weight = torch.randn(
            self.dim, self.width, device=self.device, dtype=self.dtype
        )
        self.bias = torch.randn(self.dim, device=self.device, dtype=self.dtype)

    def _run(self, conv_state, x, dst, src=None):
        return causal_conv1d_update(
            x.clone(),
            conv_state,
            self.weight,
            bias=self.bias,
            activation="silu",
            conv_state_indices=dst,
            src_conv_state_indices=src,
        )

    def test_cow_matches_preloaded_parent(self):
        parent, child = 2, 5
        base = _rand_state(
            self.num_slots, self.dim, self.state_len, self.device, self.dtype
        )
        x = torch.randn(1, self.dim, device=self.device, dtype=self.dtype)
        dst = torch.tensor([child], dtype=torch.int32, device=self.device)
        src = torch.tensor([parent], dtype=torch.int32, device=self.device)

        cow_state = base.clone()
        cow_out = self._run(cow_state, x, dst, src)

        # Oracle: the copy the double buffer exists to avoid. Identity src keeps
        # it on the same (triton) kernel, so the comparison can be exact.
        ref_state = base.clone()
        ref_state[child] = ref_state[parent]
        ref_out = self._run(ref_state, x, dst, dst.clone())

        self.assertTrue(torch.equal(cow_out, ref_out))
        self.assertTrue(torch.equal(cow_state[child], ref_state[child]))

    def test_parent_state_is_not_modified(self):
        # Two beams fork off one parent in the same step. If the kernel wrote
        # through the read slot, the second child would start from state the
        # first already advanced.
        parent = 1
        base = _rand_state(
            self.num_slots, self.dim, self.state_len, self.device, self.dtype
        )
        parent_before = base[parent].clone()
        x = torch.randn(2, self.dim, device=self.device, dtype=self.dtype)
        dst = torch.tensor([4, 6], dtype=torch.int32, device=self.device)
        src = torch.tensor([parent, parent], dtype=torch.int32, device=self.device)

        state = base.clone()
        self._run(state, x, dst, src)

        self.assertTrue(torch.equal(state[parent], parent_before))

    def test_siblings_from_one_parent_stay_independent(self):
        parent = 0
        base = _rand_state(
            self.num_slots, self.dim, self.state_len, self.device, self.dtype
        )
        # Distinct inputs -> distinct resulting states. Sharing a slot would
        # collapse them to whichever program finished last.
        x = torch.stack(
            [
                torch.randn(self.dim, device=self.device, dtype=self.dtype),
                torch.randn(self.dim, device=self.device, dtype=self.dtype),
            ]
        )
        dst = torch.tensor([3, 7], dtype=torch.int32, device=self.device)
        src = torch.tensor([parent, parent], dtype=torch.int32, device=self.device)

        state = base.clone()
        self._run(state, x, dst, src)

        self.assertFalse(torch.equal(state[3], state[7]))

        for row, slot in enumerate((3, 7)):
            ref = base.clone()
            ref[slot] = ref[parent]
            one = dst[row : row + 1]
            self._run(ref, x[row : row + 1], one, one.clone())
            self.assertTrue(torch.equal(state[slot], ref[slot]))

    def test_none_src_matches_identity_src(self):
        # "No prune this tick" (src=None) and "pruned into the same slot"
        # (src==dst) must describe the same computation. They may run on
        # different backends -- None allows the fused CUDA kernel, src forces
        # triton -- so this is the one cross-backend comparison and it carries
        # a tolerance rather than claiming bitwise equality.
        base = _rand_state(
            self.num_slots, self.dim, self.state_len, self.device, self.dtype
        )
        x = torch.randn(3, self.dim, device=self.device, dtype=self.dtype)
        dst = torch.tensor([0, 1, 2], dtype=torch.int32, device=self.device)

        a_state = base.clone()
        a_out = self._run(a_state, x, dst, None)

        b_state = base.clone()
        b_out = self._run(b_state, x, dst, dst.clone())

        self.assertTrue(torch.allclose(a_out, b_out, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(a_state, b_state, atol=1e-5, rtol=1e-5))

    def test_identity_src_leaves_other_slots_alone(self):
        # A tick that touches rows 5 and 6 must not disturb any other slot.
        base = _rand_state(
            self.num_slots, self.dim, self.state_len, self.device, self.dtype
        )
        x = torch.randn(2, self.dim, device=self.device, dtype=self.dtype)
        dst = torch.tensor([5, 6], dtype=torch.int32, device=self.device)

        state = base.clone()
        self._run(state, x, dst, dst.clone())

        untouched = [i for i in range(self.num_slots) if i not in (5, 6)]
        self.assertTrue(torch.equal(state[untouched], base[untouched]))


if __name__ == "__main__":
    unittest.main()
