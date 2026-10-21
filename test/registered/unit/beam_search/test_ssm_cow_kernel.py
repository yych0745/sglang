"""GPU tests for the SSM (GDN recurrent state) copy-on-write read/write split.

Companion to test_mamba_cow_kernel.py, which covers the conv half. Both halves
must honour the split: conv state alone is not the whole recurrent state, so a
beam reading the right conv window but the wrong SSM state is still wrong.

Same oracle: COW from parent P into child C must equal preloading C with P's
state and running the plain kernel. Here both sides run the same triton kernel,
so the comparison is exact.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

_HAS_CUDA = torch.cuda.is_available()

if _HAS_CUDA:
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as _decode,
    )


@unittest.skipUnless(_HAS_CUDA, "needs CUDA")
class TestSsmCowSplit(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"
        self.dtype = torch.float32
        self.H = 2
        self.HV = 2
        self.K = 32
        self.V = 32
        self.num_slots = 8
        self.scale = self.K**-0.5
        self.A_log = torch.randn(self.HV, device=self.device, dtype=self.dtype)
        self.dt_bias = torch.randn(self.HV, device=self.device, dtype=self.dtype)

    def _inputs(self, batch):
        qkv_dim = 2 * self.H * self.K + self.HV * self.V
        return (
            torch.randn(batch, qkv_dim, device=self.device, dtype=self.dtype),
            torch.randn(batch, self.HV, device=self.device, dtype=self.dtype),
            torch.randn(batch, self.HV, device=self.device, dtype=self.dtype),
        )

    def _states(self):
        return torch.randn(
            self.num_slots,
            self.HV,
            self.V,
            self.K,
            device=self.device,
            dtype=self.dtype,
        )

    def _run(self, states, mixed_qkv, a, b, dst, src=None):
        out = mixed_qkv.new_empty(mixed_qkv.shape[0], 1, self.HV, self.V)
        _decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.scale,
            initial_state=states,
            out=out,
            ssm_state_indices=dst,
            src_ssm_state_indices=src,
            use_qk_l2norm_in_kernel=True,
        )
        return out

    def test_cow_matches_preloaded_parent(self):
        parent, child = 3, 6
        base = self._states()
        mixed_qkv, a, b = self._inputs(1)
        dst = torch.tensor([child], dtype=torch.int32, device=self.device)
        src = torch.tensor([parent], dtype=torch.int32, device=self.device)

        cow_states = base.clone()
        cow_out = self._run(cow_states, mixed_qkv, a, b, dst, src)

        ref_states = base.clone()
        ref_states[child] = ref_states[parent]
        ref_out = self._run(ref_states, mixed_qkv, a, b, dst)

        self.assertTrue(torch.equal(cow_out, ref_out))
        self.assertTrue(torch.equal(cow_states[child], ref_states[child]))

    def test_parent_state_is_not_modified(self):
        parent = 1
        base = self._states()
        parent_before = base[parent].clone()
        mixed_qkv, a, b = self._inputs(2)
        dst = torch.tensor([4, 5], dtype=torch.int32, device=self.device)
        src = torch.tensor([parent, parent], dtype=torch.int32, device=self.device)

        states = base.clone()
        self._run(states, mixed_qkv, a, b, dst, src)

        self.assertTrue(torch.equal(states[parent], parent_before))

    def test_siblings_from_one_parent_stay_independent(self):
        parent = 0
        base = self._states()
        mixed_qkv, a, b = self._inputs(2)
        dst = torch.tensor([2, 7], dtype=torch.int32, device=self.device)
        src = torch.tensor([parent, parent], dtype=torch.int32, device=self.device)

        states = base.clone()
        self._run(states, mixed_qkv, a, b, dst, src)

        self.assertFalse(torch.equal(states[2], states[7]))
        for row, slot in enumerate((2, 7)):
            ref = base.clone()
            ref[slot] = ref[parent]
            one = dst[row : row + 1]
            self._run(
                ref, mixed_qkv[row : row + 1], a[row : row + 1], b[row : row + 1], one
            )
            self.assertTrue(torch.equal(states[slot], ref[slot]))

    def test_none_src_equals_identity_src(self):
        base = self._states()
        mixed_qkv, a, b = self._inputs(3)
        dst = torch.tensor([0, 1, 2], dtype=torch.int32, device=self.device)

        a_states = base.clone()
        a_out = self._run(a_states, mixed_qkv, a, b, dst, None)
        b_states = base.clone()
        b_out = self._run(b_states, mixed_qkv, a, b, dst, dst.clone())

        # Same kernel either way here, so this one is exact.
        self.assertTrue(torch.equal(a_out, b_out))
        self.assertTrue(torch.equal(a_states, b_states))

    def test_negative_src_slot_emits_zeros(self):
        # -1 means "skip". A skipped source has no parent state to continue
        # from, so the row must not read slot -1.
        base = self._states()
        mixed_qkv, a, b = self._inputs(2)
        dst = torch.tensor([4, 5], dtype=torch.int32, device=self.device)
        src = torch.tensor([-1, 3], dtype=torch.int32, device=self.device)

        states = base.clone()
        out = self._run(states, mixed_qkv, a, b, dst, src)

        self.assertTrue(torch.equal(out[0], torch.zeros_like(out[0])))
        self.assertFalse(torch.equal(out[1], torch.zeros_like(out[1])))

    def test_mismatched_src_shape_is_rejected(self):
        base = self._states()
        mixed_qkv, a, b = self._inputs(2)
        dst = torch.tensor([4, 5], dtype=torch.int32, device=self.device)
        bad = torch.tensor([1], dtype=torch.int32, device=self.device)

        with self.assertRaises(ValueError):
            self._run(base.clone(), mixed_qkv, a, b, dst, bad)


if __name__ == "__main__":
    unittest.main()
