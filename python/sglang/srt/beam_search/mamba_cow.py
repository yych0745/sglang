"""Copy-on-write double buffer for mamba/SSM state across a beam prune.

Why beam search needs this
--------------------------
Attention KV is reparented by rewriting `req_to_token`: a member row that
survives under a new parent just points at the parent's slots, and no KV data
moves (see fork.remap_kv_mapping). SSM state cannot be reparented that way --
it is not indexed by position, it is one opaque per-slot recurrent state. Two
survivors of the same parent therefore cannot share a slot: the first decode
step would write through it and corrupt the other.

The naive fix is to copy the parent's state into each survivor's slot on every
prune, which is `beam_width` state-sized copies per decode step.

The double buffer avoids the copy by splitting read from write for exactly one
tick. On a prune the caller:

  1. notes each pruned row's current (parent) slot as its READ source,
  2. allocates a fresh slot per pruned row and installs it as the row's slot
     in the pool, i.e. its WRITE target,
  3. stages the row's previous slot for release.

The kernel then reads the old state and writes the new one, which is exactly
the copy -- fused into the step's own write, so it costs nothing extra.

Two invariants this class exists to hold
----------------------------------------
read mapping is one-shot. It describes a single forward. A later tick that
does not prune must read and write the same slot; a leftover mapping would send
it back to a parent slot whose state is now one step stale. The backend clears
it right after consuming it, and `read_mapping_for` returns None when absent so
the backend can fall back to the pool's own mapping.

source slots are freed one tick late. The staged slots are still being read by
the kernel this tick. Returning them to the allocator now lets the next
allocation hand the same slot to another request, which would then overwrite
state the current step still needs. They are released at the top of the next
decode instead.
"""

from __future__ import annotations

from typing import Optional

import torch


class BeamMambaCow:
    """Per-pool state for the beam-prune double buffer.

    Holds no slots of its own: the read mapping is a snapshot of the pool's
    req-row -> mamba-slot table, and the source slots are ids the caller has
    already detached from that table and will hand back to the allocator.
    """

    def __init__(self) -> None:
        # Snapshot of req_index_to_mamba_index_mapping with the pruned rows
        # redirected to their parents. None means "no prune this tick".
        self._read_mapping: Optional[torch.Tensor] = None
        # Slots detached this tick, still under read by the in-flight kernel.
        self._source_slots: Optional[torch.Tensor] = None

    # ---- read side -------------------------------------------------------

    def read_mapping_for(self, req_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Source slots for these req rows, or None when nothing was pruned."""
        if self._read_mapping is None:
            return None
        return self._read_mapping[req_indices]

    def set_read_mapping(
        self,
        base_mapping: torch.Tensor,
        req_indices: torch.Tensor,
        src_slots: torch.Tensor,
    ) -> None:
        """Redirect `req_indices` to `src_slots` for this tick only.

        `base_mapping` is the pool's live table. It is cloned on first use, so
        the caller is free to reassign the pruned rows to their fresh slots
        immediately afterwards -- that write must NOT be visible here, or the
        kernel would read the very slot it is about to write. Later calls in
        the same tick reuse the existing snapshot for the same reason: the
        earlier group's rows have already been moved in `base_mapping`.
        """
        if self._read_mapping is None:
            self._read_mapping = base_mapping.clone()
        self._read_mapping[req_indices] = src_slots.to(
            dtype=self._read_mapping.dtype
        )

    def clear_read_mapping(self) -> None:
        """Drop the mapping after one forward has consumed it."""
        self._read_mapping = None

    # ---- free side -------------------------------------------------------

    def stage_source_slots(
        self, slots: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> None:
        """Queue slots for release on the next tick. Accumulates, since several
        groups can prune in the same decode."""
        if slots.numel() == 0:
            return
        if dtype is not None:
            slots = slots.to(dtype=dtype)
        if self._source_slots is None:
            self._source_slots = slots
        else:
            self._source_slots = torch.cat((self._source_slots, slots))

    def take_source_slots(self) -> Optional[torch.Tensor]:
        """Hand back the staged slots and forget them.

        Take-once on purpose: returning the same ids twice would let the
        allocator issue one slot to two requests.
        """
        slots, self._source_slots = self._source_slots, None
        return slots

    # ---- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        """Forget both halves. Used by the pool's own clear()."""
        self._read_mapping = None
        self._source_slots = None
