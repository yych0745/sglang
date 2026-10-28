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

a snapshot must not outlive its group's ownership of the rows. The mapping is
armed by a prune and meant to be consumed by the next decode. But a group can
end in the very commit that follows the arming (early stopping, abort,
retraction), and then no decode ever comes: the mapped rows go back to the
allocator and the next request admitted gets them. A snapshot naming them is
still in place, so that request's first decode reads the source slots its
predecessor detached and freed -- a use-after-free on someone else's state.
`forget_rows` is the undo: it drops a retiring group's own rows from the
snapshot without disturbing rows another group armed in the same tick.

Known remaining defect (not fixed here): the snapshot is cleared wholesale by
whatever decode forward consumes it, including a forward that carries none of
the rows it names. Pruning groups therefore have to share a tick today; making
the clear row-scoped is a separate change.
"""

from __future__ import annotations

from typing import Optional, Sequence

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
        # Rows the snapshot actually applies to; everything else reads live.
        self._pruned_mask: Optional[torch.Tensor] = None
        # Slots detached this tick, still under read by the in-flight kernel.
        self._source_slots: Optional[torch.Tensor] = None
        # Host-side mirror of the rows the snapshot is armed for, one entry per
        # `set_read_mapping` call. Kept so a group that ends before anything
        # consumed the snapshot can withdraw its own rows (see `forget_rows`)
        # without a device->host sync of `_pruned_mask`.
        self._armed_rows: set[int] = set()

    # ---- read side -------------------------------------------------------

    def read_mapping_for(
        self, req_indices: torch.Tensor, base_mapping: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Source slots for these req rows, or None when nothing was pruned.

        Only rows this tick actually pruned are served from the snapshot; every
        other row reads `base_mapping`, the live table. The snapshot is a full
        copy of the table as it looked mid-prune, so a row that was handed a
        slot *after* the snapshot was taken -- any request admitted later in
        the same tick -- would otherwise be sent to whatever slot it held
        before, i.e. state belonging to a different request.
        """
        if self._read_mapping is None:
            return None
        return torch.where(
            self._pruned_mask[req_indices],
            self._read_mapping[req_indices],
            base_mapping[req_indices],
        )

    def set_read_mapping(
        self,
        base_mapping: torch.Tensor,
        req_indices: torch.Tensor,
        src_slots: torch.Tensor,
        rows_cpu: Optional[Sequence[int]] = None,
    ) -> None:
        """Redirect `req_indices` to `src_slots` for this tick only.

        `base_mapping` is the pool's live table. It is cloned on first use, so
        the caller is free to reassign the pruned rows to their fresh slots
        immediately afterwards -- that write must NOT be visible here, or the
        kernel would read the very slot it is about to write. Later calls in
        the same tick reuse the existing snapshot for the same reason: the
        earlier group's rows have already been moved in `base_mapping`.

        `_pruned_mask` records which rows this applies to, so rows nobody
        pruned keep reading the live table (see `read_mapping_for`).

        `rows_cpu` is the same row numbers as `req_indices`, on the host. It is
        recorded in `_armed_rows` so `forget_rows` can withdraw them later.
        Production callers MUST pass it: the fallback reads `req_indices` back
        off the device, which is a sync, and it is only here for tests and
        cold paths.
        """
        if self._read_mapping is None:
            self._read_mapping = base_mapping.clone()
            self._pruned_mask = torch.zeros(
                base_mapping.shape[0], dtype=torch.bool, device=base_mapping.device
            )
        self._read_mapping[req_indices] = src_slots.to(
            dtype=self._read_mapping.dtype
        )
        self._pruned_mask[req_indices] = True
        self._armed_rows.update(req_indices.tolist() if rows_cpu is None else rows_cpu)

    def forget_rows(self, req_indices: torch.Tensor, rows_cpu: Sequence[int]) -> None:
        """Withdraw one group's rows from an unconsumed snapshot.

        The snapshot is supposed to live for exactly one thing: the decode that
        follows the prune that armed it, which reads the parent's state through
        it. But nothing guarantees that decode happens -- the group can finish
        in the very commit that follows (early stopping, abort, retraction),
        and then its rows go back to the allocator while the snapshot is still
        armed for them. The next request admitted gets those row numbers, and
        its first decode would read through the leftover mapping: slots the
        previous group detached and freed, i.e. a use-after-free on state that
        now belongs to somebody else.

        So a retiring group withdraws its rows here. Only its own rows: another
        group that pruned in the same tick shares this snapshot and is still
        waiting for its decode, so its entries stay. Once no group is left
        waiting the whole snapshot is dropped, which also reopens the cuda
        graph gate that `read_mapping_pending` closes.

        Device-side work only (`_pruned_mask` write) plus host-side set
        arithmetic -- no sync. A no-op when there is no snapshot, so it is safe
        to call on every retire.
        """
        if self._read_mapping is None:
            return
        self._pruned_mask[req_indices] = False
        self._armed_rows.difference_update(rows_cpu)
        if not self._armed_rows:
            # Nobody is waiting for this snapshot any more; keeping it would
            # only serve stale entries to whoever gets these rows next.
            self.clear_read_mapping()

    def clear_read_mapping(self) -> None:
        """Drop the mapping after one forward has consumed it."""
        self._read_mapping = None
        self._pruned_mask = None
        self._armed_rows.clear()

    def read_mapping_pending(self) -> bool:
        """Whether a prune's mapping is still unconsumed.

        The graph paths ask this before building replay metadata: they have no
        slot for the mapping, so a pending one reaching them means the prune's
        copy-on-write would be dropped and the rows would silently decode from
        the wrong state. Failing there beats replaying wrong.
        """
        return self._read_mapping is not None

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
        self._pruned_mask = None
        self._source_slots = None
        self._armed_rows.clear()
