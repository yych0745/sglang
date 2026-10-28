# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-group repetition penalties for beam search: frequency / presence /
repetition.

Why the group owns them and not the sampler: a beam leader is a normal Req
whose row params are neutralized (see fork.neutral_member_sampling_params), and
the k-1 member rows are not requests at all. The penalties therefore cannot
come from the batched penaltylib -- they live here, one state object per
BeamGroup, and are applied by the selection path that already scores every
frontier row.

SCOPE: only the tokens a branch has GENERATED are counted. The prompt is never
counted, so the first (prefill) selection sees all-zero counts and is scored
raw. This is deliberately narrower than HF's default repetition_penalty, which
runs inside the logits-processor list -- i.e. AFTER log_softmax -- and counts
the prompt too; the two are not interchangeable and HF is not an oracle here.

ORDER (fixed, applied to RAW logits before the logsumexp, so it is not
equivalent to penalizing log-probs):

    fp32 -> frequency (subtract f * count)
         -> presence  (subtract p where count > 0)
         -> repetition (seen positions only: negative logit * r,
                        non-negative logit / r)
         -> logsumexp -> (the trie mask, still AFTER the normalisation)

The additive pair runs before the multiplicative one, so a token can be pushed
across zero and then scaled on the far side.

TENSORS: one [frontier_rows, vocab] int32 count table per group, allocated
lazily on the selection stream (one row at prefill, k rows after the first
selection) and shared by all three penalties. `advance` re-parents it with an
index_select (never an in-place row loop), so a row may repeat a parent, be its
own parent, or swap parents with a sibling. Nothing here reads tensor contents
on the host: no D2H, no .item(), no data-dependent shape.
"""

from __future__ import annotations

from typing import Optional

import torch

# The no-op defaults. A request at these values gets no state at all, so the
# legacy selection path costs nothing extra.
NO_FREQUENCY = 0.0
NO_PRESENCE = 0.0
NO_REPETITION = 1.0


class BeamPenaltyState:
    """Generated-token counts and the three penalty scalars for one group.

    Pure host object until the first `apply`: the count table is sized from the
    logits it is handed, on the logits' own device, so it is allocated on the
    forward/relay stream inside the launch half rather than at admission.
    """

    def __init__(
        self,
        frequency_penalty: float = NO_FREQUENCY,
        presence_penalty: float = NO_PRESENCE,
        repetition_penalty: float = NO_REPETITION,
    ):
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)
        self.repetition_penalty = float(repetition_penalty)
        # [frontier_rows, vocab] int32 generated-token counts; None until the
        # first selection scores a row (lazy: the device and the vocab size are
        # only known there).
        self._counts: Optional[torch.Tensor] = None

    @property
    def required(self) -> bool:
        """Whether any of the three departs from its no-op default."""
        return (
            self.frequency_penalty != NO_FREQUENCY
            or self.presence_penalty != NO_PRESENCE
            or self.repetition_penalty != NO_REPETITION
        )

    @property
    def counts(self) -> Optional[torch.Tensor]:
        """The live count table, or None if no selection has scored yet.

        Host code may read its shape only: its contents live on device, and a
        read would synchronize. Tests read it; production must not.
        """
        return self._counts

    def apply(self, logits: torch.Tensor, row_start: int = 0) -> torch.Tensor:
        """Return a penalized fp32 copy of `logits` rows [row_start, ...).

        `logits` is a piece of the captured pre-sample logits (any dtype); it is
        never modified -- the returned tensor is a fresh fp32 work buffer, so
        the capture the sampler still owns stays raw. `row_start` is the piece's
        offset in the group's concatenated frontier rows, which is what lets the
        leader piece and the member tail share one count table.
        """
        rows = logits.shape[0]
        if rows == 0:
            return logits
        vocab = logits.shape[1]

        if self._counts is None:
            # First selection: one row at prefill, or `row_start + rows` if a
            # caller hands over the whole frontier in one piece.
            self._counts = torch.zeros(
                (row_start + rows, vocab), dtype=torch.int32, device=logits.device
            )
        elif self._counts.shape[0] < row_start + rows:
            # A shape this far off is a caller bug (counts are grown by
            # `advance`, never by apply); fail loudly rather than penalize a
            # slice of the wrong rows.
            raise AssertionError(
                f"beam penalty counts hold {self._counts.shape[0]} rows but "
                f"row offset {row_start} + {rows} rows were requested; the "
                f"frontier was advanced without the penalty state."
            )
        counts = self._counts[row_start : row_start + rows]

        work = logits.detach().to(torch.float32)
        if work.data_ptr() == logits.data_ptr():
            # `.to()` returned `self` (the source is already fp32): the copy
            # below is what keeps the caller's capture out of our writes.
            work = work.clone()

        if self.frequency_penalty != NO_FREQUENCY:
            work.sub_(counts.to(torch.float32), alpha=self.frequency_penalty)

        seen = None
        if self.presence_penalty != NO_PRESENCE:
            seen = counts > 0
            work.sub_(seen.to(torch.float32), alpha=self.presence_penalty)
        if self.repetition_penalty != NO_REPETITION:
            if seen is None:
                seen = counts > 0
            # r for seen tokens, 1.0 for unseen ones -- SELECTED between the
            # two, never rebuilt as seen * (r - 1) + 1: in fp32 that sum
            # cancels to exactly 0.0 for a legal tiny r (1e-8) and to a ~19%
            # -off 1.19e-7 for r = 1e-7, which turned the non-negative branch
            # into a division by zero (inf, or NaN at a zero logit, and a NaN
            # row once the logsumexp met the inf). Same shape / dtype / device
            # as `work`, so still fp32, still no D2H and no host branch on
            # tensor contents.
            scaling = torch.where(
                seen,
                torch.full_like(work, self.repetition_penalty),
                torch.ones_like(work),
            )
            work = torch.where(work < 0, work * scaling, work / scaling)
        return work

    def advance(self, parent_idx: torch.Tensor, next_tokens: torch.Tensor) -> None:
        """Re-parent the counts and count the tokens just selected.

        Launch-half work: `parent_idx` / `next_tokens` are the step's [k]
        selections in frontier-row order. The gather is an index_select into a
        NEW tensor, so repeated, self-, and swapped parents are all safe -- an
        in-place row loop would alias them.
        """
        counts = self._counts
        if counts is None:
            # Nothing has been scored yet, so there is no history to extend.
            return
        parent_idx = parent_idx.to(device=counts.device, dtype=torch.int64).reshape(-1)
        next_tokens = next_tokens.to(device=counts.device, dtype=torch.int64).reshape(-1)
        gathered = counts.index_select(0, parent_idx)
        gathered.scatter_add_(
            1,
            next_tokens.unsqueeze(1),
            torch.ones(
                (gathered.shape[0], 1), dtype=counts.dtype, device=counts.device
            ),
        )
        self._counts = gathered


def make_beam_penalty_state(
    frequency_penalty: float = NO_FREQUENCY,
    presence_penalty: float = NO_PRESENCE,
    repetition_penalty: float = NO_REPETITION,
) -> Optional[BeamPenaltyState]:
    """The group's penalty state, or None when all three are at their defaults.

    None means the selection path is bit-for-bit the legacy one: no count table
    is ever allocated and no extra fp32 work buffer is built.
    """
    if (
        frequency_penalty == NO_FREQUENCY
        and presence_penalty == NO_PRESENCE
        and repetition_penalty == NO_REPETITION
    ):
        return None
    return BeamPenaltyState(
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
