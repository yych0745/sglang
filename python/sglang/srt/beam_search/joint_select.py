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
"""Joint token selection for beam search: pure tensor functions.

Contract (overlap / cuda-graph ready): tensor-in / tensor-out, fixed output
shapes for a given (num_rows, num_candidates, beam_width) signature, no D2H
sync, no data-dependent host branches (the only python branch keys on a
static tensor shape). The caller owns the single sync point per step.

Selection semantics: walk the top num_candidates extensions in descending cumulative-logprob order; stop-token
candidates finish, non-stop candidates survive; a candidate is examined only
while fewer than beam_width survivors precede it in score order.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch


class SelectResult(msgspec.Struct):
    """Fixed-shape outputs of one expansion step. Valid in [0, num_survivors)
    and [0, num_finished); past that, dump-slot zeros that must not be read."""

    next_tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64, row index into the input frontier
    new_cum_logprobs: torch.Tensor  # [beam_width] float32
    num_survivors: torch.Tensor  # [] int64; < beam_width means the group must finish
    fin_tokens: torch.Tensor  # [num_candidates] int64
    fin_parent_idx: torch.Tensor  # [num_candidates] int64
    fin_cum_logprobs: torch.Tensor  # [num_candidates] float32
    num_finished: torch.Tensor  # [] int64


class FinalSelect(msgspec.Struct):
    """Top-beam_width candidates of a length-terminated step (all finished)."""

    tokens: torch.Tensor  # [beam_width] int64
    parent_idx: torch.Tensor  # [beam_width] int64
    cum_logprobs: torch.Tensor  # [beam_width] float32


def _scatter_fixed(
    src: torch.Tensor,
    slot: torch.Tensor,
    size: int,
    fill: Optional[float] = None,
) -> torch.Tensor:
    # Fixed-shape compaction: element i lands at slot[i]; non-selected elements
    # all target the dump slot `size`, which is sliced away.
    #
    # `fill` is what an unfilled position carries. It must stay 0 for integer
    # payloads (token ids / parent rows) and become -inf for logprobs: a padded
    # slot left at 0.0 would outrank every real candidate, whose cumulative
    # logprob is a sum of non-positive terms.
    if fill is None:
        buf = src.new_zeros(size + 1)
    else:
        buf = src.new_full((size + 1,), fill)
    buf.scatter_(0, slot, src)
    return buf[:size]


def _ranked_candidates(cum_logprobs, top_logprobs, top_tokens, num_out):
    num_candidates = top_logprobs.shape[1]
    scores = cum_logprobs.unsqueeze(1) + top_logprobs
    cand_scores, cand_idx = scores.reshape(-1).topk(num_out, sorted=True)
    parent = cand_idx // num_candidates
    tokens = top_tokens.reshape(-1).gather(0, cand_idx)
    return cand_scores, parent, tokens


def joint_select(
    cum_logprobs: torch.Tensor,  # [num_rows] float32, frontier cumulative logprobs
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    stop_token_ids: torch.Tensor,  # [num_stop] int64, may be empty (ignore_eos)
    beam_width: int,
) -> SelectResult:
    num_candidates = top_logprobs.shape[1]
    k = beam_width

    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, num_candidates
    )

    # A candidate carried by a -inf logprob is not a real extension: either the
    # row itself is a dummy seated to keep the shape fixed, or a constraint
    # masked it out. Such candidates never survive and never finish, so a short
    # step degrades to padding instead of a bogus "no live beams" finish.
    finite = torch.isfinite(cand_scores)
    if stop_token_ids.numel() > 0:  # static-shape branch, constant under capture
        is_stop = torch.isin(tokens, stop_token_ids)
    else:
        is_stop = torch.zeros_like(finite)
    usable = finite & ~is_stop

    # Survivor = usable candidates, lowest rank first, up to k. Finished = stop
    # candidates met while survivors are still being collected.
    usable_rank = usable.long().cumsum(0)  # 1-based at usable positions
    survivor = usable & (usable_rank <= k)
    # Examined while fewer than k survivors strictly precede the candidate.
    examined = (usable_rank - usable.long()) < k
    # `finite` matters here too: a dummy row's stop candidate also carries -inf,
    # and a short step leaves examined slots open for it to slip into. Letting it
    # through would publish a padding artifact as a finished beam.
    finished = is_stop & finite & examined
    fin_rank = finished.long().cumsum(0)

    surv_slot = torch.where(survivor, usable_rank - 1, k)
    fin_slot = torch.where(finished, fin_rank - 1, num_candidates)

    return SelectResult(
        # Integer payloads pad with 0: a dummy's parent row simply aliases the
        # best real row, which keeps every req_to_token mapping valid (the
        # group-wide dedup free already expects rows to share slots).
        next_tokens=_scatter_fixed(tokens, surv_slot, k),
        parent_idx=_scatter_fixed(parent, surv_slot, k),
        # Logprob padding must be -inf so dummies sort last forever.
        new_cum_logprobs=_scatter_fixed(cand_scores, surv_slot, k, fill=float("-inf")),
        num_survivors=survivor.long().sum(),
        fin_tokens=_scatter_fixed(tokens, fin_slot, num_candidates),
        fin_parent_idx=_scatter_fixed(parent, fin_slot, num_candidates),
        fin_cum_logprobs=_scatter_fixed(
            cand_scores, fin_slot, num_candidates, fill=float("-inf")
        ),
        num_finished=finished.long().sum(),
    )


def select_final_topk(
    cum_logprobs: torch.Tensor,  # [num_rows] float32
    top_logprobs: torch.Tensor,  # [num_rows, num_candidates] float32
    top_tokens: torch.Tensor,  # [num_rows, num_candidates] int64
    beam_width: int,
) -> FinalSelect:
    """Length-terminated step: the best beam_width extensions all finish. No
    stop check needed -- the caller decides this step from the step counter.

    Dummy leaves contribute -inf extensions; they sort last, so the committed
    set is only padded with them when fewer than beam_width real beams are live.
    The commit path drops those."""
    cand_scores, parent, tokens = _ranked_candidates(
        cum_logprobs, top_logprobs, top_tokens, beam_width
    )
    return FinalSelect(tokens=tokens, parent_idx=parent, cum_logprobs=cand_scores)
