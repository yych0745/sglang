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
"""CSR prefix trie for beam-search constrained decoding.

A product-SID dictionary (every entry is a fixed-depth token path) is compiled
into a flat CSR graph that lives on the target device:

    children_start / children_count   per-node slice into the flat edge arrays
    child_token / child_node          the edges, token-ascending per parent

Contract (overlap / cuda-graph ready): every query -- `root_cursor`, `advance`,
`build_mask` -- is a pure device op with fixed output shapes, no D2H read-back
and no host branch on tensor content, so it can run inside a capture and on the
launch path without stalling the host. The only host values the queries read
(`_max_branch`, `_max_token`, `_scan_offsets`) are frozen at construction from
the *static* trie topology; nothing derived from runtime data ever reaches a
Python branch or a shape.

`advance` scans inside each parent's edge slice instead of running one global
`searchsorted`: CSR is only segment-wise sorted, so the flat `child_token`
array is not globally monotonic and cannot be binary-searched as a whole. A
global searchsorted would need a per-segment offset trick that costs more than
it saves here -- the widest fan-out is a few hundred edges and the scan below
is fully vectorised over [k, max_branch].

This module owns no tokenizer and imports no sglang internal: it consumes plain
token-id paths, so it can be unit-tested standalone on CPU.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

INVALID_NODE: int = -1

# CSR indices are int32; stay one below the limit so -1 stays unambiguous.
_INT32_LIMIT: int = 2**31 - 1
_CACHE_VERSION: int = 1

# One dictionary line is exactly three SID layers. The capture groups are the
# fragments handed to `token_to_id`, not the numeric indices.
#
# The layer prefixes are free-form (<a_0><b_0><c_0> in production; any
# <name_index> triple works) so the same parser serves a synthetic dictionary
# built on an unrelated vocabulary. The strictness that matters is kept: three
# angle-bracketed fragments, nothing else on the line, so a truncated or padded
# entry still fails loudly with its line number.
_DICT_LINE_RE = re.compile(r"^(<[A-Za-z]+_\d+>)(<[A-Za-z]+_\d+>)(<[A-Za-z]+_\d+>)$")


def _host_int32(x: torch.Tensor) -> torch.Tensor:
    """Detached CPU int32 view, used for construct-time bookkeeping only."""
    return x.detach().to(device="cpu", dtype=torch.int32)


class BeamTrie:
    """Device-resident CSR trie over token-id paths.

    Node 0 is the root. A cursor is a [rows] int32 tensor of node ids; -1
    (INVALID_NODE) marks a row whose prefix already left the dictionary.
    """

    children_start: torch.Tensor  # int32 [num_nodes]
    children_count: torch.Tensor  # int32 [num_nodes]
    child_token: torch.Tensor  # int32 [num_edges]
    child_node: torch.Tensor  # int32 [num_edges]
    depth: int  # tokens in the longest path
    eos_token_id: int
    num_nodes: int
    num_edges: int
    device: torch.device

    def __init__(
        self,
        children_start: torch.Tensor,
        children_count: torch.Tensor,
        child_token: torch.Tensor,
        child_node: torch.Tensor,
        depth: int,
        eos_token_id: int,
        device: torch.device | str,
    ) -> None:
        # Everything below reads host values off the *input* arrays: this runs
        # once, at load time, never on the launch path.
        start_h = _host_int32(children_start)
        count_h = _host_int32(children_count)
        token_h = _host_int32(child_token)
        node_h = _host_int32(child_node)

        num_nodes = int(start_h.numel())
        num_edges = int(token_h.numel())
        if num_nodes == 0:
            raise ValueError("trie has no nodes")
        if num_edges == 0:
            raise ValueError("trie has no edges: the dictionary is empty")
        if num_nodes > _INT32_LIMIT:
            raise ValueError(
                f"trie has {num_nodes} nodes, past the int32 CSR limit "
                f"{_INT32_LIMIT}"
            )
        if num_edges > _INT32_LIMIT:
            raise ValueError(
                f"trie has {num_edges} edges, past the int32 CSR limit "
                f"{_INT32_LIMIT}"
            )
        if int(count_h.numel()) != num_nodes:
            raise ValueError(
                f"children_count has {count_h.numel()} entries but there are "
                f"{num_nodes} nodes"
            )
        if int(node_h.numel()) != num_edges:
            raise ValueError(
                f"child_node has {node_h.numel()} entries but there are "
                f"{num_edges} edges"
            )
        if bool(((start_h < 0) | (count_h < 0) | (start_h + count_h > num_edges)).any()):
            raise ValueError(
                "CSR offset/count pairs point outside the edge array"
            )
        if bool(((node_h < 0) | (node_h >= num_nodes)).any()):
            raise ValueError(
                f"child_node names an id outside [0, {num_nodes})"
            )
        if bool(((token_h < 0) | (token_h > _INT32_LIMIT)).any()):
            raise ValueError("child_token holds a negative or non-int32 token id")

        depth = int(depth)
        eos_token_id = int(eos_token_id)
        if depth <= 0:
            raise ValueError(f"trie depth must be positive, got {depth}")
        if eos_token_id < 0:
            raise ValueError(f"eos_token_id must be non-negative, got {eos_token_id}")

        device = torch.device(device)
        self.device = device
        self.children_start = start_h.to(device).contiguous()
        self.children_count = count_h.to(device).contiguous()
        self.child_token = token_h.to(device).contiguous()
        self.child_node = node_h.to(device).contiguous()
        self.depth = depth
        self.eos_token_id = eos_token_id
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # Static topology, frozen here so no query ever recomputes it:
        # `_max_branch` is the widest fan-out (it sizes the scan, and it must
        # not come from a runtime tensor), `_max_token` bounds vocab_size.
        self._max_branch: int = int(count_h.max())
        self._max_token: int = int(token_h.max())
        # The scan dimension must be >= 1 even for an all-leaf trie: a
        # zero-width dim makes argmax(dim=1) illegal.
        self._scan_width: int = max(self._max_branch, 1)
        # Clamp target for the "gather a safe index" trick; num_edges >= 1.
        self._clamp_max: int = num_edges - 1
        self._scan_offsets: torch.Tensor = torch.arange(
            self._scan_width, dtype=torch.int32, device=device
        ).unsqueeze(0)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_paths(
        cls,
        paths: Sequence[Sequence[int]],
        device: torch.device | str,
        eos_token_id: int,
    ) -> "BeamTrie":
        """Compile token-id paths into a trie. `paths` is what the caller gets
        after mapping dictionary fragments through a tokenizer."""
        children_start, children_count, child_token, child_node, depth = _build_csr(
            paths
        )
        return cls(
            children_start,
            children_count,
            child_token,
            child_node,
            depth=depth,
            eos_token_id=eos_token_id,
            device=device,
        )

    # -- queries (pure device ops) ------------------------------------------
    def root_cursor(self, num_rows: int) -> torch.Tensor:
        """[num_rows] int32, all root. Every group starts here."""
        return torch.zeros((num_rows,), dtype=torch.int32, device=self.device)

    def advance(
        self,
        cursor: torch.Tensor,
        parent_idx: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Walk one token per output row.

        cursor:     [rows] int32 node ids (INVALID_NODE = dead row)
        parent_idx: [k] int64/int32, row index in [0, rows)
        tokens:     [k] int64/int32 token ids

        Returns [k] int32: the child node reached, or INVALID_NODE when the
        parent is dead / a leaf / has no edge for that token.
        """
        k = parent_idx.shape[0]
        rows = cursor.shape[0]
        # Static-shape branches, constant under capture.
        if k == 0:
            return torch.zeros((0,), dtype=torch.int32, device=self.device)
        if rows == 0:
            # No live row to inherit from, so every parent index is invalid.
            return torch.full(
                (k,), INVALID_NODE, dtype=torch.int32, device=self.device
            )

        src = parent_idx.to(torch.int32)
        # Defensive: dummy rows seat themselves at 0, but a caller may hand us
        # an out-of-range index; such rows resolve to INVALID_NODE instead of
        # reading whatever the clamp happens to land on.
        in_range = (src >= 0) & (src < rows)
        src_safe = src.clamp(min=0, max=max(rows - 1, 0))

        node = cursor[src_safe]  # [k] int32
        count = self.children_count[node]  # [k]
        start = self.children_start[node]  # [k]

        # Scan each parent's [start, start+count) slice. Out-of-slice lanes are
        # clamped to a legal index and then masked off by `valid`.
        offsets = self._scan_offsets  # [1, B]
        idx = (start[:, None] + offsets).clamp(max=self._clamp_max)  # [k, B]
        valid = offsets < count[:, None]  # [k, B]
        hit = valid & (self.child_token[idx] == tokens[:, None])  # [k, B]

        found = hit.any(dim=1)
        # argmax returns 0 when nothing matched; `found` discards that.
        first = hit.to(torch.int32).argmax(dim=1)  # [k]
        child = self.child_node[idx].gather(1, first[:, None]).squeeze(1)

        alive = found & (node != INVALID_NODE) & in_range
        return torch.where(alive, child, torch.full_like(child, INVALID_NODE))

    def build_mask(self, cursor: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """[rows, vocab_size] bool, True = forbidden.

        A live node releases exactly its own child tokens. A dead row
        (INVALID_NODE) or a leaf releases eos_token_id only, which is what
        lets a beam terminate once its prefix is a complete dictionary entry.
        """
        vocab_size = int(vocab_size)
        if vocab_size <= self.eos_token_id:
            raise ValueError(
                f"vocab_size {vocab_size} must exceed eos_token_id "
                f"{self.eos_token_id}"
            )
        if vocab_size <= self._max_token:
            raise ValueError(
                f"vocab_size {vocab_size} must exceed the largest trie token "
                f"id {self._max_token}"
            )

        rows = cursor.shape[0]
        forbidden = torch.ones(
            (rows, vocab_size), dtype=torch.bool, device=self.device
        )
        # Dead rows would otherwise alias the root (index -1 or any negative);
        # they are masked out below, so the clamp only has to stay in bounds.
        safe = cursor.clamp(min=0)
        count = self.children_count[safe]  # [rows]
        start = self.children_start[safe]  # [rows]

        offsets = self._scan_offsets  # [1, B]
        idx = (start[:, None] + offsets).clamp(max=self._clamp_max)  # [rows, B]
        valid = (offsets < count[:, None]) & (cursor != INVALID_NODE)[:, None]

        tok = self.child_token[idx]  # [rows, B]
        row_ids = torch.arange(
            rows, dtype=torch.int32, device=self.device
        )[:, None].expand_as(tok)
        forbidden[row_ids[valid], tok[valid]] = False

        # No legal child at all: leaf or dead row -> eos only.
        terminal = ~valid.any(dim=1)  # [rows]
        forbidden[terminal] = True
        forbidden[terminal, self.eos_token_id] = False
        return forbidden


def _build_csr(
    paths: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Insert token-id paths into a trie and flatten it to CSR.

    Node ids grow monotonically with insertion order, so a child id is always
    greater than its parent's. Edges of a node are emitted token-ascending.
    """
    if not paths:
        raise ValueError("empty dictionary: no paths to compile")

    edges: Dict[Tuple[int, int], int] = {}
    num_nodes = 1  # root
    depth = 0

    for path in paths:
        if len(path) == 0:
            raise ValueError("dictionary contains an empty path")
        node = 0
        for raw_tok in path:
            token = int(raw_tok)
            if token < 0 or token > _INT32_LIMIT:
                raise ValueError(
                    f"token id {token} is not a non-negative int32"
                )
            key = (node, token)
            child = edges.get(key)
            if child is None:
                if num_nodes >= _INT32_LIMIT:
                    raise ValueError(
                        f"dictionary exceeds the int32 node limit {_INT32_LIMIT}"
                    )
                child = num_nodes
                num_nodes += 1
                edges[key] = child
            node = child
        if len(path) > depth:
            depth = len(path)

    buckets: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for (parent, token), child in edges.items():
        buckets[parent].append((token, child))
    for bucket in buckets:
        bucket.sort()  # token-ascending per parent

    starts: List[int] = []
    counts: List[int] = []
    flat_token: List[int] = []
    flat_node: List[int] = []
    for bucket in buckets:
        starts.append(len(flat_token))
        counts.append(len(bucket))
        for token, child in bucket:
            flat_token.append(token)
            flat_node.append(child)

    return (
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(counts, dtype=torch.int32),
        torch.tensor(flat_token, dtype=torch.int32),
        torch.tensor(flat_node, dtype=torch.int32),
        depth,
    )


def _parse_dictionary(
    dict_path: str, token_to_id: Callable[[str], Optional[int]]
) -> List[List[int]]:
    """Stream a <a_N><b_N><c_N> dictionary into token-id paths.

    Line-numbered ValueErrors: a 700k-line file has to tell the operator where
    it is broken.
    """
    paths: List[List[int]] = []
    # Streamed on purpose: the real dictionary is ~750k lines and readlines()
    # would hold all of it at once.
    with open(dict_path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            match = _DICT_LINE_RE.match(line)
            if match is None:
                raise ValueError(
                    f"invalid dictionary line {lineno} in {dict_path}: {line!r}"
                )
            path: List[int] = []
            for fragment in match.groups():
                token = token_to_id(fragment)
                if token is None:
                    raise ValueError(
                        f"unknown token {fragment!r} at line {lineno} in "
                        f"{dict_path}"
                    )
                path.append(int(token))
            paths.append(path)

    if not paths:
        raise ValueError(f"empty dictionary: {dict_path}")
    return paths


def _cache_key(
    dict_path: str,
    eos_token_id: int,
    tokenizer_fingerprint: Optional[str] = None,
) -> Tuple[int, int, int, str]:
    # The fingerprint is part of the key because the cached CSR holds *token
    # ids*: the same dictionary compiled against a different (or retrained)
    # tokenizer yields the same topology with different ids, and serving that
    # stale entry silently decodes the wrong products. mtime+size only pin the
    # dictionary file, so both sides of the mapping have to be in the key.
    stat = os.stat(dict_path)
    return (
        stat.st_mtime_ns,
        stat.st_size,
        int(eos_token_id),
        tokenizer_fingerprint or "",
    )


def _save_cache(cache_path: str, payload: Dict[str, object]) -> None:
    """Write via tmp + os.replace so a reader never sees a partial file."""
    directory = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
        os.replace(tmp_path, cache_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_or_compile(
    dict_path: str,
    device: torch.device | str,
    eos_token_id: int,
    token_to_id: Callable[[str], Optional[int]],
    *,
    cache_path: Optional[str] = None,
    tokenizer_fingerprint: Optional[str] = None,
) -> BeamTrie:
    """Compile `dict_path` into a BeamTrie, reusing the on-disk cache.

    `token_to_id` maps one dictionary fragment (e.g. "<a_7>") to its token id;
    None means the fragment is not a single token, which is treated as a
    corrupted dictionary.

    `tokenizer_fingerprint` identifies the tokenizer the ids came from and is
    part of the cache key: a stale hit would hand back another tokenizer's ids.
    The caller computes it (e.g. a hash of the tokenizer file's path, mtime and
    size) -- this module owns no tokenizer and will not guess where one lives.

    The cache holds host-side int32 CSR arrays plus depth/eos and the key
    (mtime_ns, size, eos_token_id, tokenizer_fingerprint) -- never device
    tensors, so the same cache file serves any device. A key mismatch or an
    unreadable file falls back to a recompile; a corrupt cache is never fatal.
    """
    if cache_path is None:
        cache_path = dict_path + ".trie.pt"
    key = _cache_key(dict_path, eos_token_id, tokenizer_fingerprint)

    try:
        payload = torch.load(cache_path)
        if (
            int(payload.get("version", -1)) == _CACHE_VERSION
            and tuple(payload["key"]) == key
        ):
            return BeamTrie(
                payload["children_start"],
                payload["children_count"],
                payload["child_token"],
                payload["child_node"],
                depth=int(payload["depth"]),
                eos_token_id=eos_token_id,
                device=device,
            )
        logger.info("trie cache %s is stale, recompiling", cache_path)
    except Exception as exc:  # unreadable / stale / hand-edited cache
        logger.info("trie cache %s unusable (%s), recompiling", cache_path, exc)

    paths = _parse_dictionary(dict_path, token_to_id)
    trie = BeamTrie.from_paths(paths, device=device, eos_token_id=eos_token_id)

    payload = {
        "version": _CACHE_VERSION,
        "key": key,
        # Stored for troubleshooting: lets an operator tell which tokenizer
        # produced the ids in this file without re-deriving the key.
        "tokenizer_fingerprint": tokenizer_fingerprint or "",
        "depth": trie.depth,
        "eos_token_id": trie.eos_token_id,
        # Host tensors only: the device is a runtime choice, not a cache one.
        "children_start": trie.children_start.cpu(),
        "children_count": trie.children_count.cpu(),
        "child_token": trie.child_token.cpu(),
        "child_node": trie.child_node.cpu(),
    }
    try:
        _save_cache(cache_path, payload)
    except Exception as exc:  # cache is best-effort
        logger.warning("could not write trie cache %s: %s", cache_path, exc)
    return trie
