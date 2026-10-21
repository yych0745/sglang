"""CSR prefix trie for beam search: structure, advance, build_mask, cache.

Every query the coordinator runs on the launch path (`root_cursor`, `advance`,
`build_mask`) has to stay a pure device op: no `.item()`, no host branch on
tensor content. These tests drive that machinery on CPU, where "device" is
simply torch.device("cpu").

`TestLseBeforeMaskOrder` freezes the call order "log-softmax over the full
vocab, then mask" against the *production* function (`coordinator.
_rows_topk_logprobs`): masking first would renormalise each row over a
different legal set, making cumulative logprobs incomparable across beams
sitting at different depths.

`TestBeamGroupTrieCursor` covers the group-side cursor: one node per frontier
row, advanced by `advance_frontier` alongside `frontier_cum_logprobs`.
"""

import os
import tempfile
import unittest

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import _rows_topk_logprobs
from sglang.srt.beam_search.joint_select import SelectResult, joint_select
from sglang.srt.beam_search.trie import (
    INVALID_NODE,
    BeamTrie,
    _cache_key,
    load_or_compile,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

EOS_TOKEN_ID = 9
VOCAB_SIZE = 16

# root -> {1, 7}; 1 -> {2, 5}; 2 -> {3, 4}; 7 -> {8}; leaves: 3, 4, 5, 8
PATHS = [[1, 2, 3], [1, 2, 4], [1, 5], [7, 8]]

# --- synthetic dictionary fragments -----------------------------------------
# The real dictionary is <a_N><b_N><c_N>; tests use a hand-made fragment -> id
# table so no tokenizer is ever loaded.
FRAG_TO_ID = {
    "<a_0>": 1,
    "<b_0>": 2,
    "<c_0>": 3,
    "<c_1>": 4,
    "<b_1>": 5,
    "<a_1>": 7,
    "<b_2>": 8,
}

DICT_LINES = [
    "<a_0><b_0><c_0>",  # [1, 2, 3]
    "<a_0><b_0><c_1>",  # [1, 2, 4]
    "<a_1><b_2><c_0>",  # [7, 8, 3]
]


def frag_lookup(fragment: str):
    return FRAG_TO_ID.get(fragment)


def child_of(trie: BeamTrie, node: int, token: int) -> int:
    """Reference lookup straight off the CSR arrays, independent of `advance`."""
    start = int(trie.children_start[node])
    count = int(trie.children_count[node])
    tokens = trie.child_token[start : start + count].tolist()
    if token not in tokens:
        return INVALID_NODE
    return int(trie.child_node[start + tokens.index(token)])


def allowed_of(mask_row: torch.Tensor) -> list:
    return [i for i, forbidden in enumerate(mask_row.tolist()) if not forbidden]


class TestCsrStructure(CustomTestCase):
    """The flat CSR layout the device kernels index into."""

    def setUp(self):
        self.t = BeamTrie.from_paths(PATHS, device="cpu", eos_token_id=EOS_TOKEN_ID)

    def test_csr_structure(self):
        t = self.t
        start = t.children_start.tolist()
        count = t.children_count.tolist()
        token = t.child_token.tolist()
        node = t.child_node.tolist()

        self.assertEqual(t.depth, 3)
        self.assertEqual(t.num_nodes, 8)
        self.assertEqual(t.num_edges, 7)

        for name in ("children_start", "children_count", "child_token", "child_node"):
            self.assertEqual(getattr(t, name).dtype, torch.int32)

        self.assertEqual(start, [0, 2, 4, 6, 6, 6, 6, 7])
        self.assertEqual(count, [2, 2, 2, 0, 0, 0, 1, 0])
        self.assertEqual(token, [1, 7, 2, 5, 3, 4, 8])
        self.assertEqual(node, [1, 6, 2, 5, 3, 4, 7])

        # Offsets tile the edge array exactly, in node order.
        self.assertEqual(sum(count), len(token))
        running, expected_start = 0, []
        for c in count:
            expected_start.append(running)
            running += c
        self.assertEqual(start, expected_start)

        # Per-parent edges are token-ascending, and a child is always created
        # after its parent, so child ids strictly increase with depth.
        for n in range(t.num_nodes):
            lo = start[n]
            hi = lo + count[n]
            seg = token[lo:hi]
            self.assertEqual(seg, sorted(seg))
            kids = node[lo:hi]
            self.assertEqual(len(set(kids)), len(kids))
            self.assertTrue(all(k > n for k in kids))

        # Root children are the first token of every path.
        self.assertEqual(set(token[start[0] : start[0] + count[0]]), {1, 7})

        # Every input path is walkable and ends on a leaf.
        for path in PATHS:
            cur = 0
            for tok in path:
                cur = child_of(t, cur, tok)
                self.assertNotEqual(cur, INVALID_NODE)
            self.assertEqual(int(t.children_count[cur]), 0)


class TestAdvance(CustomTestCase):
    """`advance` walks one token per row, fully vectorised, no host sync."""

    def setUp(self):
        self.t = BeamTrie.from_paths(PATHS, device="cpu", eos_token_id=EOS_TOKEN_ID)
        self.n_a = child_of(self.t, 0, 1)
        self.n_b = child_of(self.t, self.n_a, 2)
        self.n_c = child_of(self.t, 0, 7)
        self.leaf_3 = child_of(self.t, self.n_b, 3)
        self.leaf_4 = child_of(self.t, self.n_b, 4)
        self.leaf_5 = child_of(self.t, self.n_a, 5)
        self.leaf_8 = child_of(self.t, self.n_c, 8)

    def _cursor(self, *nodes: int) -> torch.Tensor:
        return torch.tensor(list(nodes), dtype=torch.int32, device=self.t.device)

    def _advance(self, cursor, parents, tokens) -> list:
        out = self.t.advance(
            cursor,
            torch.tensor(parents, dtype=torch.int64),
            torch.tensor(tokens, dtype=torch.int64),
        )
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(out.shape, (len(parents),))
        return out.tolist()

    def test_advance_hit(self):
        root = self.t.root_cursor(1)
        self.assertEqual(root.tolist(), [0])
        self.assertEqual(root.dtype, torch.int32)

        a = self.t.advance(root, torch.tensor([0]), torch.tensor([1]))
        self.assertEqual(a.tolist(), [self.n_a])

        b = self.t.advance(a, torch.tensor([0]), torch.tensor([2]))
        self.assertEqual(b.tolist(), [self.n_b])

        leaf = self.t.advance(b, torch.tensor([0]), torch.tensor([3]))
        self.assertEqual(leaf.tolist(), [self.leaf_3])
        # depth 3 is the longest path, so this is a leaf.
        self.assertEqual(int(self.t.children_count[self.leaf_3]), 0)

    def test_advance_miss(self):
        # 2 is not a root child (only 1 and 7 are).
        self.assertEqual(
            self._advance(self._cursor(0), [0], [2]), [INVALID_NODE]
        )
        # 5 hangs under node 1, not under node 2.
        self.assertEqual(
            self._advance(self._cursor(self.n_b), [0], [5]), [INVALID_NODE]
        )
        # A token outside the dictionary entirely.
        self.assertEqual(
            self._advance(self._cursor(self.n_a), [0], [11]), [INVALID_NODE]
        )

    def test_advance_from_leaf(self):
        for leaf in (self.leaf_3, self.leaf_4, self.leaf_5, self.leaf_8):
            self.assertEqual(int(self.t.children_count[leaf]), 0)
            self.assertEqual(
                self._advance(self._cursor(leaf), [0], [1]), [INVALID_NODE]
            )
            self.assertEqual(
                self._advance(self._cursor(leaf), [0], [EOS_TOKEN_ID]),
                [INVALID_NODE],
            )

    def test_advance_from_invalid(self):
        # A dead row must stay dead, and must not index out of bounds.
        cursor = self._cursor(INVALID_NODE, INVALID_NODE, INVALID_NODE)
        self.assertEqual(
            self._advance(cursor, [0, 1, 2], [1, 2, 7]),
            [INVALID_NODE, INVALID_NODE, INVALID_NODE],
        )
        # Mixed: only the row whose parent is INVALID dies.
        cursor = self._cursor(INVALID_NODE, 0, self.n_a)
        self.assertEqual(
            self._advance(cursor, [0, 1, 2], [1, 7, 2]),
            [INVALID_NODE, self.n_c, self.n_b],
        )

    def test_advance_rows_independent(self):
        cursor = self._cursor(0, self.n_a, self.n_b)
        self.assertEqual(
            self._advance(cursor, [0, 1, 2], [1, 2, 3]),
            [self.n_a, self.n_b, self.leaf_3],
        )
        # Same rows, different tokens: row 0 takes the 7 branch, row 2 takes 4.
        self.assertEqual(
            self._advance(cursor, [0, 1, 2], [7, 5, 4]),
            [self.n_c, self.leaf_5, self.leaf_4],
        )

    def test_advance_dummy_duplicate_parent(self):
        # Dummy padding: every row names parent 0. Three identical results,
        # none of them influencing the others.
        cursor = self._cursor(0, 0, 0)
        self.assertEqual(
            self._advance(cursor, [0, 0, 0], [1, 1, 1]),
            [self.n_a, self.n_a, self.n_a],
        )

        # Two rows share node A as parent but walk different tokens: row 1
        # hits 5, row 2 misses with 3. Their outcomes must not bleed together.
        cursor = self._cursor(0, self.n_a, self.n_b)
        self.assertEqual(
            self._advance(cursor, [0, 1, 1], [1, 5, 3]),
            [self.n_a, self.leaf_5, INVALID_NODE],
        )

        # int32 parent_idx must work just as well as int64.
        out = self.t.advance(
            cursor,
            torch.tensor([0, 1, 1], dtype=torch.int32),
            torch.tensor([1, 5, 3], dtype=torch.int32),
        )
        self.assertEqual(out.tolist(), [self.n_a, self.leaf_5, INVALID_NODE])


class TestBuildMask(CustomTestCase):
    """`build_mask` returns True = forbidden, [rows, vocab_size]."""

    def setUp(self):
        self.t = BeamTrie.from_paths(PATHS, device="cpu", eos_token_id=EOS_TOKEN_ID)
        self.n_a = child_of(self.t, 0, 1)
        self.n_b = child_of(self.t, self.n_a, 2)
        self.leaf_3 = child_of(self.t, self.n_b, 3)

    def test_build_mask_root(self):
        mask = self.t.build_mask(self.t.root_cursor(1), VOCAB_SIZE)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(list(mask.shape), [1, VOCAB_SIZE])
        self.assertEqual(allowed_of(mask[0]), [1, 7])

    def test_build_mask_leaf(self):
        cursor = torch.tensor([self.leaf_3], dtype=torch.int32)
        mask = self.t.build_mask(cursor, VOCAB_SIZE)
        self.assertEqual(allowed_of(mask[0]), [EOS_TOKEN_ID])

    def test_build_mask_invalid(self):
        cursor = torch.full((2,), INVALID_NODE, dtype=torch.int32)
        mask = self.t.build_mask(cursor, VOCAB_SIZE)
        self.assertEqual(allowed_of(mask[0]), [EOS_TOKEN_ID])
        self.assertEqual(allowed_of(mask[1]), [EOS_TOKEN_ID])

    def test_build_mask_multi_row(self):
        cursor = torch.tensor(
            [0, self.n_a, self.n_b, self.leaf_3, INVALID_NODE], dtype=torch.int32
        )
        mask = self.t.build_mask(cursor, VOCAB_SIZE)
        self.assertEqual(list(mask.shape), [5, VOCAB_SIZE])
        self.assertEqual(allowed_of(mask[0]), [1, 7])
        self.assertEqual(allowed_of(mask[1]), [2, 5])
        self.assertEqual(allowed_of(mask[2]), [3, 4])
        self.assertEqual(allowed_of(mask[3]), [EOS_TOKEN_ID])
        self.assertEqual(allowed_of(mask[4]), [EOS_TOKEN_ID])


class TestConstrainedGeneration(CustomTestCase):
    """End-to-end on a synthetic trie: mask -> topk -> joint_select must only
    ever emit paths that exist in the dictionary.

    Logits are hand-built so that the greedy (unconstrained) choice is always a
    token OUTSIDE the trie, which makes "stayed inside" a real assertion rather
    than a coincidence of the fixture.
    """

    K = 2

    def setUp(self):
        self.t = BeamTrie.from_paths(PATHS, device="cpu", eos_token_id=EOS_TOKEN_ID)
        # Every forbidden token outranks every legal one: without the mask this
        # would walk straight out of the dictionary.
        self.logits = torch.full((1, VOCAB_SIZE), 100.0)
        for tok in (1, 7, 2, 5, 3, 4, 8, EOS_TOKEN_ID):
            self.logits[0, tok] = 0.0

    def _step(self, cursor, frontier, k):
        # `frontier` carries exactly `cursor.shape[0]` rows, which is what
        # joint_select requires (one cumulative logprob per frontier row).
        rows = cursor.shape[0]
        piece = self.logits.expand(rows, -1).clone()
        mask = self.t.build_mask(cursor, VOCAB_SIZE)
        top_logprobs, top_tokens = _rows_topk_logprobs([piece], 2 * k, mask=mask)
        return joint_select(
            frontier, top_logprobs, top_tokens, torch.tensor([EOS_TOKEN_ID]), k
        )

    def test_first_step_only_legal_root_children(self):
        cursor = self.t.root_cursor(1)
        sel = self._step(cursor, torch.zeros(1), self.K)
        survivors = sel.next_tokens[: int(sel.num_survivors)].tolist()
        # Only root children (1 and 7) may appear; the 100.0 forbidden tokens
        # are the ones a mask-free path would have picked.
        self.assertEqual(sorted(survivors), [1, 7])

    def test_walk_to_leaf_then_only_eos(self):
        # Follow the single best beam down a real path. K=1 keeps cursor and
        # frontier on the same one row, so each step is a plain greedy walk.
        # The dictionary is depth 3 here, so at most 3 token steps are legal;
        # the step that meets a leaf must return EOS and stop adding tokens.
        cursor = self.t.root_cursor(1)
        frontier = torch.zeros(1)
        path = []
        for _ in range(self.t.depth):
            sel = self._step(cursor, frontier, 1)
            survived = int(sel.num_survivors)
            if survived == 0:
                # Every candidate was EOS: the beam terminated here, so the
                # filler in next_tokens must not be read as a token.
                break
            path.append(int(sel.next_tokens[0]))
            cursor = self.t.advance(cursor, sel.parent_idx, sel.next_tokens)
            frontier = sel.new_cum_logprobs
            self.assertNotEqual(
                int(cursor[0]), INVALID_NODE, f"left the dictionary after {path}"
            )
        self.assertTrue(
            any(path == list(p) for p in PATHS), f"{path} is not a dictionary path"
        )
        # At a leaf the mask releases EOS only, so the next step terminates.
        self.assertEqual(int(self.t.children_count[int(cursor[0])]), 0)
        mask = self.t.build_mask(cursor, VOCAB_SIZE)
        self.assertEqual(allowed_of(mask[0]), [EOS_TOKEN_ID])

    def test_every_beam_stays_inside_the_dictionary(self):
        # Track the real beams for depth steps; every live prefix must remain a
        # node of the trie. Dummy seats (cum_logprob -inf) are excluded: they
        # are padding, carry no sequence, and their cursor is inert by design.
        cursor = self.t.root_cursor(1)
        frontier = torch.zeros(1)
        # Steps 1..depth-1 emit real tokens; the depth-th step reaches leaves,
        # where the only legal token is EOS and every beam terminates.
        for step in range(1, self.t.depth):
            sel = self._step(cursor, frontier, self.K)
            real = int(sel.num_survivors)
            self.assertGreaterEqual(
                real, 1, f"constraint left no live beam at step {step}"
            )
            cursor = self.t.advance(
                cursor, sel.parent_idx[:real], sel.next_tokens[:real]
            )
            frontier = sel.new_cum_logprobs[:real]
            for node in cursor.tolist():
                self.assertNotEqual(
                    node, INVALID_NODE, f"a live beam left the dictionary at step {step}"
                )


class TestLseBeforeMaskOrder(CustomTestCase):
    """Contract: normalise with the *unmasked* logsumexp, then mask.

    Driven through the production function, not through a hand-rolled copy of
    the arithmetic: a test that only re-derives the identity on hardcoded
    logits stays green no matter what order the caller implements.
    """

    def test_lse_before_mask_semantics(self):
        # The arithmetic identity itself, spelled out: it is the reference the
        # production assertions below are checked against.
        logits = torch.tensor([[10.0, 0.0, 9.0, -1.0, 8.0, 7.0, 6.0, 5.0]])
        legal = (1, 3)
        mask = torch.ones(1, 8, dtype=torch.bool)
        for i in legal:
            mask[0, i] = False

        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        out = (logits - lse).masked_fill(mask, float("-inf"))

        # -inf lands exactly on the forbidden positions.
        self.assertTrue(torch.equal(out == float("-inf"), mask))

        # Legal positions carry logit - lse(full vocab).
        for i in legal:
            self.assertAlmostEqual(
                float(out[0, i]), float(logits[0, i] - lse[0, 0]), places=5
            )

        # ... and explicitly NOT the masked renormalisation, which is what a
        # mask-before-lse implementation would produce.
        lse_masked = torch.logsumexp(
            logits.masked_fill(mask, float("-inf")), dim=-1, keepdim=True
        )
        self.assertNotAlmostEqual(float(lse[0, 0]), float(lse_masked[0, 0]), places=2)
        for i in legal:
            self.assertNotAlmostEqual(
                float(out[0, i]), float(logits[0, i] - lse_masked[0, 0]), places=2
            )

    def test_production_orders_lse_before_mask(self):
        # 1 row, vocab 8. Legal tokens are 1 and 3, both low-logit; every
        # high-logit position is forbidden. A mask-before-lse implementation
        # would renormalize over {1,3} and hand out ~0 for both.
        logits = torch.tensor([[10.0, 0.0, 9.0, -1.0, 8.0, 7.0, 6.0, 5.0]])
        mask = torch.ones(1, 8, dtype=torch.bool)
        mask[0, 1] = False
        mask[0, 3] = False

        v, t = _rows_topk_logprobs([logits], 2, mask=mask)

        # Only legal tokens survive the topk.
        self.assertEqual(sorted(t.tolist()[0]), [1, 3])
        # Each score is logit - lse(FULL vocab), not the masked renormalization.
        lse_full = torch.logsumexp(logits, dim=-1, keepdim=True)
        lse_masked = torch.logsumexp(
            logits.masked_fill(mask, float("-inf")), dim=-1, keepdim=True
        )
        self.assertNotAlmostEqual(float(lse_full), float(lse_masked), places=2)
        got = {int(tok): float(val) for tok, val in zip(t[0], v[0])}
        for tok in (1, 3):
            self.assertAlmostEqual(
                got[tok], float(logits[0, tok] - lse_full[0, 0]), places=5
            )
            self.assertNotAlmostEqual(
                got[tok], float(logits[0, tok] - lse_masked[0, 0]), places=2
            )

    def test_mask_row_offset_across_pieces(self):
        # Two pieces of different row counts, an empty piece in between: the
        # mask must line up with the CONCATENATED row order, so piece 2 sees
        # mask rows [2,4), not [1,3).
        #
        # Logits are deliberately tie-free: the point is the row offset, and a
        # tie would make the expected index a topk implementation detail.
        p1 = torch.tensor([[0.0, 5.0, 2.0, 1.0], [0.0, 5.0, 2.0, 1.0]])
        empty = torch.zeros(0, 4)
        p2 = torch.tensor([[5.0, 0.0, 2.0, 1.0], [0.0, 5.0, 2.0, 1.0]])
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[0, 1] = True  # piece1 row0: forbid token 1 -> best is token 2
        mask[2, 0] = True  # piece2 row0: forbid token 0 -> best is token 2
        # A wrong offset would let piece2 row0 keep token 0 (logit 5): [2,1,0,1].
        v, t = _rows_topk_logprobs([p1, empty, p2], 1, mask=mask)
        self.assertEqual(t[:, 0].tolist(), [2, 1, 2, 1])

        # And every score is still logit - lse(full vocab) of its own row: the
        # empty piece contributes no row to either the mask or the output.
        rows = torch.cat([p1, p2])
        lse_full = torch.logsumexp(rows, dim=-1, keepdim=True)
        for r in range(rows.shape[0]):
            tok = int(t[r, 0])
            self.assertAlmostEqual(
                float(v[r, 0]), float(rows[r, tok] - lse_full[r, 0]), places=5
            )

    def test_no_mask_matches_legacy(self):
        # mask=None must keep the exact previous behaviour (top-2k over raw logits).
        logits = torch.randn(3, 20)
        v, t = _rows_topk_logprobs([logits], 4)
        ref_v, ref_t = torch.topk(torch.log_softmax(logits, -1), 4, dim=-1)
        self.assertTrue(torch.allclose(v, ref_v, atol=1e-5))
        self.assertTrue(torch.equal(t, ref_t))


class TestCompileAndCache(CustomTestCase):
    """Dictionary parsing and the on-disk compiled cache."""

    def setUp(self):
        self._tmp = []

    def tearDown(self):
        for path in self._tmp:
            try:
                os.remove(path)
            except OSError:
                pass

    def _write_dict(self, lines) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".dict", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            path = f.name
        self._tmp.append(path)
        return path

    def test_roundtrip_cache(self):
        dict_path = self._write_dict(DICT_LINES)
        cache_path = dict_path + ".trie.pt"
        self._tmp.append(cache_path)

        first = load_or_compile(
            dict_path, "cpu", EOS_TOKEN_ID, frag_lookup, cache_path=cache_path
        )
        self.assertTrue(os.path.exists(cache_path))

        second = load_or_compile(
            dict_path, "cpu", EOS_TOKEN_ID, frag_lookup, cache_path=cache_path
        )
        for name in (
            "children_start",
            "children_count",
            "child_token",
            "child_node",
        ):
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)))
        self.assertEqual(first.depth, second.depth)
        self.assertEqual(first.num_nodes, second.num_nodes)
        self.assertEqual(first.num_edges, second.num_edges)

        # A different eos changes the cache key, so the entry is rebuilt and
        # rewritten rather than served stale.
        third = load_or_compile(
            dict_path, "cpu", EOS_TOKEN_ID + 1, frag_lookup, cache_path=cache_path
        )
        self.assertEqual(third.eos_token_id, EOS_TOKEN_ID + 1)
        self.assertTrue(torch.equal(third.children_start, first.children_start))
        self.assertTrue(torch.equal(third.child_token, first.child_token))
        payload = torch.load(cache_path)
        self.assertEqual(payload["eos_token_id"], EOS_TOKEN_ID + 1)

        # Back to the original eos: rebuilt again, identical CSR.
        fourth = load_or_compile(
            dict_path, "cpu", EOS_TOKEN_ID, frag_lookup, cache_path=cache_path
        )
        self.assertEqual(fourth.eos_token_id, EOS_TOKEN_ID)
        self.assertTrue(torch.equal(fourth.child_node, first.child_node))

    def test_cache_key_includes_tokenizer_fingerprint(self):
        # The cache stores token *ids*, so the tokenizer is half of the key:
        # a stale hit would serve another tokenizer's ids against the same
        # dictionary file (silently wrong output, worse than a crash).
        dict_path = self._write_dict(["<a_0><b_0><c_0>"])
        cache_path = dict_path + ".trie.pt"
        self._tmp.append(cache_path)

        first = load_or_compile(
            dict_path,
            "cpu",
            EOS_TOKEN_ID,
            frag_lookup,
            cache_path=cache_path,
            tokenizer_fingerprint="tokA",
        )
        self.assertEqual(int(first.child_token[0]), FRAG_TO_ID["<a_0>"])
        self.assertEqual(torch.load(cache_path)["tokenizer_fingerprint"], "tokA")

        # A different tokenizer maps <a_0> elsewhere. `other` records whether
        # it is consulted at all, which is what separates "hit" from
        # "recompiled".
        consulted = []
        remapped = dict(FRAG_TO_ID)
        remapped["<a_0>"] = 11

        def other(fragment):
            consulted.append(fragment)
            return remapped.get(fragment)

        # Same fingerprint: cache hit, so `other` is never even called and the
        # CSR stays the one the first call built.
        hit = load_or_compile(
            dict_path,
            "cpu",
            EOS_TOKEN_ID,
            other,
            cache_path=cache_path,
            tokenizer_fingerprint="tokA",
        )
        self.assertEqual(consulted, [])
        self.assertTrue(torch.equal(hit.child_token, first.child_token))
        self.assertEqual(int(hit.child_token[0]), FRAG_TO_ID["<a_0>"])

        # Different fingerprint: no stale hit, so the dictionary is re-parsed
        # through the new tokenizer and the ids follow it.
        second = load_or_compile(
            dict_path,
            "cpu",
            EOS_TOKEN_ID,
            other,
            cache_path=cache_path,
            tokenizer_fingerprint="tokB",
        )
        self.assertTrue(consulted)
        self.assertEqual(int(second.child_token[0]), 11)
        self.assertNotEqual(int(second.child_token[0]), int(first.child_token[0]))
        self.assertEqual(torch.load(cache_path)["tokenizer_fingerprint"], "tokB")

    def test_fingerprint_none_is_the_empty_fingerprint(self):
        # trie.py never guesses a tokenizer, so an unset fingerprint must be a
        # stable placeholder -- not a per-call object that always misses.
        dict_path = self._write_dict(DICT_LINES)
        self.assertEqual(
            _cache_key(dict_path, EOS_TOKEN_ID, None),
            _cache_key(dict_path, EOS_TOKEN_ID, ""),
        )
        self.assertEqual(_cache_key(dict_path, EOS_TOKEN_ID)[-1], "")

    def test_parse_rejects_bad_line(self):
        dict_path = self._write_dict([DICT_LINES[0], "not-a-sid", DICT_LINES[1]])
        with self.assertRaises(ValueError) as ctx:
            load_or_compile(dict_path, "cpu", EOS_TOKEN_ID, frag_lookup)
        self.assertIn("line 2", str(ctx.exception))

    def test_parse_rejects_unknown_token(self):
        dict_path = self._write_dict(["<a_0><b_0><c_0>", "<a_9><b_9><c_9>"])
        with self.assertRaises(ValueError) as ctx:
            load_or_compile(dict_path, "cpu", EOS_TOKEN_ID, frag_lookup)
        self.assertIn("line 2", str(ctx.exception))


def make_sel(
    next_tokens,
    parent_idx,
    new_cum_logprobs,
    num_survivors=None,
    num_candidates=4,
) -> SelectResult:
    """Hand-built SelectResult for the launch half.

    advance_frontier reads only next_tokens / parent_idx / new_cum_logprobs,
    but every field of the struct is fixed-shape and must be present.
    """
    k = len(next_tokens)
    if num_survivors is None:
        num_survivors = k
    return SelectResult(
        next_tokens=torch.tensor(next_tokens, dtype=torch.int64),
        parent_idx=torch.tensor(parent_idx, dtype=torch.int64),
        new_cum_logprobs=torch.tensor(new_cum_logprobs, dtype=torch.float32),
        num_survivors=torch.tensor(num_survivors, dtype=torch.int64),
        fin_tokens=torch.zeros(num_candidates, dtype=torch.int64),
        fin_parent_idx=torch.zeros(num_candidates, dtype=torch.int64),
        fin_cum_logprobs=torch.full((num_candidates,), float("-inf")),
        num_finished=torch.tensor(0, dtype=torch.int64),
    )


class TestBeamGroupTrieCursor(CustomTestCase):
    """The group-side cursor: one trie node per frontier row, advanced by the
    same launch-half call that evolves frontier_cum_logprobs.

    Preflight step is 1 row (the prompt pseudo-row) -> k rows; every later step
    is k rows -> k rows, indexed by parent_idx.
    """

    def setUp(self):
        self.t = BeamTrie.from_paths(PATHS, device="cpu", eos_token_id=EOS_TOKEN_ID)
        self.n_a = child_of(self.t, 0, 1)
        self.n_b = child_of(self.t, self.n_a, 2)
        self.n_c = child_of(self.t, 0, 7)
        self.leaf_3 = child_of(self.t, self.n_b, 3)
        self.leaf_4 = child_of(self.t, self.n_b, 4)
        self.leaf_5 = child_of(self.t, self.n_a, 5)
        self.leaf_8 = child_of(self.t, self.n_c, 8)

    def _group(self, beam_width=2, **kwargs) -> BeamGroup:
        defaults = dict(
            beam_width=beam_width,
            stop_token_ids=[EOS_TOKEN_ID],
            max_new_tokens=8,
            device="cpu",
            trie=self.t,
        )
        defaults.update(kwargs)
        return BeamGroup(**defaults)

    def test_group_cursor_starts_at_root(self):
        group = self._group()
        self.assertIs(group.trie, self.t)
        self.assertEqual(group.trie_cursor.dtype, torch.int32)
        # One pseudo-row, mirroring frontier_cum_logprobs' torch.zeros(1).
        self.assertEqual(list(group.trie_cursor.shape), [1])
        self.assertEqual(group.trie_cursor.tolist(), [0])

    def test_group_without_trie_has_no_cursor(self):
        group = BeamGroup(
            beam_width=2,
            stop_token_ids=[EOS_TOKEN_ID],
            max_new_tokens=8,
            device="cpu",
        )
        self.assertIsNone(group.trie)
        self.assertIsNone(group.trie_cursor)
        # Unconstrained groups keep working; nothing touches a cursor.
        group.advance_frontier(make_sel([1, 7], [0, 0], [-0.1, -0.2]))
        self.assertIsNone(group.trie_cursor)
        self.assertEqual(group.num_generated, 1)
        for got, want in zip(group.frontier_cum_logprobs.tolist(), [-0.1, -0.2]):
            self.assertAlmostEqual(got, want, places=5)

    def test_prefill_step_expands_one_row_to_k(self):
        group = self._group(beam_width=2)
        self.assertEqual(list(group.trie_cursor.shape), [1])
        group.advance_frontier(make_sel([1, 7], [0, 0], [-0.1, -0.2]))
        # 1 pseudo-row -> k rows, every row naming parent 0.
        self.assertEqual(list(group.trie_cursor.shape), [2])
        self.assertEqual(group.trie_cursor.tolist(), [self.n_a, self.n_c])
        # Same call, same tick: the two halves of launch state stay in step.
        for got, want in zip(group.frontier_cum_logprobs.tolist(), [-0.1, -0.2]):
            self.assertAlmostEqual(got, want, places=5)

    def test_decode_step_follows_parent_idx(self):
        group = self._group(beam_width=2)
        group.advance_frontier(make_sel([1, 7], [0, 0], [-0.1, -0.2]))
        self.assertEqual(group.trie_cursor.tolist(), [self.n_a, self.n_c])
        # parent_idx [1, 0]: row 0 inherits the 7-branch, row 1 the 1-branch.
        group.advance_frontier(make_sel([8, 5], [1, 0], [-0.3, -0.4]))
        self.assertEqual(group.trie_cursor.tolist(), [self.leaf_8, self.leaf_5])

    def test_short_layer_dummy_rows_keep_valid_cursor(self):
        group = self._group(beam_width=3)
        group.advance_frontier(make_sel([1, 1, 1], [0, 0, 0], [-0.1, -0.2, -0.3]))
        # Short layer: one real survivor, two dummy rows seated at parent 0
        # carrying the same token and -inf, so they can never win a selection.
        group.advance_frontier(
            make_sel(
                [2, 2, 2],
                [0, 0, 0],
                [-0.4, float("-inf"), float("-inf")],
                num_survivors=1,
            )
        )
        self.assertEqual(group.trie_cursor.tolist(), [self.n_b] * 3)
        self.assertNotIn(INVALID_NODE, group.trie_cursor.tolist())
        # A padded row still resolves to a legal node, so its mask is real.
        mask = self.t.build_mask(group.trie_cursor, VOCAB_SIZE)
        for row in range(3):
            self.assertEqual(allowed_of(mask[row]), [3, 4])

    def test_cursor_matches_generated_path(self):
        group = self._group(beam_width=2)
        # depth 3: [1, 2, 3] and [1, 2, 4].
        group.advance_frontier(make_sel([1, 1], [0, 0], [-0.1, -0.2]))
        group.advance_frontier(make_sel([2, 2], [0, 1], [-0.3, -0.4]))
        group.advance_frontier(make_sel([3, 4], [0, 1], [-0.5, -0.6]))
        self.assertEqual(group.num_generated, 3)
        cursor = group.trie_cursor.tolist()
        self.assertEqual(cursor, [self.leaf_3, self.leaf_4])
        for node in cursor:
            self.assertEqual(int(self.t.children_count[node]), 0)


if __name__ == "__main__":
    unittest.main()
