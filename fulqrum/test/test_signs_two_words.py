# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import random

import numpy as np
import pytest

from fulqrum.gpu.common import wide
from fulqrum.gpu.distributed.aabb_direct import sorted_beta_graph
from fulqrum.gpu.distributed.aabb_parents import (
    _residual_sell,
    parent_masks,
    parent_plan,
)
from fulqrum.gpu.distributed.arm_pair_tables import bb_words
from fulqrum.gpu.distributed.doubles_classes import double_class_tables
from fulqrum.gpu.distributed.hash_utils import _build_hash_table_simple
from fulqrum.gpu.distributed.tables import _alpha2_apid_asign

SIZES = (70, 100, 128)


def sign(s, p, q):
    """Return the fermionic sign of moving an electron between p and q in s."""
    lo, hi = min(p, q), max(p, q)
    between = s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
    return -1 if between.bit_count() & 1 else 1


def pair_id(p, q, m):
    lo, hi = min(p, q), max(p, q)
    return lo * (2 * m - lo - 1) // 2 + hi - lo - 1


def random_strings(rng, n, count):
    """Return distinct strings of n orbitals; at least a third use bits above 63."""
    out = set()
    while len(out) < count:
        value = rng.getrandbits(n)
        if len(out) % 3 == 0:
            value |= 1 << rng.randrange(64, n)
        out.add(value)
    return sorted(out)


def random_csr(rng, rows, n, per_row):
    """Return a CSR of (p < q) pairs per row; about half straddle bit 63."""
    ptr, p0, p1 = [0], [], []
    for _ in range(rows):
        for _ in range(per_row):
            if rng.random() < 0.5:
                p, q = rng.randrange(64), rng.randrange(64, n)
            else:
                p, q = sorted(rng.sample(range(n), 2))
            p0.append(p)
            p1.append(q)
        ptr.append(len(p0))
    return (
        np.array(ptr, dtype=np.int64),
        np.array(p0, dtype=np.uint8),
        np.array(p1, dtype=np.uint8),
    )


def all_straddles(rows, n):
    pairs = [(p, q) for p in range(64) for q in range(64, n)]
    ptr = np.arange(rows + 1, dtype=np.int64) * len(pairs)
    p0 = np.array([p for _ in range(rows) for p, _ in pairs], dtype=np.uint8)
    p1 = np.array([q for _ in range(rows) for _, q in pairs], dtype=np.uint8)
    return ptr, p0, p1


def edge_rows(ptr):
    return np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))


def csr_cases():
    rng = random.Random(7)
    for n in SIZES:
        values = random_strings(rng, n, 40)
        yield n, values, random_csr(rng, len(values), n, 12)
    values = random_strings(rng, 128, 3)
    yield 128, values, all_straddles(len(values), 128)


@pytest.mark.parametrize("n,values,csr", list(csr_cases()))
def test_beta_pair_words_match_the_oracle_across_words(n, values, csr):
    ptr, p0, p1 = csr
    word, pid, signs = bb_words(ptr, p0, p1, wide.pack_values(values, n), n)
    rows = edge_rows(ptr)
    want_sign = [sign(values[r], int(p), int(q)) for r, p, q in zip(rows, p0, p1)]
    want_pid = [pair_id(int(p), int(q), n) for p, q in zip(p0, p1)]
    assert signs.tolist() == want_sign
    assert pid.tolist() == want_pid
    negative = (word & np.uint32(0x80000000)) != 0
    assert negative.tolist() == [s < 0 for s in want_sign]
    assert (word & np.uint32(0x7FFFFFFF)).tolist() == want_pid


@pytest.mark.parametrize("n,values,csr", list(csr_cases()))
def test_alpha_pair_signs_match_the_oracle_across_words(n, values, csr):
    ptr, p0, p1 = csr
    apid, asign = _alpha2_apid_asign(ptr, p0, p1, wide.pack_values(values, n), n)
    rows = edge_rows(ptr)
    assert asign.dtype == np.int8 and apid.dtype == np.int32
    assert asign.tolist() == [
        sign(values[r], int(p), int(q)) for r, p, q in zip(rows, p0, p1)
    ]
    assert apid.tolist() == [pair_id(int(p), int(q), n) for p, q in zip(p0, p1)]
    assert int(apid.max()) < 0xFFFF


@pytest.mark.parametrize("n,values,csr", list(csr_cases()))
def test_sorted_beta_graph_words_match_the_oracle_across_words(n, values, csr):
    ptr, p0, p1 = csr
    rng = np.random.default_rng(n)
    columns = rng.integers(0, len(values), int(ptr[-1])).astype(np.int32)
    graph = sorted_beta_graph(
        ptr, columns, p0, p1, wide.pack_values(values, n), n, len(values)
    )
    rows = edge_rows(ptr)
    want = sorted(
        (
            int(r),
            int(j),
            int(j)
            | pair_id(int(p), int(q), n) << 32
            | (0xFFFF if sign(values[r], int(p), int(q)) < 0 else 1) << 48,
        )
        for r, j, p, q in zip(rows, columns, p0, p1)
    )
    got = list(zip(graph["rows"].tolist(), graph["columns"].tolist()))
    assert got == [(r, j) for r, j, _ in want]
    assert sorted(zip(got, graph["words"].tolist())) == sorted(
        ((r, j), w) for r, j, w in want
    )


def shift_case(rng, n, low, count):
    """Return strings inside bits [low, low+64) of n orbitals and their W=1 shift."""
    values = sorted({rng.getrandbits(64) << low for _ in range(count)})
    values = [v & ((1 << n) - 1) for v in values]
    return values, [v >> low for v in values]


@pytest.mark.parametrize("n,low", [(70, 6), (100, 30), (128, 40), (128, 64)])
def test_signs_are_unchanged_when_strings_move_across_the_word_boundary(n, low):
    rng = random.Random(n + low)
    wide_values, narrow_values = shift_case(rng, n, low, 30)
    top = min(64, n - low)
    ptr, p0, p1 = [0], [], []
    for _ in wide_values:
        for _ in range(16):
            p, q = sorted(rng.sample(range(top), 2))
            p0.append(p)
            p1.append(q)
        ptr.append(len(p0))
    ptr = np.array(ptr, dtype=np.int64)
    p0, p1 = np.array(p0, dtype=np.uint8), np.array(p1, dtype=np.uint8)
    wide_p0, wide_p1 = p0 + np.uint8(low), p1 + np.uint8(low)
    narrow = np.array(narrow_values, dtype=np.uint64)
    packed = wide.pack_values(wide_values, n)
    assert packed.shape == (len(wide_values), 2)
    _, _, narrow_sign = bb_words(ptr, p0, p1, narrow, top)
    _, _, wide_sign = bb_words(ptr, wide_p0, wide_p1, packed, n)
    assert np.array_equal(narrow_sign, wide_sign)
    _, narrow_asign = _alpha2_apid_asign(ptr, p0, p1, narrow, top)
    _, wide_asign = _alpha2_apid_asign(ptr, wide_p0, wide_p1, packed, n)
    assert np.array_equal(narrow_asign, wide_asign)


def pad(value, at, count):
    """Insert count empty orbitals at bit position at."""
    low = value & ((1 << at) - 1)
    return low | ((value >> at) << (at + count))


def test_padded_strings_give_the_single_word_signs_and_shifted_pairs():
    rng = random.Random(3)
    n, extra = 60, 60
    values = random_strings_narrow(rng, n, 40)
    ptr, p0, p1 = random_csr_narrow(rng, len(values), n, 10)
    narrow = np.array(values, dtype=np.uint64)
    padded = wide.pack_values([pad(v, 3, extra) for v in values], n + extra)
    wp0, wp1 = (np.where(p >= 3, p + extra, p).astype(np.uint8) for p in (p0, p1))
    _, _, narrow_sign = bb_words(ptr, p0, p1, narrow, n)
    _, wide_pid, wide_sign = bb_words(ptr, wp0, wp1, padded, n + extra)
    assert np.array_equal(narrow_sign, wide_sign)
    assert wide_pid.tolist() == [
        pair_id(int(p), int(q), n + extra) for p, q in zip(wp0, wp1)
    ]
    _, narrow_asign = _alpha2_apid_asign(ptr, p0, p1, narrow, n)
    _, wide_asign = _alpha2_apid_asign(ptr, wp0, wp1, padded, n + extra)
    assert np.array_equal(narrow_asign, wide_asign)
    columns = np.arange(int(ptr[-1]), dtype=np.int32) % len(values)
    g1 = sorted_beta_graph(ptr, columns, p0, p1, narrow, n, len(values))
    g2 = sorted_beta_graph(ptr, columns, wp0, wp1, padded, n + extra, len(values))
    assert np.array_equal(g1["rows"], g2["rows"])
    assert np.array_equal(g1["columns"], g2["columns"])
    assert np.array_equal(g1["words"] >> np.uint64(48), g2["words"] >> np.uint64(48))


def random_strings_narrow(rng, n, count):
    out = set()
    while len(out) < count:
        out.add(rng.getrandbits(n))
    return sorted(out)


def random_csr_narrow(rng, rows, n, per_row):
    ptr, p0, p1 = [0], [], []
    for _ in range(rows):
        for _ in range(per_row):
            p, q = sorted(rng.sample(range(n), 2))
            p0.append(p)
            p1.append(q)
        ptr.append(len(p0))
    return (
        np.array(ptr, dtype=np.int64),
        np.array(p0, dtype=np.uint8),
        np.array(p1, dtype=np.uint8),
    )


def clique_strings(rng, n, parents, electrons, kind="union"):
    """Return cliques of strings that share one parent across both words.

    Union cliques remove one electron from a parent; intersection cliques
    add one electron to a parent core.
    """
    out = set()
    for _ in range(parents):
        count = electrons + 1 if kind == "union" else electrons - 1
        upper = min(count // 2, (n - 64) // 2)
        low = rng.sample(range(64), count - upper)
        high = rng.sample(range(64, n), upper)
        parent = sum(1 << o for o in low + high)
        if kind == "union":
            out.update(parent & ~(1 << o) for o in low + high)
        else:
            empty = [o for o in range(n) if not (parent >> o) & 1]
            below = [o for o in empty if o < 64]
            above = [o for o in empty if o >= 64]
            chosen = rng.sample(below, 10 - min(5, len(above)))
            chosen += rng.sample(above, min(5, len(above)))
            out.update(parent | (1 << o) for o in chosen)
    return sorted(out)


def single_edges(values, m):
    """Pack every directed single excitation between values as SELL words."""
    rows, words = [], []
    for r, a in enumerate(values):
        for j, b in enumerate(values):
            x = a ^ b
            if r == j or x.bit_count() != 2:
                continue
            p, q = [i for i in range(m) if (x >> i) & 1]
            s = sign(b, p, q)
            rows.append(r)
            words.append(j | pair_id(p, q, m) << 32 | (s & 0xFFFF) << 48)
    order = np.argsort(rows, kind="stable")
    return _residual_sell(
        np.array(rows, dtype=np.int64)[order],
        np.array(words, dtype=np.uint64)[order],
        len(values),
    )


def check_members(plan, values, kind):
    ptr, members = plan["parent_ptr"], plan["members"]
    for g in range(len(ptr) - 1):
        group = members[ptr[g] : ptr[g + 1]].tolist()
        strings = [values[w & 0xFFFFFFFF] for w in group]
        key = 0 if kind == "union" else (1 << 200) - 1
        for s in strings:
            key = key | s if kind == "union" else key & s
        for w, s in zip(group, strings):
            orbital = (w >> 32) & 0xFF
            phase = (w >> 48) & 0xFFFF
            assert (key ^ s) == 1 << orbital
            want = -1 if (key & ((1 << orbital) - 1)).bit_count() & 1 else 1
            assert phase == want & 0xFFFF
    return ptr, members


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("kind", ["union", "intersection"])
def test_parent_plan_matches_the_oracle_across_words(n, kind):
    rng = random.Random(n)
    values = clique_strings(rng, n, 6, 9, kind)
    off, lengths, pack = single_edges(values, n)
    plan = parent_plan(wide.pack_values(values, n), off, lengths, pack, n, 8, kind)
    assert plan["stats"]["electron_count"] == 9
    assert plan["stats"]["parents"] >= 6
    ptr, members = check_members(plan, values, kind)
    orbitals = (members >> np.uint64(32)) & np.uint64(0xFF)
    assert np.any(orbitals < 64) and np.any(orbitals >= 64)
    masks, ids = parent_masks(plan, words=2)
    assert masks.shape == (len(ptr) - 1, 2)
    for g in range(len(ptr) - 1):
        want = 0
        for w in members[ptr[g] : ptr[g + 1]].tolist():
            want |= 1 << ((w >> 32) & 0xFF)
        assert wide.unpack_ints(masks[g : g + 1]) == [want]
    assert ids.tolist() == list(range(len(ptr) - 1))


def test_parent_plan_rejects_a_two_word_sign_error():
    rng = random.Random(5)
    values = clique_strings(rng, 128, 3, 9)
    off, lengths, pack = single_edges(values, 128)
    flipped = pack.copy()
    live = np.flatnonzero((flipped & np.uint64(0xFFFFFFFF)) != np.uint64(0))
    flipped[live[0]] ^= np.uint64(0xFFFE) << np.uint64(48)
    with pytest.raises(ValueError, match="signs"):
        parent_plan(wide.pack_values(values, 128), off, lengths, flipped, 128, 8)


@pytest.mark.parametrize("kind", ["union", "intersection"])
def test_padded_parent_plan_equals_the_single_word_plan(kind):
    rng = random.Random(11)
    n, extra = 40, 70
    values = set()
    for _ in range(5):
        parent = sum(1 << o for o in rng.sample(range(n), 10))
        for o in range(n):
            if ((parent >> o) & 1) == (kind == "union"):
                values.add(parent ^ (1 << o))
    values = sorted(values)
    padded = [pad(v, 3, extra) for v in values]
    narrow = parent_plan(
        np.array(values, dtype=np.uint64), *single_edges(values, n), n, 8, kind
    )
    wide_plan = parent_plan(
        wide.pack_values(padded, n + extra),
        *single_edges(padded, n + extra),
        n + extra,
        8,
        kind,
    )
    assert np.array_equal(narrow["parent_ptr"], wide_plan["parent_ptr"])
    a, b = narrow["members"], wide_plan["members"]
    field = np.uint64(0xFF) << np.uint64(32)
    orbital_a = ((a & field) >> np.uint64(32)).astype(np.int64)
    orbital_b = ((b & field) >> np.uint64(32)).astype(np.int64)
    assert np.array_equal(
        np.where(orbital_a >= 3, orbital_a + extra, orbital_a), orbital_b
    )
    assert np.array_equal(a & ~field, b & ~field)
    assert narrow["stats"]["parents"] >= 5
    for key in ("parents", "members", "selected_edges", "residual_edges"):
        assert narrow["stats"][key] == wide_plan["stats"][key]


def two_members(first, second):
    return {
        "parent_ptr": np.array([0, 2], dtype=np.int32),
        "members": np.array([o << 32 | 1 << 48 for o in (first, second)], np.uint64),
    }


def test_parent_masks_keep_the_single_word_form():
    plan = {
        "parent_ptr": np.array([0, 2, 5], dtype=np.int32),
        "members": np.array(
            [o << 32 | 1 << 48 for o in (1, 63, 0, 5, 7)], dtype=np.uint64
        ),
    }
    masks, _ = parent_masks(plan)
    assert masks.dtype == np.uint64 and masks.shape == (2,)
    assert masks.tolist() == [(1 << 1) | (1 << 63), 1 | (1 << 5) | (1 << 7)]
    with pytest.raises(ValueError, match="ascending"):
        parent_masks(two_members(3, 64))
    masks, _ = parent_masks(two_members(3, 127), words=2)
    assert wide.unpack_ints(masks) == [(1 << 3) | (1 << 127)]


def class_case(rng, rows, cols, n, density):
    offsets, columns = [0], []
    for _ in range(rows):
        row = sorted(rng.sample(range(cols), max(1, int(density * cols))))
        columns.extend(row)
        offsets.append(len(columns))
    values = np.array([rng.choice([-1.0, 1.0]) * rng.random() for _ in columns])
    return np.array(offsets, dtype=np.int64), np.array(columns, dtype=np.int32), values


def class_strings(rng, count, n, electrons, low=0):
    out = set()
    while len(out) < count:
        out.add(sum(1 << (low + o) for o in rng.sample(range(n), electrons)))
    return sorted(out)


def assert_same_class_tables(a, b):
    for name in (
        "blocks",
        "block_offsets",
        "tile_info",
        "row_members",
        "col_members",
        "residual_offsets",
        "residual_columns",
        "residual_values",
        "in_class",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.stats == b.stats


@pytest.mark.parametrize("low", [20, 40, 64])
def test_double_classes_are_unchanged_when_strings_move_across_the_word_boundary(low):
    rng = random.Random(low)
    out = class_strings(rng, 50, 12, 4)
    src = class_strings(rng, 60, 12, 4)
    offsets, columns, values = class_case(rng, len(out), len(src), 12, 0.3)
    narrow = double_class_tables(
        offsets, columns, values, np.array(out, np.uint64), np.array(src, np.uint64), 2
    )
    wide_tables = double_class_tables(
        offsets,
        columns,
        values,
        wide.pack_values([v << low for v in out], 128),
        wide.pack_values([v << low for v in src], 128),
        2,
    )
    assert narrow.stats["classes"] > 0
    assert_same_class_tables(narrow, wide_tables)


def test_padded_double_classes_equal_the_single_word_classes():
    rng = random.Random(2)
    out = class_strings(rng, 40, 10, 3)
    src = class_strings(rng, 45, 10, 3)
    offsets, columns, values = class_case(rng, len(out), len(src), 10, 0.4)
    narrow = double_class_tables(
        offsets, columns, values, np.array(out, np.uint64), np.array(src, np.uint64), 2
    )
    padded = double_class_tables(
        offsets,
        columns,
        values,
        wide.pack_values([pad(v, 3, 100) for v in out], 110),
        wide.pack_values([pad(v, 3, 100) for v in src], 110),
        2,
    )
    assert_same_class_tables(narrow, padded)


def test_row_unique_matches_python_ints():
    rng = random.Random(4)
    pool = random_strings(rng, 128, 30)
    values = [rng.choice(pool) for _ in range(200)]
    keys, inverse, counts = wide.unique_w(
        wide.pack_values(values, 128), return_counts=True
    )
    distinct = sorted(set(values))
    assert wide.unpack_ints(keys) == distinct
    assert [distinct[i] for i in inverse.tolist()] == values
    assert counts.tolist() == [values.count(v) for v in distinct]
    single = np.array([v & ((1 << 64) - 1) for v in values], dtype=np.uint64)
    got = wide.unique_w(single, return_counts=True)
    want = np.unique(single, return_inverse=True, return_counts=True)
    for a, b in zip(got, want):
        assert np.array_equal(a, b)
    empty = wide.unique_w(np.zeros((0, 2), dtype=np.uint64), return_counts=True)
    assert [len(x) for x in empty] == [0, 0, 0]


def test_single_bit_index_and_low_parity_across_words():
    orbitals = np.arange(128)
    bits = wide.orbital_bits_w(orbitals, 2)
    assert wide.single_bit_index_w(bits).tolist() == orbitals.tolist()
    rng = random.Random(9)
    values = random_strings(rng, 128, 20)
    packed = wide.pack_values(values, 128)
    for orbital in (0, 1, 63, 64, 65, 127):
        got = wide.below_parity_w(packed, np.full(len(values), orbital))
        want = [-1 if (v & ((1 << orbital) - 1)).bit_count() & 1 else 1 for v in values]
        assert got.tolist() == want


def test_string_hash_tables_reject_two_word_strings():
    with pytest.raises(AssertionError):
        _build_hash_table_simple(np.zeros((4, 2), dtype=np.uint64))
