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


import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest

from fulqrum.exceptions import FulqrumError
from fulqrum.gpu.common import wide
from fulqrum.gpu.distributed import rdm_cart

XP = SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("_")})
XP.asnumpy = np.asarray


def strings_with(rng, n, electrons, count, high=True):
    out = set()
    while len(out) < count:
        occupied = rng.sample(range(n), electrons)
        if high and len(out) % 2 == 0 and not any(o >= 64 for o in occupied):
            occupied[0] = rng.randrange(64, n)
            if len(set(occupied)) < electrons:
                continue
        out.add(sum(1 << o for o in occupied))
    return sorted(out)


def sign(s, p, q):
    lo, hi = min(p, q), max(p, q)
    between = s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
    return -1.0 if between.bit_count() & 1 else 1.0


def test_word_index_finds_rows_exactly():
    rng = random.Random(0)
    keys = wide.pack_values(strings_with(rng, 128, 6, 300), 128)
    index = rdm_cart._WordIndex(np, keys)
    queries = np.concatenate(
        [
            keys[rng.sample(range(len(keys)), 100)],
            keys[:50] ^ np.array([1, 0], np.uint64),
            keys[:50] ^ np.array([0, 1 << 3], np.uint64),
            keys[:20] ^ np.array([1 << 40, 1 << 9], np.uint64),
        ]
    )
    found, rows = index.find(queries)
    table = {tuple(k): i for i, k in enumerate(keys.tolist())}
    for query, hit, row in zip(queries.tolist(), found.tolist(), rows.tolist()):
        want = table.get(tuple(query))
        assert hit == (want is not None)
        if hit:
            assert row == want


def test_word_index_misses_words_absent_from_every_key():
    keys = np.array([[1, 5], [2, 5], [1, 6]], dtype=np.uint64)
    index = rdm_cart._WordIndex(np, keys)
    found, rows = index.find(np.array([[2, 6], [3, 5], [1, 7], [1, 6]], np.uint64))
    assert found.tolist() == [False, False, False, True]
    assert rows[3] == 2


@pytest.mark.parametrize("n", [70, 100, 128])
def test_occupations_and_signs_read_both_words(n):
    rng = random.Random(n)
    values = strings_with(rng, n, 5, 40)
    keys = wide.pack_values(values, n)
    occ, vir = rdm_cart._occ_vir(np, keys, n, 5)
    for value, o, v in zip(values, occ.tolist(), vir.tolist()):
        assert o == [i for i in range(n) if (value >> i) & 1]
        assert v == [i for i in range(n) if not (value >> i) & 1]
    rows = np.repeat(np.arange(len(values)), 6)
    created = np.array([rng.randrange(n) for _ in rows])
    removed = np.array(
        [rng.randrange(64) if k % 2 else rng.randrange(n) for k in range(len(rows))]
    )
    got = rdm_cart._excite_sign(np, keys[rows], created, removed)
    want = [sign(values[r], int(c), int(d)) for r, c, d in zip(rows, created, removed)]
    assert got.dtype == np.float64 and got.tolist() == want
    assert rdm_cart._fixed_ne(keys) == 5
    empty = next(o for o in range(n - 1, -1, -1) if not (values[3] >> o) & 1)
    mixed = wide.pack_values(values[:3] + [values[3] | 1 << empty], n)
    assert rdm_cart._fixed_ne(mixed) is None


def pad(value, at, count):
    low = value & ((1 << at) - 1)
    return low | ((value >> at) << (at + count))


def remap_label(label, m, extra, at=3):
    p, q = divmod(int(label), m)
    shift = lambda o: o + extra if o >= at else o
    return shift(p) * (m + extra) + shift(q)


@pytest.mark.parametrize("two", [False, True])
def test_padded_lists_equal_the_single_word_lists(two):
    rng = random.Random(3)
    m, ne, extra = 10, 4, 100
    values = strings_with(rng, m, ne, 60, high=False)
    padded = [pad(v, 3, extra) for v in values]
    starts = np.array([0, 25, 60], np.int64)
    narrow = rdm_cart._sector_lists(
        XP, np.array(values, np.uint64), (5, 45), m, ne, two, 1 << 16, starts
    )
    wide_lists = rdm_cart._sector_lists(
        XP,
        wide.pack_values(padded, m + extra),
        (5, 45),
        m + extra,
        ne,
        two,
        1 << 16,
        starts,
    )
    assert narrow["source"].size > 0
    for key in ("source", "target", "sign"):
        assert np.array_equal(narrow[key], wide_lists[key]), key
    assert [remap_label(x, m, extra) for x in narrow["label"]] == wide_lists[
        "label"
    ].tolist()
    if two:
        assert np.array_equal(narrow["edge_ptr"], wide_lists["edge_ptr"])
        assert np.array_equal(narrow["cols"], wide_lists["cols"])


def test_two_word_lists_match_python_integers():
    rng = random.Random(9)
    m, ne = 100, 3
    values = strings_with(rng, m, ne, 50)
    keys = wide.pack_values(values, m)
    starts = np.array([0, len(values)], np.int64)
    lists = rdm_cart._sector_lists(
        XP, keys, (0, len(values)), m, ne, True, 1 << 16, starts
    )
    position = {v: i for i, v in enumerate(values)}
    expected = set()
    for r, value in enumerate(values):
        for p in range(m):
            if (value >> p) & 1:
                expected.add((r, r, 1.0, p * m + p))
                for q in range(m):
                    if not (value >> q) & 1:
                        target = value ^ (1 << p) ^ (1 << q)
                        if target in position:
                            expected.add(
                                (r, position[target], sign(value, q, p), q * m + p)
                            )
    got = set(
        zip(
            lists["source"].tolist(),
            lists["target"].tolist(),
            lists["sign"].tolist(),
            lists["label"].tolist(),
        )
    )
    assert got == expected
    neighbours = []
    for r, value in enumerate(values):
        row = [
            j for j, other in enumerate(values) if (value ^ other).bit_count() in (2, 4)
        ]
        neighbours.append(sorted(row))
    cols = lists["cols"]
    ptr = lists["edge_ptr"]
    for r in range(len(values)):
        assert sorted(cols[ptr[r] : ptr[r + 1]].tolist()) == neighbours[r]


def kernel_same_diagonal(weight, values, m):
    out = np.zeros((m, m, m, m))
    occupation = np.zeros((m, m))
    for w, value in zip(weight, values):
        if w == 0.0:
            continue
        bits = [i for i in range(m) if (value >> i) & 1]
        for u, v in itertools.permutations(bits, 2):
            occupation[u, v] += w
    for u in range(m):
        for v in range(m):
            if occupation[u, v] != 0.0:
                out[u, u, v, v] += occupation[u, v]
                out[u, v, v, u] -= occupation[u, v]
    return out


@pytest.mark.parametrize("m", [8, 70])
def test_same_diagonal_gemm_matches_the_kernel_loop(m):
    rng = random.Random(m)
    values = strings_with(rng, m, 4, 30, high=m > 64)
    weight = np.array([rng.random() for _ in values])
    weight[::7] = 0.0
    keys = wide.pack_values(values, m)
    keys = keys if keys.ndim == 2 else keys.reshape(-1, 1)
    out = np.zeros(m**4)
    rdm_cart._same_diagonal_gemm(np, weight, keys, out, m)
    want = kernel_same_diagonal(weight, values, m)
    assert np.abs(out.reshape(m, m, m, m) - want).max() <= 1e-14 * np.abs(want).max()


def test_rank_two_with_two_words_is_refused():
    ham = SimpleNamespace(
        _num_alpha=70,
        _num_beta=70,
        _wide_strings=True,
        _alpha_uniq=np.zeros((1, 2), np.uint64),
        _beta_uniq=np.zeros((1, 2), np.uint64),
    )
    pytest.importorskip("cupy")
    with pytest.raises(FulqrumError, match="rank=2 RDMs support at most 64"):
        rdm_cart.compute_rdms_cart(ham, np.zeros(1), rank=2)


def test_more_than_128_orbitals_are_refused():
    ham = SimpleNamespace(_num_alpha=129, _num_beta=129)
    pytest.importorskip("cupy")
    with pytest.raises(FulqrumError, match="at most 128 orbitals"):
        rdm_cart.compute_rdms_cart(ham, np.zeros(1), rank=1)
