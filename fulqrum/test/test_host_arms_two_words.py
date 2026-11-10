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

from fulqrum.gpu import config
from fulqrum.gpu.common import wide
from fulqrum.gpu.distributed import wide_sign_check
from fulqrum.gpu.distributed.aabb_cliques import beta_clique_plan
from fulqrum.gpu.distributed.aabb_direct import sorted_beta_graph
from fulqrum.gpu.distributed.aabb_lbg import small_cliques
from fulqrum.gpu.distributed.aabb_parents import (
    _residual_sell,
    orbital_g,
    validate_clique_plan,
)
from fulqrum.gpu.distributed.arm_pair_tables import aa_table, occupation_bits
from fulqrum.gpu.distributed.launch import single_word_strings
from fulqrum.gpu.distributed.tables import (
    chunked_pair_table,
    pair_table,
    pair_table_rows,
)


def factor_rows(rng, rows, orbitals, asymmetric_row=None):
    pairs = orbitals * (orbitals - 1) // 2
    forward = rng.standard_normal((rows, pairs))
    factors = np.repeat(forward, 2, axis=1)
    factors[:, 1::2] += 1e-15 * rng.standard_normal((rows, pairs))
    if asymmetric_row is not None:
        factors[asymmetric_row, 1] += 1e-3
    return factors


@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("transpose", [False, True])
def test_chunked_pair_table_equals_the_whole_table(rows, transpose):
    rng = np.random.default_rng(rows)
    factors = factor_rows(rng, rows, 9)
    whole = pair_table(factors, 9, 1e-12)
    if transpose:
        whole = np.ascontiguousarray(whole.T)
    for chunk in sorted({1, 3, rows, rows + 5, max(1, rows // 2)}):
        got = chunked_pair_table(
            lambda a, b: factors[a:b], rows, 9, 1e-12, chunk, transpose=transpose
        )
        assert got.flags.c_contiguous
        assert got.shape == whole.shape
        assert got.tobytes() == whole.tobytes()


@pytest.mark.parametrize("row", [0, 5, 12])
def test_chunked_pair_table_raises_where_the_whole_table_raises(row):
    rng = np.random.default_rng(3)
    factors = factor_rows(rng, 13, 6, asymmetric_row=row)
    with pytest.raises(ValueError, match="direction"):
        pair_table(factors, 6, 1e-12)
    for chunk in (1, 4, 13):
        with pytest.raises(ValueError, match="direction"):
            chunked_pair_table(lambda a, b: factors[a:b], 13, 6, 1e-12, chunk)


def test_chunked_pair_table_uses_the_global_scale():
    factors = np.array([[1.0, 1.0], [1e6, 1e6 + 1e-7]])
    pair_table(factors, 2, 1e-12)
    for chunk in (1, 2):
        chunked_pair_table(lambda a, b: factors[a:b], 2, 2, 1e-12, chunk)
    small_only = factors[:1].copy()
    small_only[0, 1] += 1e-7
    with pytest.raises(ValueError):
        pair_table(small_only, 2, 1e-12)
    with pytest.raises(ValueError):
        chunked_pair_table(lambda a, b: small_only[a:b], 1, 2, 1e-12, 1)


def test_chunked_pair_table_keeps_not_a_number_handling():
    factors = np.array([[1.0, 1.0], [np.nan, np.nan], [2.0, 2.5]])
    for values in (factors, factors[::-1].copy()):
        try:
            pair_table(values, 2, 1e-12)
            whole = None
        except ValueError:
            whole = "raised"
        for chunk in (1, 2, 3):
            try:
                chunked_pair_table(lambda a, b, v=values: v[a:b], 3, 2, 1e-12, chunk)
                got = None
            except ValueError:
                got = "raised"
            assert got == whole


def test_chunk_rows_fit_the_budget():
    pairs = 128 * 127 // 2
    rows = pair_table_rows(pairs)
    assert rows >= 1 and 8 * rows * 2 * pairs * 8 <= 1 << 30
    assert pair_table_rows(1, chunk_bytes=1) == 1


def test_occupation_bits_read_both_words():
    rng = random.Random(1)
    values = [rng.getrandbits(128) for _ in range(30)]
    got = occupation_bits(wide.pack_values(values, 128), 128)
    want = [[(v >> o) & 1 for o in range(128)] for v in values]
    assert got.dtype == np.float64 and got.tolist() == want


def test_single_word_occupation_bits_are_unchanged():
    rng = np.random.default_rng(2)
    strings = rng.integers(0, 1 << 63, 40, dtype=np.uint64) | np.uint64(1 << 63)
    for n in (1, 30, 64):
        shifts = np.arange(n, dtype=np.uint64)
        want = ((strings[:, None] >> shifts[None, :]) & np.uint64(1)).astype(float)
        assert occupation_bits(strings, n).tobytes() == want.tobytes()


def test_padded_pair_tables_equal_the_single_word_tables():
    rng = random.Random(4)
    narrow_values = sorted({rng.getrandbits(20) for _ in range(25)})
    padded = wide.pack_values([v << 60 for v in narrow_values], 80)
    d = np.random.default_rng(4).standard_normal((7, 20))
    narrow = aa_table(d, np.array(narrow_values, dtype=np.uint64), 20)
    d_wide = np.zeros((7, 80))
    d_wide[:, 60:] = d
    assert aa_table(d_wide, padded, 80).tobytes() == narrow.tobytes()


def pad(value, at, count):
    low = value & ((1 << at) - 1)
    return low | ((value >> at) << (at + count))


def clique_values(rng, n, parents, electrons):
    out = set()
    for _ in range(parents):
        parent = sum(1 << o for o in rng.sample(range(n), electrons + 1))
        out.update(parent & ~(1 << o) for o in range(n) if (parent >> o) & 1)
    return sorted(out)


def beta_graph(values, n):
    rows, columns, p0, p1 = [], [], [], []
    for r, a in enumerate(values):
        for j, b in enumerate(values):
            x = a ^ b
            if r != j and x.bit_count() == 2:
                p, q = [i for i in range(n) if (x >> i) & 1]
                rows.append(r)
                columns.append(j)
                p0.append(p)
                p1.append(q)
    ptr = np.searchsorted(np.array(rows), np.arange(len(values) + 1))
    return ptr, np.array(columns), np.array(p0), np.array(p1)


def graph_for(values, n, strings):
    ptr, columns, p0, p1 = beta_graph(values, n)
    return sorted_beta_graph(ptr, columns, p0, p1, strings, n, len(values))


def test_padded_beta_clique_plan_equals_the_single_word_plan():
    rng = random.Random(6)
    n, extra = 30, 70
    values = clique_values(rng, n, 5, 7)
    padded = [pad(v, 3, extra) for v in values]
    narrow_strings = np.array(values, dtype=np.uint64)
    wide_strings = wide.pack_values(padded, n + extra)
    narrow = beta_clique_plan(
        narrow_strings, graph_for(values, n, narrow_strings), n, 4, kind="union"
    )
    wide_plan = beta_clique_plan(
        wide_strings,
        graph_for(padded, n + extra, wide_strings),
        n + extra,
        4,
        kind="union",
    )
    assert narrow["stats"]["parents"] >= 5
    assert np.array_equal(narrow["vptr"], wide_plan["vptr"])
    assert np.array_equal(narrow["vids"], wide_plan["vids"])
    field = np.uint64(0xFF) << np.uint64(32)
    a, b = narrow["vmembers"], wide_plan["vmembers"]
    orbital_a = ((a & field) >> np.uint64(32)).astype(np.int64)
    orbital_b = ((b & field) >> np.uint64(32)).astype(np.int64)
    shifted = np.where(orbital_a >= 3, orbital_a + extra, orbital_a)
    assert np.array_equal(shifted, orbital_b)
    assert np.array_equal(a & ~field, b & ~field)
    assert np.array_equal(
        narrow["vres_words"] >> np.uint64(48), wide_plan["vres_words"] >> np.uint64(48)
    )


def test_padded_small_cliques_equal_the_single_word_groups():
    rng = random.Random(8)
    n, extra = 24, 80
    values = []
    for _ in range(6):
        parent = sum(1 << o for o in rng.sample(range(n), 4))
        values += [parent & ~(1 << o) for o in range(n) if (parent >> o) & 1]
    values = sorted(set(values))
    padded = [pad(v, 3, extra) for v in values]

    def residual(vals, m, strings):
        graph = graph_for(vals, m, strings)
        rows = np.asarray(graph["rows"], dtype=np.int64)
        return _residual_sell(rows, np.asarray(graph["words"]), len(vals))

    narrow_strings = np.array(values, dtype=np.uint64)
    wide_strings = wide.pack_values(padded, n + extra)
    try:
        narrow = small_cliques(
            narrow_strings, *residual(values, n, narrow_strings), None, n
        )
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).split(";")[0]):
            small_cliques(
                wide_strings,
                *residual(padded, n + extra, wide_strings),
                None,
                n + extra,
            )
        return
    wide_groups = small_cliques(
        wide_strings, *residual(padded, n + extra, wide_strings), None, n + extra
    )
    assert narrow.keys() == wide_groups.keys()
    for key in narrow:
        for x, y in zip(narrow[key][:2], wide_groups[key][:2]):
            assert np.array_equal(x, y)


def test_clique_route_accepts_two_word_orbital_counts():
    g = np.zeros((1, 128 * 127 // 2))
    assert orbital_g(g, 128).shape == (1, 128, 128)
    with pytest.raises(ValueError):
        orbital_g(np.zeros((1, 129 * 128 // 2)), 129)


def test_clique_plan_orbital_limit_is_two_words():
    base = {
        "transposed": True,
        "nparents": 1,
        "residual_edges": 0,
        "n_beta_edges": 0,
        "n_rows": 32,
        "n_b_src": 32,
        "n_b_out": 32,
        "tile_rows": 32,
        "eb": 8,
        "grouped_edges": None,
        "max_shared_bytes": 1 << 20,
    }
    for orbitals, ok in ((128, True), (129, False)):
        try:
            validate_clique_plan(orbitals=orbitals, **base)
            raised = None
        except ValueError as error:
            raised = str(error)
        assert (raised is None or "orbitals" not in raised) == ok


def test_string_reading_launches_refuse_two_word_strings():
    class Device:
        def __init__(self, ndim):
            self.ndim = ndim
            self.data = type("Pointer", (), {"ptr": 1234})()

    assert single_word_strings(None, "x") == 0
    assert single_word_strings(Device(1), "x") == 1234
    with pytest.raises(ValueError, match="cross SELL"):
        single_word_strings(Device(2), "cross SELL")


def test_wide_sign_check_flag_is_registered():
    assert config.REGISTRY["FQ_WIDE_SIGN_CHECK"].group == config.TEST
    snapshot = config.GpuConfig.from_environ({})
    assert snapshot.value("FQ_WIDE_SIGN_CHECK") == "auto"
    assert wide_sign_check.check_enabled("auto", (1, 2))
    assert not wide_sign_check.check_enabled("auto", (1, 1))
    on = config.GpuConfig.from_environ({"FQ_WIDE_SIGN_CHECK": "1"})
    assert wide_sign_check.check_enabled(on.value("FQ_WIDE_SIGN_CHECK"), (1, 1))
    off = config.GpuConfig.from_environ({"FQ_WIDE_SIGN_CHECK": "0"})
    assert not wide_sign_check.check_enabled(off.value("FQ_WIDE_SIGN_CHECK"), (2, 2))


int_sign = wide_sign_check._int_sign


def doubles_case(rng, n, rows, flip=None):
    """Return double connections whose values carry the occupation signs."""
    strings = sorted(
        {rng.getrandbits(n) | (1 << rng.randrange(64, n)) for _ in range(rows)}
    )
    coefficient = {}
    ptr, positions, values = [0], [[], [], [], []], []
    for s in strings:
        for _ in range(12):
            a, b, c, d = sorted(rng.sample(range(n), 4))
            key = (a, b, c, d, tuple((s >> o) & 1 for o in (a, b, c, d)))
            base = coefficient.setdefault(key, rng.choice([-1, 1]) * rng.uniform(1, 2))
            spectators = s & ~((1 << a) | (1 << b) | (1 << c) | (1 << d))
            sign = int_sign(spectators, a, b) * int_sign(spectators, c, d)
            for k, o in enumerate((a, b, c, d)):
                positions[k].append(o)
            values.append(base * sign)
        ptr.append(len(values))
    values = np.array(values)
    if flip is not None:
        values[flip] = -values[flip]
    return (
        np.array(ptr),
        [np.array(p, dtype=np.uint8) for p in positions],
        values,
        wide.pack_values(strings, n),
    )


def test_double_sign_check_passes_consistent_signs_and_finds_a_flip():
    rng = random.Random(10)
    ptr, positions, values, bra = doubles_case(rng, 128, 40)
    assert wide_sign_check.double_sign_error(ptr, positions, values, bra, "t") is None
    a, b, c, d = 2, 40, 70, 100
    moved = (1 << a) | (1 << c)
    strings = [moved | (1 << 20) | (1 << 80), moved | (1 << 80)]
    s_first, s_second = (
        int_sign(s & ~moved, a, b) * int_sign(s & ~moved, c, d) for s in strings
    )
    assert s_first != s_second
    positions = [np.array([o, o], dtype=np.uint8) for o in (a, b, c, d)]
    good = np.array([0.5 * s_first, 0.5 * s_second])
    bra = wide.pack_values(strings, 128)
    ptr = np.array([0, 1, 2])
    assert wide_sign_check.double_sign_error(ptr, positions, good, bra, "t") is None
    bad = np.array([0.5, 0.5])
    error = wide_sign_check.double_sign_error(ptr, positions, bad, bra, "alpha doubles")
    assert error is not None and "alpha doubles" in error


def test_beta_word_sign_check_finds_a_flipped_sign():
    rng = random.Random(12)
    n = 100
    values = sorted({rng.getrandbits(n) for _ in range(30)})
    strings = wide.pack_values(values, n)
    ptr, columns, p0, p1 = [0], [], [], []
    for _ in values:
        for _ in range(8):
            p, q = rng.randrange(64), rng.randrange(64, n)
            columns.append(rng.randrange(len(values)))
            p0.append(p)
            p1.append(q)
        ptr.append(len(p0))
    graph = sorted_beta_graph(
        np.array(ptr),
        np.array(columns),
        np.array(p0),
        np.array(p1),
        strings,
        n,
        len(values),
    )
    assert wide_sign_check.beta_word_sign_error(graph, strings, n, "t") is None
    for k in range(len(graph["words"])):
        pid = int((int(graph["words"][k]) >> 32) & 0xFFFF)
        assert wide_sign_check._pair_of(pid, n) in {
            (int(a), int(b)) for a, b in zip(p0, p1)
        }
    graph["words"] = graph["words"].copy()
    graph["words"][3] ^= np.uint64(0xFFFE) << np.uint64(48)
    error = wide_sign_check.beta_word_sign_error(graph, strings, n, "beta edges")
    assert error is not None and "beta edges" in error


def test_sign_check_samples_at_most_the_requested_edges():
    picked = wide_sign_check._sample(100_000, 10_000, 3)
    assert len(picked) == 10_000 and len(np.unique(picked)) == 10_000
    assert np.array_equal(wide_sign_check._sample(50, 10_000, 3), np.arange(50))


def test_pair_ids_round_trip():
    for n in (2, 5, 64, 128):
        for p in range(n):
            for q in range(p + 1, n):
                pid = p * (2 * n - p - 1) // 2 + q - p - 1
                assert wide_sign_check._pair_of(pid, n) == (p, q)
