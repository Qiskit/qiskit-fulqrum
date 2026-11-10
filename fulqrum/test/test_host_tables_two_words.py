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


import numpy as np
import pytest

from fulqrum.gpu.common.wide import sort_strings
from fulqrum.test.test_cartesian_two_words import padded_hamiltonians

tile_tables = pytest.importorskip("fulqrum.gpu.distributed.tile_tables")

from fulqrum.gpu.distributed.tables import (
    aabb_coefficients,
    chunked_pair_table,
    pair_table,
)

WIDTH = 70
ACTIVE = list(range(58, 70))


@pytest.fixture(scope="module")
def pair():
    narrow, wide = padded_hamiltonians(WIDTH, ACTIVE, (4, 3), 30, 3)
    return narrow, wide


def gpu_strings(hsub, words):
    alpha, beta = hsub.spmv.subspace._alpha_beta_words(words)
    return sort_strings(alpha), sort_strings(beta)


def pair_map(m_small, m_wide):
    """Map each narrow pair id to its padded pair id."""
    out = []
    for p in range(m_small):
        for q in range(p + 1, m_small):
            a, b = ACTIVE[p], ACTIVE[q]
            out.append(a * (2 * m_wide - a - 1) // 2 + b - a - 1)
    return np.array(out)


def core_pair_map(m_small):
    """Map each narrow core pair id (high * (high - 1) / 2 + low) to its padded one."""
    out = np.zeros(m_small * (m_small - 1) // 2, dtype=np.int64)
    for q in range(m_small):
        for p in range(q):
            a, b = ACTIVE[p], ACTIVE[q]
            out[q * (q - 1) // 2 + p] = b * (b - 1) // 2 + a
    return out


def assert_close_to_scale(got, want):
    scale = np.abs(want).max()
    assert scale > 0
    assert np.abs(got - want).max() <= 1e-14 * scale


def test_two_word_inputs_pass_the_tile_check(pair):
    narrow, wide = pair
    alpha, beta = gpu_strings(narrow, 1)
    assert tile_tables.tile_input_error(narrow.spmv, len(ACTIVE), alpha, beta) is None
    alpha, beta = gpu_strings(wide, 2)
    assert alpha.shape[1] == 2
    assert tile_tables.tile_input_error(wide.spmv, WIDTH, alpha, beta) is None


@pytest.mark.parametrize("change", ["high", "low", "shape", "count"])
def test_tile_check_compares_every_word(pair, change):
    _, wide = pair
    alpha, beta = gpu_strings(wide, 2)
    alpha = alpha.copy()
    if change == "high":
        alpha[3, 1] ^= np.uint64(1)
    elif change == "low":
        alpha[3, 0] ^= np.uint64(1 << 5)
    elif change == "shape":
        alpha = alpha[:, :1].copy()
    else:
        alpha = alpha[:-1]
    reason = tile_tables.tile_input_error(wide.spmv, WIDTH, alpha, beta)
    expected = "differ from the GPU strings" if change in ("high", "low") else "strings"
    assert reason is not None and expected in reason


def test_padded_tiles_equal_the_single_word_tiles(pair):
    narrow, wide = pair
    n_a = narrow.spmv.subspace.to_alpha_beta_arrays()[0].size
    n_b = narrow.spmv.subspace.to_alpha_beta_arrays()[1].size
    m = len(ACTIVE)
    for sector, count in ((0, n_a), (1, n_b)):
        bounds = [0, count // 2, count]
        small = tile_tables.sector_tiles(narrow.spmv, sector, 0, count, bounds)
        large = tile_tables.sector_tiles(wide.spmv, sector, 0, count, bounds)
        mapping = core_pair_map(m)
        for a, b in zip(small, large):
            for key in (
                "single_offsets",
                "columns",
                "own_signs",
                "jw_signs",
                "double_offsets",
                "double_columns",
            ):
                assert np.array_equal(a[key], b[key]), key
            pairs = a["directed_pairs"].astype(np.int64)
            assert np.array_equal(
                2 * mapping[pairs // 2] + pairs % 2,
                b["directed_pairs"].astype(np.int64),
            )
            for key in ("single_values", "double_values"):
                if a[key].size:
                    assert_close_to_scale(b[key], a[key])
                else:
                    assert not b[key].size


@pytest.mark.parametrize("chunk", [1, 4, 1000])
def test_padded_pair_tables_equal_the_single_word_tables(pair, chunk):
    narrow, wide = pair
    m = len(ACTIVE)
    mapping = pair_map(m, WIDTH)
    for sector in (0, 1):
        other = narrow.spmv.subspace.to_alpha_beta_arrays()[1 - sector].size
        small = pair_table(
            tile_tables.cross_table(narrow.spmv, sector, 0, other), m, 1e-12
        )
        large = chunked_pair_table(
            lambda a, b, s=sector: tile_tables.cross_table(wide.spmv, s, a, b),
            other,
            WIDTH,
            1e-12,
            chunk,
        )
        assert large.shape == (other, WIDTH * (WIDTH - 1) // 2)
        assert_close_to_scale(large[:, mapping], small)
        rest = np.setdiff1d(np.arange(large.shape[1]), mapping)
        assert not large[:, rest].any()


def test_chunked_host_pair_table_is_bitwise_the_whole_table(pair):
    narrow, _ = pair
    m = len(ACTIVE)
    for sector in (0, 1):
        other = narrow.spmv.subspace.to_alpha_beta_arrays()[1 - sector].size
        whole = pair_table(
            tile_tables.cross_table(narrow.spmv, sector, 0, other), m, 1e-12
        )
        for chunk in (1, 3, other, other + 2):
            got = chunked_pair_table(
                lambda a, b, s=sector: tile_tables.cross_table(narrow.spmv, s, a, b),
                other,
                m,
                1e-12,
                chunk,
            )
            assert got.tobytes() == whole.tobytes()


def test_padded_mixed_spin_coefficients_equal_the_single_word_ones(pair):
    narrow, wide = pair
    m = len(ACTIVE)
    small = aabb_coefficients(tile_tables.cross_sector_table(narrow.spmv), m, m)
    large = aabb_coefficients(tile_tables.cross_sector_table(wide.spmv), WIDTH, WIDTH)
    p_small, p_wide = m * (m - 1) // 2, WIDTH * (WIDTH - 1) // 2
    mapping = pair_map(m, WIDTH)
    small = small.reshape(p_small, p_small)
    large = large.reshape(p_wide, p_wide)
    assert_close_to_scale(large[np.ix_(mapping, mapping)], small)
    rest = np.setdiff1d(np.arange(p_wide), mapping)
    assert not large[rest].any() and not large[:, rest].any()
