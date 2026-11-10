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

from fulqrum.gpu.common.wide import pack_values
from fulqrum.gpu.distributed import construction, tile_discovery
from fulqrum.gpu.distributed.grid import _signatures

EMPTY_MASKS = {"alpha2": [], "alpha4": [], "beta2": [], "beta4": []}


def window_strings(rng, width, window, electrons, count):
    """Return sorted distinct strings with electrons inside window plus one fixed spectator."""
    out = set()
    while len(out) < count:
        occupied = set(rng.sample(window, electrons))
        out.add(sum(1 << o for o in occupied) | 1)
    return sorted(out)


def reference_csr(bra, src, popcount):
    rows = [[] for _ in bra]
    for r, b in enumerate(bra):
        for s, v in enumerate(src):
            x = b ^ v
            if x.bit_count() == popcount:
                rows[r].append((s, [i for i in range(x.bit_length()) if (x >> i) & 1]))
    ptr = np.zeros(len(bra) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(row) for row in rows])
    cols = [s for row in rows for s, _ in row]
    positions = np.zeros((4, len(cols)), dtype=np.uint8)
    flat = [p for row in rows for _s, p in row]
    for k, p in enumerate(flat):
        positions[: len(p), k] = p
    return ptr, np.array(cols, dtype=np.int32), positions


def assert_csr_equal(got, want, popcount):
    ptr, cols, positions = want
    np.testing.assert_array_equal(got[0], ptr)
    nnz = int(ptr[-1])
    assert nnz > 20
    np.testing.assert_array_equal(np.asarray(got[1])[:nnz], cols)
    for k in range(4):
        expected = positions[k] if k < popcount else np.zeros(nnz, np.uint8)
        np.testing.assert_array_equal(np.asarray(got[2 + k])[:nnz], expected)


def problem(width, seed, count=150):
    rng = random.Random(seed)
    window = list(range(width - 14, width)) if width < 80 else list(range(56, 70))
    strings = window_strings(rng, width, window, 4, 2 * count)
    rng.shuffle(strings)
    bra = sorted(strings[:count])
    src = sorted(strings[count // 2 :])
    return bra, src, pack_values(bra, width), pack_values(src, width)


@pytest.fixture(params=["native", "python"])
def build(request, monkeypatch):
    if request.param == "native" and not tile_discovery._native_tile_build_enabled():
        pytest.skip("the native tile build is not available")
    if request.param == "python":
        monkeypatch.setattr(tile_discovery, "_native_tile_build_enabled", lambda: False)
    return request.param


@pytest.mark.parametrize("width", [70, 100, 128])
@pytest.mark.parametrize("popcount", [2, 4])
def test_two_word_dense_csr_equals_python_integers(build, width, popcount):
    bra, src, bra_words, src_words = problem(width, width + popcount)
    assert bra_words.shape[1] == 2
    high = sum(1 for v in bra if v >> 64)
    assert high >= 0.3 * len(bra)
    want = reference_csr(bra, src, popcount)
    got = tile_discovery._build_tile_flip_csr_dense(bra_words, src_words, popcount)
    assert_csr_equal(got, want, popcount)
    straddle = [
        k
        for k in range(int(want[0][-1]))
        if want[2][0, k] < 64 <= want[2][popcount - 1, k]
    ]
    assert straddle


@pytest.mark.parametrize("width", [70, 128])
def test_two_word_has_popcount(build, width):
    bra, src, bra_words, src_words = problem(width, 7)
    for popcount in (2, 4):
        want = any((b ^ v).bit_count() == popcount for b in bra for v in src)
        assert (
            tile_discovery._tile_has_popcount(bra_words, src_words, (popcount,)) == want
        )
    far = pack_values([sum(1 << o for o in range(width - 10, width))], width)
    near = pack_values([sum(1 << o for o in range(10))], width)
    assert not tile_discovery._tile_has_popcount(far, near, (2, 4))


def test_native_and_python_csr_agree(monkeypatch):
    if not tile_discovery._native_tile_build_enabled():
        pytest.skip("the native tile build is not available")
    _bra, _src, bra_words, src_words = problem(100, 3)
    native = tile_discovery._build_tile_flip_csr_dense(
        bra_words, src_words, 4, threads=4
    )
    monkeypatch.setattr(tile_discovery, "_native_tile_build_enabled", lambda: False)
    python = tile_discovery._build_tile_flip_csr_dense(bra_words, src_words, 4)
    nnz = int(native[0][-1])
    np.testing.assert_array_equal(native[0], python[0])
    for a, b in zip(native[1:], python[1:]):
        np.testing.assert_array_equal(np.asarray(a)[:nnz], np.asarray(b)[:nnz])


def pad(value, k):
    return (value & 0b111) | ((value >> 3) << (3 + k))


@pytest.mark.parametrize("width, k", [(40, 40), (64, 64)])
@pytest.mark.parametrize("popcount", [2, 4])
def test_padding_shifts_the_positions(build, width, k, popcount):
    rng = random.Random(width + k + popcount)
    window = list(range(width - 12, width)) + [1, 2]
    strings = sorted(set(window_strings(rng, width, window, 4, 300)))
    bra, src = strings[:200], strings[100:]
    narrow = tile_discovery._build_tile_flip_csr_dense(
        pack_values(bra, width), pack_values(src, width), popcount
    )
    wide = tile_discovery._build_tile_flip_csr_dense(
        pack_values([pad(v, k) for v in bra], width + k),
        pack_values([pad(v, k) for v in src], width + k),
        popcount,
    )
    nnz = int(narrow[0][-1])
    assert nnz > 20
    np.testing.assert_array_equal(wide[0], narrow[0])
    np.testing.assert_array_equal(
        np.asarray(wide[1])[:nnz], np.asarray(narrow[1])[:nnz]
    )
    for index in range(popcount):
        positions = np.asarray(narrow[2 + index])[:nnz].astype(np.int64)
        shifted = np.where(positions >= 3, positions + k, positions)
        np.testing.assert_array_equal(np.asarray(wide[2 + index])[:nnz], shifted)


def test_source_needed_with_two_words(build):
    width = 100
    bra, src, bra_words, src_words = problem(width, 11)
    beta_bra, beta_src, _beta_words, _beta_src_words = problem(width, 12)

    def needed(a_bra, b_bra, a_src, b_src):
        return tile_discovery._tile_source_needed(
            a_bra, b_bra, a_src, b_src, EMPTY_MASKS
        )

    def reference(a_bra, b_bra, a_src, b_src):
        a0 = bool(set(a_bra) & set(a_src))
        b0 = bool(set(b_bra) & set(b_src))

        def has(left, right, pc):
            return any((x ^ y).bit_count() == pc for x in left for y in right)

        a2, a4 = has(a_bra, a_src, 2), has(a_bra, a_src, 4)
        b2, b4 = has(b_bra, b_src, 2), has(b_bra, b_src, 4)
        return (a0 and b0) or (b0 and (a2 or a4)) or (a0 and (b2 or b4)) or (a2 and b2)

    disjoint = sorted(set(src) - set(bra))
    cases = [
        (bra, beta_bra, src, beta_src),
        (bra, beta_bra, disjoint, beta_bra),
        (bra[:1], beta_bra[:1], disjoint[:3], beta_src[-3:]),
    ]
    for a_bra, b_bra, a_src, b_src in cases:
        got = needed(
            pack_values(a_bra, width),
            pack_values(b_bra, width),
            pack_values(a_src, width),
            pack_values(b_src, width),
        )
        assert got == reference(a_bra, b_bra, a_src, b_src)
    assert tile_discovery._contains_rows(bra_words, src_words)
    assert not tile_discovery._contains_rows(bra_words, pack_values(disjoint, width))
    same_low = [v ^ (1 << 90) for v in bra if not (v >> 90) & 1]
    assert not tile_discovery._contains_rows(pack_values(same_low, width), bra_words)


def test_connection_tables_with_two_words_use_dense(monkeypatch):
    built = []
    original = tile_discovery._build_hash_table_simple

    def spy(values):
        built.append(np.shape(values))
        return original(values)

    monkeypatch.setattr(tile_discovery, "_build_hash_table_simple", spy)
    monkeypatch.setattr(construction, "_build_hash_table_simple", spy)
    _bra, _src, bra_words, src_words = problem(70, 21)
    tables = tile_discovery._build_tile_connection_tables(
        bra_words, bra_words, src_words, src_words, EMPTY_MASKS
    )
    axis = construction._build_axis_connection_tables(bra_words, src_words, [], [])
    assert built == []
    for popcount, key in ((2, "alpha2"), (4, "alpha4"), (2, "beta2"), (4, "beta4")):
        dense = tile_discovery._build_tile_flip_csr_dense(
            bra_words, src_words, popcount
        )
        for got, want in zip(tables[key], dense):
            np.testing.assert_array_equal(got, want)
    for popcount, key in ((2, "csr2"), (4, "csr4")):
        dense = tile_discovery._build_tile_flip_csr_dense(
            bra_words, src_words, popcount
        )
        for got, want in zip(axis[key], dense):
            np.testing.assert_array_equal(got, want)
    assert int(axis["csr2"][0][-1]) > 0


def test_single_word_axis_tables_without_masks_stay_empty():
    values = np.array([0b0011, 0b0101, 0b0110], dtype=np.uint64)
    axis = construction._build_axis_connection_tables(values, values, [], [])
    assert int(axis["csr2"][0][-1]) == 0 and int(axis["csr4"][0][-1]) == 0


def test_masks_are_dropped_for_two_word_sectors():
    masks = {
        "alpha2": [(np.uint64(3), (np.uint8(0), np.uint8(1)))],
        "alpha4": [],
        "beta2": [(np.uint64(5), (np.uint8(0), np.uint8(2)))],
        "beta4": [],
    }
    kept = tile_discovery.masks_for_words(masks, (1, 2))
    assert kept["alpha2"] == masks["alpha2"] and kept["beta2"] == []
    assert tile_discovery.masks_for_words(masks, (1, 1)) == masks
    tile_discovery.check_masks_fit(kept, (1, 2))
    with pytest.raises(ValueError, match="beta2 flip masks are single-word"):
        tile_discovery.check_masks_fit(masks, (1, 2))


def test_mask_paths_refuse_two_word_strings():
    words = pack_values([3, 5, 6], 70)
    masks = [(np.uint64(3), (np.uint8(0), np.uint8(1)))]
    with pytest.raises(ValueError, match="multi-word strings use dense discovery"):
        tile_discovery._build_tile_flip_csr_from_masks(words, words, masks)
    with pytest.raises(ValueError, match="multi-word strings use dense discovery"):
        tile_discovery._tile_has_mask_connection(words, None, None, 0, masks)

    from types import SimpleNamespace  # noqa: PLC0415

    def scan(flip_masks):
        return SimpleNamespace(
            flip_masks=flip_masks,
            alpha_strings=words,
            beta_strings=words,
            bra=lambda: (words, words),
        )

    with pytest.raises(ValueError, match="multi-word strings use dense discovery"):
        _signatures(scan({"alpha2": masks, "alpha4": [], "beta2": [], "beta4": []}))
    assert _signatures(scan(EMPTY_MASKS)) == (0, 0)


def test_single_word_selection_is_unchanged():
    rng = random.Random(4)
    strings = sorted(set(window_strings(rng, 20, list(range(1, 20)), 4, 400)))
    values = np.array(strings, dtype=np.uint64)
    masks = tile_discovery._group_flip_masks(
        np.array([[1, 2, 0, 0], [3, 5, 0, 0], [1, 2, 3, 4]], dtype=np.int64),
        np.array([2, 2, 4], dtype=np.int32),
        20,
    )
    assert masks["alpha2"] and len(masks["alpha2"]) * 8 < len(values)
    for popcount, key in ((2, "alpha2"), (4, "alpha4")):
        hybrid = tile_discovery._build_tile_flip_csr_hybrid(
            values, values, masks[key], popcount
        )
        expected = tile_discovery._build_tile_flip_csr_from_masks(
            values, values, masks[key]
        )
        for got, want in zip(hybrid, expected):
            np.testing.assert_array_equal(got, want)
    dense = tile_discovery._build_tile_flip_csr_hybrid(values, values, [], 2)
    for got, want in zip(
        dense, tile_discovery._build_tile_flip_csr_dense(values, values, 2)
    ):
        np.testing.assert_array_equal(got, want)
    a0 = tile_discovery._contains_rows(values[:10], values[5:])
    assert a0 == (np.intersect1d(values[:10], values[5:], assume_unique=True).size > 0)
