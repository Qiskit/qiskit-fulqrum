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
from fulqrum.gpu.common.hash_utils import _popcount_u64, _splitmix64, _splitmix64_vec
from fulqrum.gpu.distributed.arm_pair_tables import between_parity_sign

MASK = (1 << 64) - 1
WIDTHS = (63, 64, 65, 100, 127, 128)
COUNT = 10**6
CHECKED = 2000


def random_words(rng, n, count):
    words = wide.key_words(n)
    out = rng.integers(0, 2**63, size=(count, words), dtype=np.uint64) << np.uint64(1)
    out |= rng.integers(0, 2, size=(count, words), dtype=np.uint64)
    top = n - 64 * (words - 1)
    if top < 64:
        out[:, -1] &= np.uint64((1 << top) - 1)
    return out


def as_int(row):
    return sum(int(word) << (64 * i) for i, word in enumerate(row))


def hash_reference(value, words):
    parts = [(value >> (64 * i)) & MASK for i in range(words)]
    h = parts[-1]
    for word in reversed(parts[:-1]):
        h = word ^ _splitmix64((h + int(wide.HASH_WORD_OFFSET)) & MASK)
    return _splitmix64(h)


def parity_reference(value, lo, hi):
    lo, hi = min(lo, hi), max(lo, hi)
    mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    return -1 if (value & mask).bit_count() & 1 else 1


def bit_reference(words, orbital):
    word = orbital // 64
    shift = (orbital % 64).astype(np.uint64)
    selected = np.take_along_axis(words, word[:, None], axis=1)[:, 0]
    return ((selected >> shift) & np.uint64(1)).astype(np.uint8)


def popcount_reference(words):
    bits = np.unpackbits(np.ascontiguousarray(words).view(np.uint8), axis=1)
    return bits.sum(axis=1).astype(np.int64)


def parity_reference_vec(words, lo, hi):
    low, high = np.minimum(lo, hi), np.maximum(lo, hi)
    total = np.zeros(words.shape[0], np.int64)
    for orbital in range(64 * words.shape[1]):
        inside = (orbital > low) & (orbital < high)
        inside |= (low == high) & (orbital == low)
        total += inside & (bit_reference(words, np.full_like(lo, orbital)) == 1)
    return np.where(total & 1, -1, 1).astype(np.int8)


def hash_reference_vec(words):
    h = words[:, -1]
    for word in range(words.shape[1] - 2, -1, -1):
        with np.errstate(over="ignore"):
            h = words[:, word] ^ _splitmix64_vec(h + wide.HASH_WORD_OFFSET)
    return _splitmix64_vec(h)


@pytest.mark.parametrize("n", WIDTHS)
def test_primitives_match_python_integers(n):
    rng = np.random.default_rng(n)
    words = random_words(rng, n, COUNT)
    lo = rng.integers(0, n, size=COUNT)
    hi = rng.integers(0, n, size=COUNT)
    orbital = rng.integers(0, n, size=COUNT)
    popcount = wide.popcount_w(words)
    bits = wide.bit_w(words, orbital)
    parity = wide.between_parity_w(words, lo, hi)
    hashes = wide.hash_w(words)
    assert popcount.shape == bits.shape == parity.shape == hashes.shape == (COUNT,)
    assert np.array_equal(popcount, popcount_reference(words))
    assert np.array_equal(bits, bit_reference(words, orbital))
    assert np.array_equal(parity, parity_reference_vec(words, lo, hi))
    assert np.array_equal(hashes, hash_reference_vec(words))
    for i in rng.choice(COUNT, size=CHECKED, replace=False):
        value = as_int(words[i])
        assert popcount[i] == value.bit_count()
        assert bits[i] == (value >> int(orbital[i])) & 1
        assert parity[i] == parity_reference(value, int(lo[i]), int(hi[i]))
        assert int(hashes[i]) == hash_reference(value, words.shape[1])


@pytest.mark.parametrize("n", WIDTHS)
def test_pack_strings_round_trips(n):
    rng = random.Random(100 + n)
    values = [rng.getrandbits(n) for _ in range(500)]
    values += [0, (1 << n) - 1, 1, 1 << (n - 1)]
    packed = wide.pack_strings([format(v, f"0{n}b") for v in values], n)
    if wide.key_words(n) == 1:
        assert packed.shape == (len(values),)
    else:
        assert packed.shape == (len(values), wide.key_words(n))
        assert packed.flags.c_contiguous
    assert wide.unpack_ints(packed) == values


def test_pack_strings_matches_the_single_word_packing():
    strings = [format(v, "064b") for v in (0, 1, 2**63, 2**64 - 2, 12345)]
    expected = np.array([int(s, 2) for s in strings], dtype=np.uint64)
    packed = wide.pack_strings(strings, 64)
    assert packed.dtype == np.uint64
    assert np.array_equal(packed, expected)


def test_pack_strings_rejects_wrong_lengths():
    with pytest.raises(ValueError, match="string 1 has 3 characters, expected 4"):
        wide.pack_strings(["0011", "011"], 4)


def test_parity_is_exhaustive_across_the_word_boundary():
    rng = np.random.default_rng(63)
    words = random_words(rng, 128, 16)
    lo, hi = np.meshgrid(np.arange(40, 90), np.arange(40, 90), indexing="ij")
    lo, hi = lo.ravel(), hi.ravel()
    for row in words:
        value = as_int(row)
        got = wide.between_parity_w(np.repeat(row[None, :], lo.size, axis=0), lo, hi)
        want = [parity_reference(value, int(a), int(b)) for a, b in zip(lo, hi)]
        assert np.array_equal(got, want)


def test_masks_cover_the_open_interval_across_words():
    for lo in range(0, 128, 7):
        for hi in range(lo + 1, 128, 5):
            mask = wide.between_mask_w(lo, hi, 2)
            assert as_int(mask) == ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
            assert as_int(wide.between_mask_w(hi, lo, 2)) == as_int(mask)


def test_single_word_parity_equals_the_existing_sign():
    rng = np.random.default_rng(1)
    bra = random_words(rng, 64, 64)[:, 0]
    p, q = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    p, q = p.ravel(), q.ravel()
    for value in bra:
        rows = np.full(p.size, value, dtype=np.uint64)
        assert np.array_equal(
            wide.between_parity_w(rows, p, q), between_parity_sign(rows, p, q)
        )


def test_single_word_hash_equals_splitmix64():
    rng = np.random.default_rng(2)
    keys = np.concatenate(
        [
            random_words(rng, 64, 100000)[:, 0],
            np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64),
        ]
    )
    assert np.array_equal(wide.hash_w(keys), _splitmix64_vec(keys))
    for key in keys[-4:]:
        assert int(wide.hash_w(np.array([key]))[0]) == _splitmix64(int(key))


def test_single_word_popcount_equals_the_existing_popcount():
    keys = random_words(np.random.default_rng(3), 64, 100000)[:, 0]
    assert np.array_equal(wide.popcount_w(keys), _popcount_u64(keys).astype(np.int64))


def test_two_word_hash_is_the_nested_form():
    keys = random_words(np.random.default_rng(4), 128, 1000)
    inner = _splitmix64_vec(keys[:, 1] + wide.HASH_WORD_OFFSET)
    assert np.array_equal(wide.hash_w(keys), _splitmix64_vec(keys[:, 0] ^ inner))


def test_equality_xor_and_the_empty_key():
    a = random_words(np.random.default_rng(5), 128, 50)
    b = a.copy()
    b[7, 1] ^= np.uint64(1 << 3)
    same = wide.eq_w(a, b)
    assert same.tolist() == [i != 7 for i in range(50)]
    assert np.array_equal(wide.xor_w(wide.xor_w(a, b), b), a)
    assert wide.EMPTY_W.tolist() == [2**64 - 1]
    assert wide.empty_w(2).tolist() == [2**64 - 1] * 2
    assert not wide.eq_w(a, np.broadcast_to(wide.empty_w(2), a.shape)).any()


def test_single_word_inputs_keep_their_shape():
    keys = np.array([3, 5, 6], dtype=np.uint64)
    assert wide.popcount_w(keys).tolist() == [2, 2, 2]
    assert wide.bit_w(keys, np.array([0, 0, 0])).tolist() == [1, 1, 0]
    assert wide.eq_w(keys, keys).tolist() == [True] * 3
