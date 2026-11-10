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



from __future__ import annotations

import numpy as np

from ...exceptions import FulqrumError
from ..distributed.rank_agreement import raise_on_every_rank
from .hash_utils import _splitmix64_vec

MAX_KEY_WORDS = 1
WORD_BITS = 64
WORD_MASK = (1 << WORD_BITS) - 1
EMPTY_WORD = np.uint64(WORD_MASK)
HASH_WORD_OFFSET = np.uint64(0x9E3779B97F4A7C15)


def key_words(num_orbitals):


    return (int(num_orbitals) + 63) // 64


def width_error(widths, electrons=None, max_words=MAX_KEY_WORDS):


    limit = WORD_BITS * int(max_words)
    for i, width in enumerate(widths):
        if int(width) > limit:
            return (
                f"sector {i} has {int(width)} orbitals; "
                f"this build supports at most {limit} per sector"
            )
    for i, (width, count) in enumerate(zip(widths, electrons or ())):
        width = int(width)
        full = count is not None and int(count) >= width
        if full and width > 0 and width % WORD_BITS == 0:
            return (
                f"sector {i} is fully occupied with {width} orbitals; "
                f"a full sector of a multiple of {WORD_BITS} orbitals is not supported"
            )
    return None


def check_width(comm, widths, where, electrons=None, max_words=MAX_KEY_WORDS):


    raise_on_every_rank(
        comm, width_error(widths, electrons, max_words), FulqrumError, where
    )


def max_electrons(strings):


    if isinstance(strings, np.ndarray) and strings.dtype.kind in "iu":
        words = np.ascontiguousarray(strings, dtype=np.uint64)
        if words.size == 0:
            return None
        if words.ndim == 1:
            words = words.reshape(-1, 1)
        return int(popcount_w(words).max())
    best = None
    for value in strings:
        count = value.count("1") if isinstance(value, str) else int(value).bit_count()
        best = count if best is None else max(best, count)
    return best


def empty_w(words):


    return np.full(int(words), EMPTY_WORD, dtype=np.uint64)


EMPTY_W = empty_w(1)


def _xp(array):

    if type(array).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def _as_words(strings):

    words = _xp(strings).asarray(strings, dtype=np.uint64)
    if words.ndim == 1:
        return words.reshape(-1, 1)
    if words.ndim != 2:
        raise ValueError(f"strings must have shape (N,) or (N, W), got {words.shape}")
    return words


def pack_values(values, n):


    n = int(n)
    if n <= 0:
        raise ValueError(f"the orbital count must be positive, got {n}")
    width = key_words(n)
    values = [int(v) for v in values]
    for row, value in enumerate(values):
        if value < 0 or value >> n:
            raise ValueError(f"string {row} has a bit at or above orbital {n}")
    if width == 1:
        return np.array(values, dtype=np.uint64)
    out = np.empty((len(values), width), dtype=np.uint64)
    for word in range(width):
        shift = WORD_BITS * word
        out[:, word] = np.array([(v >> shift) & WORD_MASK for v in values], np.uint64)
    return out


def pack_strings(strs, n):


    n = int(n)
    for row, text in enumerate(strs):
        if len(text) != n:
            raise ValueError(
                f"string {row} has {len(text)} characters, expected {n}"
            )
    return pack_values([int(text, 2) for text in strs], n)


def sort_strings(strings):


    words = np.asarray(strings, dtype=np.uint64)
    if words.ndim == 1:
        return np.sort(words)
    if words.ndim != 2:
        raise ValueError(f"strings must have shape (N,) or (N, W), got {words.shape}")
    order = np.lexsort(words.T)
    return np.ascontiguousarray(words[order])


def with_words(strings, words):


    strings = np.asarray(strings, dtype=np.uint64)
    if words == 1 or (strings.ndim == 2 and strings.shape[1] == words):
        return strings
    if strings.ndim == 2 and strings.shape[1] != 1:
        raise ValueError(
            f"strings have {strings.shape[1]} words; the sector uses {words}"
        )
    out = np.zeros((strings.shape[0], int(words)), dtype=np.uint64)
    out[:, 0] = strings.reshape(-1)
    return out


def unpack_ints(strings):


    words = _as_words(strings)
    out = []
    for row in words:
        value = 0
        for word in range(words.shape[1] - 1, -1, -1):
            value = (value << WORD_BITS) | int(row[word])
        out.append(value)
    return out


def _popcount_words(x):

    xp = _xp(x)
    if hasattr(xp, "bitwise_count") and xp is not np:
        return xp.bitwise_count(x.astype(np.uint64)).astype(np.uint64)
    x = x.astype(np.uint64, copy=True)
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def popcount_w(strings):


    return _popcount_words(_as_words(strings)).sum(axis=1).astype(np.int64)


def bit_w(strings, orbital):


    words = _as_words(strings)
    xp = _xp(words)
    orbital = xp.asarray(orbital, dtype=np.int64)
    word = orbital // WORD_BITS
    shift = (orbital % WORD_BITS).astype(np.uint64)
    rows = xp.arange(words.shape[0])
    selected = words[rows, word] if word.ndim else words[:, int(word)]
    return ((selected >> shift) & np.uint64(1)).astype(np.uint8)


def _below(count, xp=np):

    count = xp.asarray(count, dtype=np.int64)
    capped = xp.minimum(count, WORD_BITS - 1).astype(np.uint64)
    return xp.where(
        count >= WORD_BITS,
        np.uint64(WORD_MASK),
        (np.uint64(1) << capped) - np.uint64(1),
    )


def _between_masks(lo, hi, words, xp=np):

    lo = xp.asarray(lo, dtype=np.int64)
    hi = xp.asarray(hi, dtype=np.int64)
    low, high = xp.minimum(lo, hi), xp.maximum(lo, hi)
    base = xp.arange(words, dtype=np.int64) * WORD_BITS
    upper = xp.clip(high[..., None] - base, 0, WORD_BITS)
    lower = xp.clip(low[..., None] + 1 - base, 0, WORD_BITS)
    return _below(upper, xp) ^ _below(lower, xp)


def between_mask_w(lo, hi, words):


    return _between_masks(int(lo), int(hi), int(words)).reshape(int(words))


def between_parity_w(strings, lo, hi):


    words = _as_words(strings)
    xp = _xp(words)
    masks = _between_masks(lo, hi, words.shape[1], xp)
    odd = (_popcount_words(words & masks).sum(axis=-1) & 1).astype(bool)
    return xp.where(odd, -1, 1).astype(np.int8)


def below_parity_w(strings, orbital):


    return between_parity_w(strings, -1, orbital)


def single_bit_index_w(strings):


    words = _as_words(strings)
    rows = np.arange(words.shape[0])
    word = np.argmax(words != 0, axis=1)
    selected = words[rows, word]
    below = _popcount_words(selected - np.uint64(1))
    return (word * WORD_BITS + below).astype(np.int64)


def orbital_bits_w(orbitals, words):


    orbitals = np.asarray(orbitals, dtype=np.int64).reshape(-1)
    out = np.zeros((orbitals.size, int(words)), dtype=np.uint64)
    out[np.arange(orbitals.size), orbitals // WORD_BITS] = np.uint64(1) << (
        orbitals % WORD_BITS
    ).astype(np.uint64)
    return out


def unique_w(strings, xp=np, return_counts=False):


    if strings.ndim == 1:
        return xp.unique(strings, return_inverse=True, return_counts=return_counts)
    n = strings.shape[0]
    if n == 0:
        empty = xp.zeros(0, dtype=xp.int64)
        out = (strings[:0], empty)
        return out + (empty,) if return_counts else out
    order = xp.lexsort(strings.T)
    ordered = strings[order]
    first = xp.ones(n, dtype=bool)
    first[1:] = xp.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = xp.empty(n, dtype=xp.int64)
    inverse[order] = xp.cumsum(first) - 1
    out = (ordered[first], inverse)
    if not return_counts:
        return out
    starts = xp.flatnonzero(first)
    ends = xp.concatenate((starts[1:], xp.asarray([n], dtype=starts.dtype)))
    return out + (ends - starts,)


def xor_w(a, b):


    a = np.asarray(a, dtype=np.uint64)
    return np.bitwise_xor(a, np.asarray(b, dtype=np.uint64))


def eq_w(a, b):


    a = _as_words(a)
    b = _as_words(b)
    return np.all(a == b, axis=1)


def hash_w(strings):


    words = _as_words(strings)
    h = words[:, words.shape[1] - 1]
    for word in range(words.shape[1] - 2, -1, -1):
        with np.errstate(over="ignore"):
            h = words[:, word] ^ _splitmix64_vec(h + HASH_WORD_OFFSET)
    return _splitmix64_vec(h)
