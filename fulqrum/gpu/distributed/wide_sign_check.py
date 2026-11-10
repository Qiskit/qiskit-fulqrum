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

from ..common.wide import unpack_ints

SAMPLE_EDGES = 10_000


def _int_sign(value: int, p: int, q: int) -> int:

    lo, hi = min(p, q), max(p, q)
    between = value & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
    return -1 if between.bit_count() & 1 else 1


def _sample(count: int, size: int, seed: int) -> np.ndarray:

    if count <= size:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, size, replace=False))


def _edge_rows(ptr) -> np.ndarray:

    ptr = np.asarray(ptr, dtype=np.int64)
    return np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))


def double_sign_error(
    ptr, positions, values, bra, label, size=SAMPLE_EDGES, seed=0
) -> str | None:


    ptr = np.asarray(ptr, dtype=np.int64)
    nnz = int(ptr[-1])
    if nnz == 0:
        return None
    values = np.asarray(values, dtype=np.float64)[:nnz]
    picked = _sample(nnz, size, seed)
    rows = _edge_rows(ptr)[picked]
    flips = np.stack(
        [np.asarray(p, dtype=np.int64)[:nnz][picked] for p in positions], axis=1
    )
    strings = unpack_ints(np.asarray(bra)[rows])
    seen = {}
    for k, (value, string, flip) in enumerate(zip(values[picked], strings, flips)):
        if value == 0.0 or not np.isfinite(value):
            continue
        a, b, c, d = sorted(int(x) for x in flip)
        occupied = tuple((string >> o) & 1 for o in (a, b, c, d))
        spectators = string & ~((1 << a) | (1 << b) | (1 << c) | (1 << d))
        sign = _int_sign(spectators, a, b) * _int_sign(spectators, c, d)
        key = (a, b, c, d, occupied)
        reduced = value * sign
        known = seen.setdefault(key, (reduced, int(picked[k])))
        if abs(known[0] - reduced) > 1e-12 * max(abs(known[0]), abs(reduced)):
            return (
                f"{label}: the double connections at edges {known[1]} and "
                f"{int(picked[k])} (orbitals {a}, {b}, {c}, {d}) disagree once "
                "their occupation signs are removed"
            )
    return None


def beta_word_sign_error(
    graph, bra, orbitals, label, size=SAMPLE_EDGES, seed=0
) -> str | None:


    words = np.asarray(graph["words"], dtype=np.uint64)
    if not len(words):
        return None
    picked = _sample(len(words), size, seed)
    rows = np.asarray(graph["rows"], dtype=np.int64)[picked]
    strings = unpack_ints(np.asarray(bra)[rows])
    for k, (word, string) in enumerate(zip(words[picked].tolist(), strings)):
        stored = -1 if (word >> 48) & 0xFFFF == 0xFFFF else 1
        p, q = _pair_of(int((word >> 32) & 0xFFFF), int(orbitals))
        if stored != _int_sign(string, p, q):
            return (
                f"{label}: the sign of mixed-spin edge {int(picked[k])} "
                f"(orbitals {p}, {q}) differs from its occupation string"
            )
    return None


def _pair_of(pair_id: int, orbitals: int) -> tuple[int, int]:

    p, offset = 0, 0
    while offset + orbitals - p - 1 <= pair_id:
        offset += orbitals - p - 1
        p += 1
    return p, p + 1 + pair_id - offset


def check_enabled(setting, key_words) -> bool:

    if setting == "auto":
        return tuple(key_words) != (1, 1)
    return bool(setting)
