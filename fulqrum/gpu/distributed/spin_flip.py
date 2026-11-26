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

"""Opposite-direction corrections for alpha-beta doubles that move one orbital pair in both sectors.

The GPU tables keep one coefficient per (alpha pair, beta pair): the same-direction value s. A Hermitian
operator such as H + lambda S^2 can also have groups whose opposite-direction value o differs (the S^2
spin-flip terms move alpha j -> i together with beta i -> j). For such a group on orbital pair {i, j}, the
correction kernel adds (o - s) to each opposite-direction connection: for an output (a, b), every i in b \\ a
and j in a \\ b, with source (a - j + i, b - i + j).

Partners come from move tables built for this rank's output strings: for string r, its k-th occupied orbital
j and m-th empty orbital i, ``sign * (index of r - j + i, plus 1)`` when that string is in the subspace, else
0, where sign is the parity of the orbitals of r strictly between i and j.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpinFlip:
    """Device arrays and sizes of the correction (see ``launch_spin_flip``)."""

    alpha_moves: object
    beta_moves: object
    corrections: object
    num_orbitals: int
    alpha_electrons: int
    beta_electrons: int

    @property
    def nbytes(self) -> int:

        return int(
            self.alpha_moves.nbytes + self.beta_moves.nbytes + self.corrections.nbytes
        )


def move_table(rows, strings, num_orbitals: int) -> np.ndarray:
    """Signed partner indices for single moves of ``rows`` into ``strings`` (sorted, one word each).

    Returns an int32 array of shape (len(rows), electrons, num_orbitals - electrons).
    """
    rows = np.ascontiguousarray(rows, dtype=np.uint64)
    strings = np.ascontiguousarray(strings, dtype=np.uint64)
    if rows.ndim != 1 or strings.ndim != 1:
        raise ValueError("spin-flip move tables need single-word strings")
    if not 0 < num_orbitals <= 64:
        raise ValueError("spin-flip move tables support 1 to 64 orbitals")
    if len(strings) and np.any(strings[1:] <= strings[:-1]):
        raise ValueError("strings must be sorted and unique")
    if len(strings) + 1 > np.iinfo(np.int32).max:
        raise ValueError("too many strings for int32 move tables")
    counts = np.bitwise_count(rows) if len(rows) else np.zeros(1, dtype=np.uint8)
    electrons = int(counts[0])
    if np.any(counts != electrons):
        raise ValueError("spin-flip move tables need one electron count per sector")
    empty = num_orbitals - electrons
    table = np.zeros((len(rows), electrons, empty), dtype=np.int32)
    if not len(rows) or not electrons or not empty:
        return table
    one = np.uint64(1)
    for j in range(num_orbitals):
        bit_j = one << np.uint64(j)
        below_j = np.uint64((1 << j) - 1)
        for i in range(num_orbitals):
            if i == j:
                continue
            bit_i = one << np.uint64(i)
            sel = np.nonzero((rows & bit_j != 0) & (rows & bit_i == 0))[0]
            if not len(sel):
                continue
            r = rows[sel]
            k = np.bitwise_count(r & below_j).astype(np.int64)
            m = np.bitwise_count(~r & np.uint64((1 << i) - 1)).astype(np.int64)
            target = r ^ bit_j ^ bit_i
            index = np.searchsorted(strings, target)
            found = index < len(strings)
            found[found] = strings[index[found]] == target[found]
            lo, hi = min(i, j), max(i, j)
            between = np.uint64(((1 << hi) - 1) & ~((1 << (lo + 1)) - 1))
            sign = np.where(np.bitwise_count(r & between) & 1, -1, 1).astype(np.int64)
            value = sign * (index.astype(np.int64) + 1)
            table[sel[found], k[found], m[found]] = value[found].astype(np.int32)
    return table


def correction_table(corrections: dict, num_orbitals: int) -> np.ndarray:
    """The corrections {(p, q): o - s} in the GPU pair order (``fq::pair_index::gpu_pair``)."""
    table = np.zeros(num_orbitals * (num_orbitals - 1) // 2, dtype=np.float64)
    for (p, q), value in corrections.items():
        low, high = min(p, q), max(p, q)
        if not 0 <= low < high < num_orbitals:
            raise ValueError(f"orbital pair {(p, q)} outside {num_orbitals} orbitals")
        table[low * (2 * num_orbitals - low - 1) // 2 + (high - low - 1)] = value
    return table


def build_spin_flip(
    corrections: dict,
    alpha_strings,
    beta_strings,
    alpha_rows: tuple,
    beta_rows: tuple,
    num_alpha: int,
    num_beta: int,
    to_device,
):
    """A ``SpinFlip`` for this rank's output tile, or None when there is nothing to correct.

    ``alpha_rows`` and ``beta_rows`` are this rank's (begin, end) ranges of the sorted string arrays.
    """
    if not corrections:
        return None
    if num_alpha != num_beta:
        raise ValueError(
            "direction-dependent alpha-beta groups need the same number of alpha and beta orbitals"
        )
    alpha_strings = np.asarray(alpha_strings)
    beta_strings = np.asarray(beta_strings)
    if alpha_strings.ndim != 1 or beta_strings.ndim != 1:
        raise ValueError(
            "direction-dependent alpha-beta groups are supported for at most 64 orbitals per sector"
        )
    alpha = move_table(
        alpha_strings[alpha_rows[0] : alpha_rows[1]], alpha_strings, num_alpha
    )
    beta = move_table(beta_strings[beta_rows[0] : beta_rows[1]], beta_strings, num_beta)
    return SpinFlip(
        alpha_moves=to_device(alpha, np.int32),
        beta_moves=to_device(beta, np.int32),
        corrections=to_device(correction_table(corrections, num_alpha), np.float64),
        num_orbitals=int(num_alpha),
        alpha_electrons=int(alpha.shape[1]) if alpha.ndim == 3 else 0,
        beta_electrons=int(beta.shape[1]) if beta.ndim == 3 else 0,
    )
