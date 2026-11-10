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

import logging

import numpy as np

from ..common.wide import unique_w, unpack_ints, width_error
from ..distributed.rank_agreement import run_on_every_rank

logger = logging.getLogger(__name__)

ENGINES = ("auto", "general", "tile")


def halves(keys, words):

    keys = np.ascontiguousarray(keys, dtype=np.uint64)
    alpha, beta = keys[:, :words], keys[:, words:]
    if words == 1:
        return np.ascontiguousarray(alpha[:, 0]), np.ascontiguousarray(beta[:, 0])
    return np.ascontiguousarray(alpha), np.ascontiguousarray(beta)


def _union(comm, strings):

    gathered = comm.allgather(unique_w(strings)[0])
    joined = np.concatenate(gathered) if gathered else strings[:0]
    return unique_w(joined)[0]


def product_halves(comm, block, words):

    alpha, beta = halves(block.keys, words)
    alpha, beta = _union(comm, alpha), _union(comm, beta)
    if len(alpha) * len(beta) != int(block.num_rows):
        return None
    return alpha, beta


def beta_cuts(num_alpha_strings, num_beta_strings, size):

    from ..distributed.dist_matvec import _split_ranges_even

    ranges = _split_ranges_even(num_beta_strings, size)
    bounds = [start for start, _stop in ranges] + [ranges[-1][1]]
    return np.asarray(bounds, dtype=np.int64) * int(num_alpha_strings)


def bit_strings(strings, num_orbitals):

    return [format(value, f"0{num_orbitals}b") for value in unpack_ints(strings)]


def prepare(comm, operator, block, words, num_orbitals):

    import fulqrum as fq

    from ..distributed.dist_matvec_tile_resident import (
        TILE_MAX_KEY_WORDS,
        _operator_tables,
        tile_operator_error,
    )

    found = product_halves(comm, block, words)
    if found is None:
        return None, "the keys are not a complete product of alpha and beta strings"
    alpha, beta = found
    if len(beta) < comm.Get_size():
        return None, f"{len(beta)} beta strings cannot fill {comm.Get_size()} ranks"
    error = width_error((num_orbitals, num_orbitals), max_words=TILE_MAX_KEY_WORDS)
    if error is not None:
        return None, error

    def build():

        subspace = fq.Subspace(
            [bit_strings(alpha, num_orbitals), bit_strings(beta, num_orbitals)]
        )
        hsub = fq.SubspaceHamiltonian(operator, subspace)
        if hsub.off_H.type != 2 or hsub.diag_H.type != 2:
            return None, "the operator is not type 2"
        tables = _operator_tables(hsub.spmv)
        reason = tile_operator_error(hsub.spmv, tables, num_orbitals, (words, words))
        return (hsub, tables), reason

    prepared, reason = run_on_every_rank(comm, "tile engine checks", build)
    reasons = [r for r in comm.allgather(reason) if r is not None]
    if reasons:
        return None, reasons[0]
    hsub, tables = prepared
    hsub._tile_operator_tables = tables
    return (hsub, alpha, beta), None
