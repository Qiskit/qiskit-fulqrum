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

from ..common.wide import unique_w, unpack_ints
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

    gathered = comm.allgather(strings)
    joined = np.concatenate(gathered) if gathered else strings[:0]
    return unique_w(joined)[0]


def product_halves(comm, block, words):

    alpha, beta = (unique_w(half)[0] for half in halves(block.keys, words))
    counts = comm.allgather((len(alpha), len(beta)))
    rows = int(block.num_rows)
    most_a, most_b = (max(c[k] for c in counts) for k in range(2))
    total_a, total_b = (sum(c[k] for c in counts) for k in range(2))
    if most_a * most_b > rows or total_a * total_b < rows:
        return None
    alpha, beta = _union(comm, alpha), _union(comm, beta)
    if len(alpha) * len(beta) != rows:
        return None
    return alpha, beta


def beta_cuts(num_alpha_strings, num_beta_strings, size):

    from ..distributed.dist_matvec import _split_ranges_even

    ranges = _split_ranges_even(num_beta_strings, size)
    bounds = [start for start, _stop in ranges] + [ranges[-1][1]]
    return np.asarray(bounds, dtype=np.int64) * int(num_alpha_strings)


def bit_strings(strings, num_orbitals):

    return [format(value, f"0{num_orbitals}b") for value in unpack_ints(strings)]


def _setting_error(config, wide):

    from ..distributed.arm_pair_tables import pair_table_options

    aa_pair, bb_pair = pair_table_options(config)
    if wide and not (aa_pair and bb_pair):
        return "FQ_AA_PAIR_TABLE=0 or FQ_BB_PAIR_TABLE=0 cannot run two-word strings"
    if wide and config.value("FQ_ARM_TABLE_SOURCE") == "kernel":
        return "FQ_ARM_TABLE_SOURCE=kernel cannot run two-word strings"
    if wide and "FQ_DIAG_RECOMPUTE" in config and config.value("FQ_DIAG_RECOMPUTE"):
        return "FQ_DIAG_RECOMPUTE=1 cannot run two-word strings"
    return None


def _arm_error(config, spmv, num_orbitals, alpha, beta, wide, factored):

    from ..distributed.arm_pair_tables import pair_table_options
    from ..distributed.dist_matvec_tile_resident import host_arm_reason

    source = config.value("FQ_ARM_TABLE_SOURCE")
    if source == "kernel":
        return None
    if alpha.ndim == 2 and num_orbitals <= 64:
        alpha, beta = alpha[:, 0].copy(), beta[:, 0].copy()
    reason = host_arm_reason(
        spmv, num_orbitals, alpha, beta, factored, *pair_table_options(config)
    )
    if reason is not None and (wide or source == "host"):
        return reason
    return None


def prepare(comm, operator, block, words, num_orbitals):

    import fulqrum as fq

    from ..config import GpuConfig
    from ..distributed.dist_matvec_tile_resident import (
        _operator_tables,
        tile_operator_error,
    )

    config = GpuConfig.from_environ()
    wide = words > 1
    reason = _setting_error(config, wide)
    if reason is not None:
        return None, reason
    found = product_halves(comm, block, words)
    if found is None:
        return None, "the keys are not a complete product of alpha and beta strings"
    alpha, beta = found
    if len(beta) < comm.Get_size():
        return None, f"{len(beta)} beta strings cannot fill {comm.Get_size()} ranks"

    def build():

        subspace = fq.Subspace(
            [bit_strings(alpha, num_orbitals), bit_strings(beta, num_orbitals)]
        )
        hsub = fq.SubspaceHamiltonian(operator, subspace)
        if hsub.off_H.type != 2 or hsub.diag_H.type != 2:
            return None, "the operator is not type 2"
        tables = _operator_tables(hsub.spmv)
        reason, grp4_direct = tile_operator_error(
            hsub.spmv, tables, num_orbitals, (words, words)
        )
        if reason is None:
            factored = int(tables["num_groups"]) > 0
            reason = _arm_error(
                config, hsub.spmv, num_orbitals, alpha, beta, wide, factored
            )
        return (hsub, tables, grp4_direct), reason

    prepared, reason = run_on_every_rank(comm, "tile engine checks", build)
    reasons = [r for r in comm.allgather(reason) if r is not None]
    if reasons:
        return None, reasons[0]
    hsub, tables, grp4_direct = prepared
    hsub._tile_operator_tables = tables
    hsub._tile_operator_checks = grp4_direct
    return (hsub, alpha, beta), None
