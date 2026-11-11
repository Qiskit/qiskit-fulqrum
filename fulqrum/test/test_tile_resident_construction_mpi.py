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

"""Exchange selection on the host_mpi backend, across four MPI processes.

The backend-independent grid, discovery and exchange-plan checks run on
in-process ranks in test_tile_resident_construction.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .conftest import MPI_RUN_HINT

pytest.importorskip("mpi4py")
from mpi4py import MPI

if MPI.COMM_WORLD.Get_size() != 4:
    pytest.skip(
        f"needs 4 MPI ranks, one GPU each (got {MPI.COMM_WORLD.Get_size()}); "
        f"run with `{MPI_RUN_HINT}`",
        allow_module_level=True,
    )

from fulqrum.gpu.distributed import _collective as col
from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
    TileGPUResidentHamiltonian,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

pytestmark = [pytest.mark.mpi, pytest.mark.gpu]


def _make_tiny_hsub() -> SimpleNamespace:
    return SimpleNamespace(spmv=SimpleNamespace(is_real=True))


def _construct_2x2():
    """Construct the 2x2 operator on the tiny subspace with the host_mpi backend."""
    col.reset_for_testing()
    return TileGPUResidentHamiltonian(
        _make_tiny_hsub(),
        np.arange(4, dtype=np.uint64),
        np.arange(4, dtype=np.uint64),
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
    )


def test_explicit_two_level_with_one_domain_raises_on_every_rank(monkeypatch):
    """FQ_EXCHANGE=two_level with one shared-memory domain raises ValueError on every rank, with no hang."""
    monkeypatch.setenv("FQ_EXCHANGE", "two_level")
    with pytest.raises(ValueError, match="at least two shared-memory domains"):
        _construct_2x2()
    COMM.Barrier()


def _h2_local_matvec(intra_node_size):
    from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian

    fop = FermionicOperator.from_json(Path(__file__).parent / "data/h2.json")
    strings = ["01", "10"]
    hsub = SubspaceHamiltonian(
        fop.extended_jw_transformation(), Subspace([strings, strings])
    )
    col.reset_for_testing()
    H = TileGPUResidentHamiltonian(
        hsub,
        strings,
        strings,
        comm=COMM,
        process_grid=(2, 2),
        comm_backend="host_mpi",
        intra_node_size=intra_node_size,
    )
    x = np.random.default_rng(7).uniform(-1.0, 1.0, hsub.shape[0])
    tile = x.reshape(H._n_b, H._n_a)[H._b0 : H._b1, H._a0 : H._a1]
    out = np.asarray(H._matvec(np.ascontiguousarray(tile).ravel()))
    return H.construction_timing, out


def test_explicit_two_level_on_host_mpi_with_two_domains_matches_ring(monkeypatch):
    """FQ_EXCHANGE=two_level on host_mpi with two domains runs two-level and matches ring."""
    monkeypatch.setenv("FQ_EXCHANGE", "ring")
    ring_timing, ring = _h2_local_matvec(None)
    assert ring_timing["exchange"] == "ring"
    monkeypatch.setenv("FQ_EXCHANGE", "two_level")
    timing, out = _h2_local_matvec(2)
    assert timing["exchange"] == "two_level"
    assert timing["exchange_reason"].startswith("2 shared-memory domains with host_mpi")
    assert np.max(np.abs(out - ring)) <= 1e-13 * max(1.0, np.max(np.abs(ring)))


def test_exchange_mode_differing_across_ranks_raises_on_every_rank(monkeypatch):
    """FQ_EXCHANGE set on one rank only raises ValueError on every rank, with no hang."""
    if RANK == 1:
        monkeypatch.setenv("FQ_EXCHANGE", "ring")
    else:
        monkeypatch.delenv("FQ_EXCHANGE", raising=False)
    with pytest.raises(ValueError, match="differs? across ranks"):
        _construct_2x2()
    COMM.Barrier()


def test_auto_exchange_on_host_mpi_runs_ring_and_records_it(monkeypatch):
    """FQ_EXCHANGE=auto on host_mpi runs ring and records the reason and the settings digest."""
    monkeypatch.delenv("FQ_EXCHANGE", raising=False)
    H = _construct_2x2()
    timing = H.construction_timing
    assert timing["exchange"] == "ring"
    assert "FQ_EXCHANGE=two_level" in timing["exchange_reason"]
    assert len(set(COMM.allgather(timing["exchange_digest"]))) == 1
