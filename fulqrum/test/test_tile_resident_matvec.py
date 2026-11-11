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

from pathlib import Path

import numpy as np
import pytest

try:
    import cupy
except ImportError:
    cupy = None

from fulqrum.gpu.distributed import _collective as col
from .conftest import (
    MPI_RUN_HINT,
    assemble_tiles,
    gpu_available,
    make_lih_hsub,
    run_local_ranks,
)


def _has_cupy() -> bool:
    return cupy is not None


_H2_PATH = Path(__file__).parent / "data/h2.json"


def _make_h2_hsub():
    from fulqrum import FermionicOperator, Subspace, SubspaceHamiltonian

    fop = FermionicOperator.from_json(_H2_PATH)
    qop = fop.extended_jw_transformation()

    alpha_strs = ["01", "10"]
    beta_strs = ["01", "10"]
    sub = Subspace([alpha_strs, beta_strs])
    hsub = SubspaceHamiltonian(qop, sub)
    return hsub, alpha_strs, beta_strs, qop


def _subspace_dense(hsub):
    n = hsub.shape[0]
    return np.column_stack([hsub @ unit for unit in np.eye(n)])


def _local_rank_matvec(hsub, alpha_strs, beta_strs, v_global, grid=(2, 2)):
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    def body(comm):
        H = TileGPUResidentHamiltonian(
            hsub,
            alpha_strs,
            beta_strs,
            process_grid=grid,
            comm=comm,
            comm_backend="inproc",
        )
        v_2d = v_global.reshape(H._n_b, H._n_a)
        v_local = np.ascontiguousarray(v_2d[H._b0 : H._b1, H._a0 : H._a1].ravel())
        Hv_local = H._matvec(v_local)
        assert Hv_local.size == H.n_local
        return (H._a0, H._a1, H._b0, H._b1), Hv_local

    return assemble_tiles(
        run_local_ranks(grid[0] * grid[1], body), len(alpha_strs), len(beta_strs)
    )


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_matvec_single_rank_h2_matches_dense():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, qop = _make_h2_hsub()

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        comm=MPI.COMM_SELF,
        comm_backend="host_mpi",
    )

    rng = np.random.default_rng(seed=1234)
    v = rng.standard_normal(H.n_local).astype(np.float64)

    Hv = H._matvec(v)

    M_dense = _subspace_dense(hsub)
    ref = M_dense @ v

    np.testing.assert_allclose(Hv, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_matvec_4rank_h2_matches_dense_on_host_mpi():
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    if nranks != 4:
        pytest.skip(f"needs 4 MPI ranks, got {nranks}; run with `{MPI_RUN_HINT}`")

    col.reset_for_testing()
    hsub, alpha_strs, beta_strs, qop = _make_h2_hsub()

    H = TileGPUResidentHamiltonian(
        hsub,
        alpha_strs,
        beta_strs,
        process_grid=(2, 2),
        comm=comm,
        comm_backend="host_mpi",
    )

    n_a = H._n_a
    n_b = H._n_b
    n_global = n_a * n_b

    rng = np.random.default_rng(seed=4321)
    v_global = rng.standard_normal(n_global).astype(np.float64)
    v_2d = v_global.reshape(n_b, n_a)
    v_local_2d = v_2d[H._b0 : H._b1, H._a0 : H._a1]
    v_local = np.ascontiguousarray(v_local_2d.ravel(), dtype=np.float64)

    Hv_local = H._matvec(v_local)
    assert Hv_local.size == H.n_local

    counts = np.array(comm.allgather(int(Hv_local.size)), dtype=np.int32)
    displs = np.zeros(nranks, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    flat_all = np.empty(int(counts.sum()), dtype=np.float64)
    comm.Allgatherv(
        np.ascontiguousarray(Hv_local),
        [flat_all, counts, displs, MPI.DOUBLE],
    )

    Hv_global_2d = np.zeros((n_b, n_a), dtype=np.float64)
    for r in range(nranks):
        a0_r, a1_r, b0_r, b1_r, _ca, _cb = H._rank_tiles[r]
        n_a_r = a1_r - a0_r
        n_b_r = b1_r - b0_r
        seg = flat_all[displs[r] : displs[r] + counts[r]]
        Hv_global_2d[b0_r:b1_r, a0_r:a1_r] = seg.reshape(n_b_r, n_a_r)
    Hv_global = Hv_global_2d.ravel()

    M_dense = _subspace_dense(hsub)
    ref = M_dense @ v_global

    err = np.max(np.abs(Hv_global - ref))
    assert err < 1e-9, (
        f"rank {rank}: max-abs error {err:.3e} on 4-rank H2 matvec exceeds 1e-9"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="GPU build / CUDA device unavailable")
def test_matvec_4rank_h2_matches_dense():
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()
    n_a, n_b = len(alpha_strs), len(beta_strs)
    v_global = np.random.default_rng(seed=4321).standard_normal(n_a * n_b)
    Hv_global = _local_rank_matvec(hsub, alpha_strs, beta_strs, v_global)
    err = np.max(np.abs(Hv_global - _subspace_dense(hsub) @ v_global))
    assert err < 1e-9, f"max-abs error {err:.3e} on 4-rank H2 matvec exceeds 1e-9"


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="GPU build / CUDA device unavailable")
def test_matvec_4rank_hv_fingerprint_smoke():
    hsub, alpha_strs, beta_strs, _qop = _make_h2_hsub()
    n_a, n_b = len(alpha_strs), len(beta_strs)
    v_global = np.random.default_rng(seed=2024).standard_normal(n_a * n_b)
    Hv = _local_rank_matvec(hsub, alpha_strs, beta_strs, v_global)
    assert float(np.sum(Hv * Hv)) > 0.0, (
        "Hv is identically zero -- every kernel arm missed"
    )
    assert Hv.max() > Hv.min(), "Hv is constant -- likely a degenerate fixture or a bug"


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
@pytest.mark.parametrize("source", ["host", "kernel"])
def test_matvec_lih_matches_cpu(monkeypatch, source):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    hsub, alpha_strs, beta_strs = make_lih_hsub()
    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", source)
    col.reset_for_testing()
    H = TileGPUResidentHamiltonian(
        hsub, alpha_strs, beta_strs, comm=MPI.COMM_SELF, comm_backend="host_mpi"
    )
    assert H.construction_timing["aabb_csr_enabled"] is True
    assert H.predicted_hbm_bytes["aabb_csr"] > 0
    assert (H._d_term_header is None) == (source == "host")

    rng = np.random.default_rng(seed=7)
    v = rng.standard_normal(H.n_local).astype(np.float64)
    for _ in range(2):
        hv = np.asarray(H._matvec(v))
        assert H._d_term_header is None
        expected = hsub @ v
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(hv, expected, atol=1e-12 * scale, rtol=1e-12)
        v = hv / np.linalg.norm(hv)

    steady = np.asarray(H._matvec(v))
    H.reset_cache()
    rebuilt = np.asarray(H._matvec(v))
    assert H.last_timing["caches_built_this_call"] == (source == "kernel")
    assert H._d_term_header is None
    np.testing.assert_array_equal(rebuilt, steady)


@pytest.mark.skipif(not _has_cupy(), reason="cupy required")
def test_kernel_source_rebuild_waits_for_uploads(monkeypatch):
    pytest.importorskip("mpi4py")
    from mpi4py import MPI

    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )

    hsub, alpha_strs, beta_strs = make_lih_hsub()
    monkeypatch.setenv("FQ_ARM_TABLE_SOURCE", "kernel")
    col.reset_for_testing()
    H = TileGPUResidentHamiltonian(
        hsub, alpha_strs, beta_strs, comm=MPI.COMM_SELF, comm_backend="host_mpi"
    )
    v = np.random.default_rng(seed=7).standard_normal(H.n_local)
    H._matvec(v)
    steady = np.asarray(H._matvec(v))
    load = cupy.random.default_rng(1).standard_normal((2048, 2048), dtype=cupy.float32)
    for _ in range(3):
        H.reset_cache()
        with cupy.cuda.Stream.null:
            for _ in range(10):
                load = load @ load
                load /= cupy.abs(load).max()
        np.testing.assert_array_equal(np.asarray(H._matvec(v)), steady)


def test_removed_flags_are_unknown():
    from fulqrum.gpu import config

    removed = (
        "FQ_TILE_RESIDENT_AABB_CSR",
        "FQ_AABB_XCOARSEN",
        "FQ_TILE_RESIDENT_BALANCE",
        "FQ_PRIMME_PRECOND_FUSED",
        "FQ_PRIMME_HOST_GEMV",
        "FQ_PRIMME_HOST_DENSE",
        "FQ_PRIMME_HOST_PRECOND",
        "FULQRUM_DIST_BACKEND",
    )
    assert config.unknown_names({name: "1" for name in removed}) == sorted(removed)
