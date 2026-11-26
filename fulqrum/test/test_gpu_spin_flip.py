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

"""Spin penalties H + lambda S^2 on the GPU operator.

S^2's spin-flip terms move alpha j -> i together with beta i -> j, so their alpha-beta groups have one value
for opposite-direction moves and another for same-direction moves. The GPU tables keep the same-direction
value and the spin-flip correction adds the difference (fulqrum/gpu/distributed/spin_flip.py).

Sectors with N_alpha != N_beta target the lowest spin of the sector, S = (N_alpha - N_beta) / 2, with the
linear penalty lambda (S^2 - S(S + 1)), as PySCF's fix_spin_ does there.
"""

import itertools

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.gpu.distributed.operator_check import (
    spin_flip_corrections,
    unsupported_groups,
)
from fulqrum.gpu.distributed.spin_flip import correction_table, move_table
from fulqrum.test.test_type2_differential import (
    NORB,
    ORDERINGS,
    bitstring,
    make_operator,
)

pyscf = pytest.importorskip("pyscf")

LAMBDA = 0.2
# (hydrogen atoms = orbitals, (N_alpha, N_beta)); at least two electrons per sector for the GPU operator
SECTORS = [(4, (2, 2)), (6, (3, 3)), (5, (3, 2)), (6, (4, 2)), (7, (5, 2))]


def s_squared(norb):
    """S^2 for any S_z, term by term: S_- S_+ + S_z + S_z^2 (alpha orbital i -> mode i, beta -> i + norb)."""
    n = norb
    terms = []
    for i in range(n):
        terms.append(("1", [i + n], 1.0))
        terms.append(("11", [i, i + n], -1.0))
        for j in range(n):
            if i != j:
                terms.append(("++--", [i + n, j, i, j + n], -1.0))
        terms.append(("1", [i], 0.5))
        terms.append(("1", [i + n], -0.5))
    for i, j in itertools.product(range(n), repeat=2):
        for p, sp in ((i, 1.0), (i + n, -1.0)):
            for q, sq in ((j, 1.0), (j + n, -1.0)):
                if p == q:
                    terms.append(("1", [p], 0.25 * sp * sq))
                else:
                    terms.append(("11", [p, q], 0.25 * sp * sq))
    return fq.FermionicOperator(2 * n, terms)


def chain_integrals(n_atoms):
    from pyscf import ao2mo, gto, scf

    mol = gto.M(
        atom="; ".join(f"H 0 0 {0.9 * k}" for k in range(n_atoms)),
        basis="sto-3g",
        spin=n_atoms % 2,
        verbose=0,
    )
    mf = (scf.ROHF if n_atoms % 2 else scf.RHF)(mol).run()
    # The sign of each orbital is arbitrary and can differ between processes;
    # fix it, so that every MPI rank builds the same Hamiltonian.
    coeff = mf.mo_coeff
    first = np.argmax(np.abs(coeff) > 1e-6, axis=0)
    coeff = coeff * np.sign(coeff[first, np.arange(coeff.shape[1])])
    h1 = coeff.T @ mf.get_hcore() @ coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, coeff), mol.nao)
    return np.ascontiguousarray(h1), np.ascontiguousarray(h2)


def sector_strings(norb, electrons):
    return sorted(
        bitstring(occ, norb) for occ in itertools.combinations(range(norb), electrons)
    )


def penalised_hamiltonian(n_atoms, nelec, lam=LAMBDA):
    """H + lam S^2 as a type-2 SubspaceHamiltonian on the full sector (no constant terms)."""
    from fulqrum.convert import integrals_to_fq_fermionic_op

    h1, h2 = chain_integrals(n_atoms)
    op = integrals_to_fq_fermionic_op(h1, h2)
    if lam:
        op = op + s_squared(n_atoms) * lam
    qubit_op = op.extended_jw_transformation()
    qubit_op, _ = qubit_op.remove_constant_terms()
    qubit_op.set_type(2)
    alpha, beta = sector_strings(n_atoms, nelec[0]), sector_strings(n_atoms, nelec[1])
    return fq.SubspaceHamiltonian(qubit_op, fq.Subspace([alpha, beta])), alpha, beta


# ----------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_spin_penalty_groups_need_and_get_the_correction(n_atoms, nelec):
    hsub, _, _ = penalised_hamiltonian(n_atoms, nelec)
    pairs = n_atoms * (n_atoms - 1) // 2
    refused = unsupported_groups(hsub.spmv, n_atoms)
    assert len(refused) == pairs
    assert {reason for _g, _f, reason in refused} == {"direction_dependent_cross_value"}
    assert unsupported_groups(hsub.spmv, n_atoms, spin_flips=True) == []
    corrections = spin_flip_corrections(hsub.spmv, n_atoms)
    assert sorted(corrections) == list(itertools.combinations(range(n_atoms), 2))
    np.testing.assert_allclose(list(corrections.values()), -LAMBDA, rtol=0, atol=1e-12)


def test_operators_without_direction_dependence_have_no_corrections():
    hsub, _, _ = penalised_hamiltonian(4, (2, 2), lam=0.0)
    assert spin_flip_corrections(hsub.spmv, 4) == {}
    assert unsupported_groups(hsub.spmv, 4, spin_flips=True) == []


@pytest.mark.parametrize("kind", ["aabb_random", "aabb_partial"])
def test_other_direction_dependent_groups_stay_refused(kind):
    alpha, beta = ORDERINGS["block"]
    halves = [bitstring(occ, NORB) for occ in itertools.combinations(range(NORB), 2)]
    qubit_op = make_operator(kind, alpha, beta).extended_jw_transformation()
    spmv = fq.SubspaceHamiltonian(qubit_op, fq.Subspace([halves, halves])).spmv
    reasons = {r for _g, _f, r in unsupported_groups(spmv, NORB, spin_flips=True)}
    assert reasons == {"direction_dependent_cross_value"}


def test_move_table_matches_brute_force():
    norb, electrons = 6, 3
    every = [int(s, 2) for s in sector_strings(norb, electrons)]
    keep = sorted(np.random.RandomState(3).choice(every, 14, replace=False).tolist())
    strings = np.array(keep, dtype=np.uint64)
    rows = strings[2:9]
    table = move_table(rows, strings, norb)
    for r, row in enumerate(rows.tolist()):
        occupied = [p for p in range(norb) if row >> p & 1]
        empty = [p for p in range(norb) if not row >> p & 1]
        for k, j in enumerate(occupied):
            for m, i in enumerate(empty):
                target = row ^ (1 << j) ^ (1 << i)
                if target not in keep:
                    assert table[r, k, m] == 0
                    continue
                lo, hi = min(i, j), max(i, j)
                between = bin(row & (((1 << hi) - 1) & ~((1 << (lo + 1)) - 1))).count(
                    "1"
                )
                assert table[r, k, m] == (-1) ** between * (keep.index(target) + 1)


def test_correction_table_uses_the_gpu_pair_order():
    norb = 5
    table = correction_table({(1, 3): 2.0, (4, 0): -1.0}, norb)
    gpu_pair = lambda low, high: low * (2 * norb - low - 1) // 2 + (high - low - 1)  # noqa: E731
    expected = np.zeros(norb * (norb - 1) // 2)
    expected[gpu_pair(1, 3)] = 2.0
    expected[gpu_pair(0, 4)] = -1.0
    np.testing.assert_array_equal(table, expected)


# ----------------------------------------------------------------------------- GPU


def _grids(size):
    return [(p, size // p) for p in range(1, size + 1) if size % p == 0]


@pytest.mark.gpu
@pytest.mark.parametrize("cliques", ["1", "0"])
@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_gpu_matvec_with_spin_penalty_matches_cpu(
    monkeypatch, n_atoms, nelec, ranks, cliques
):
    pytest.importorskip("cupy")
    pytest.importorskip("mpi4py")
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )
    from fulqrum.test.conftest import assemble_tiles, run_local_ranks

    monkeypatch.setenv("FQ_AABB_CLIQUE_ENABLE", cliques)
    hsub, alpha, beta = penalised_hamiltonian(n_atoms, nelec)
    x = np.random.RandomState(11).uniform(-1.0, 1.0, hsub.shape[0])
    expected = hsub @ x
    for grid in _grids(ranks):

        def body(comm):
            gpu = TileGPUResidentHamiltonian(
                hsub, alpha, beta, comm=comm, process_grid=grid, comm_backend="inproc"
            )
            assert (
                gpu.construction_timing.get("spin_flip_pairs")
                == n_atoms * (n_atoms - 1) // 2
            )
            tile = x.reshape(gpu._n_b, gpu._n_a)[gpu._b0 : gpu._b1, gpu._a0 : gpu._a1]
            local = gpu._matvec(np.ascontiguousarray(tile).ravel())
            return (gpu._a0, gpu._a1, gpu._b0, gpu._b1), np.asarray(local)

        result = assemble_tiles(run_local_ranks(ranks, body), len(alpha), len(beta))
        error = np.linalg.norm(result - expected) / np.linalg.norm(expected)
        assert error < 1e-12, (grid, error)


@pytest.mark.gpu
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_gpu_spin_penalty_eigenpairs_match_pyscf(n_atoms, nelec, workers):
    cupy = pytest.importorskip("cupy")
    from pyscf import fci
    from scipy.sparse.linalg import eigsh

    from fulqrum.gpu import GPUHamiltonian

    spin = 0.5 * (nelec[0] - nelec[1])
    ss = spin * (spin + 1)
    h1, h2 = chain_integrals(n_atoms)
    solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), shift=LAMBDA, ss=ss)
    solver.conv_tol = 1e-12
    e_pyscf, c_pyscf = solver.kernel(h1, h2, n_atoms, nelec)
    s2_pyscf = fci.spin_op.spin_square(c_pyscf, n_atoms, nelec)[0]

    hsub, alpha, beta = penalised_hamiltonian(n_atoms, nelec)
    device = int(cupy.cuda.runtime.getDevice())
    gpu = GPUHamiltonian(hsub, gpu_ids=(device,) * workers)
    values, vectors = eigsh(gpu, k=1, which="SA", tol=1e-12)
    vector = vectors[:, 0] / np.linalg.norm(vectors[:, 0])
    # PySCF's linear penalty is lambda (S^2 - ss); the operator here is H + lambda S^2.
    assert abs(values[0] - LAMBDA * ss - e_pyscf) < 1e-9

    s2_qubit = s_squared(n_atoms).extended_jw_transformation()
    s2_qubit.set_type(2)
    s2_sub = fq.SubspaceHamiltonian(s2_qubit, fq.Subspace([alpha, beta]))
    s2 = float(vector @ (s2_sub @ vector))
    assert abs(s2 - ss) < 1e-6
    assert abs(s2 - s2_pyscf) < 1e-6


@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize("n_atoms,nelec", [(6, (3, 3)), (5, (3, 2)), (6, (4, 2))])
def test_mpi_spin_penalty_matches_cpu_and_pyscf(n_atoms, nelec):
    """One GPU per MPI rank (``mpirun -n 2`` or more): matvec and eigenpair of H + lambda S^2."""
    MPI = pytest.importorskip("mpi4py.MPI")
    cupy = pytest.importorskip("cupy")
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs at least 2 MPI ranks; run under mpirun")
    from pyscf import fci

    from fulqrum.gpu import DistributedMPIGPUHamiltonian, distributed_primme_eigsh
    from fulqrum.gpu.distributed import _collective
    from fulqrum.test.conftest import assemble_tiles

    _collective.reset_for_testing()
    local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    device = local_rank % cupy.cuda.runtime.getDeviceCount()
    cupy.cuda.Device(device).use()
    hsub, alpha, beta = penalised_hamiltonian(n_atoms, nelec)
    ham = DistributedMPIGPUHamiltonian(hsub, comm=comm, gpu_ids=(device,))
    pairs = n_atoms * (n_atoms - 1) // 2
    assert ham.construction_timing.get("spin_flip_pairs") == pairs

    x = np.random.RandomState(11).uniform(-1.0, 1.0, hsub.shape[0])
    tile = x.reshape(ham._n_b, ham._n_a)[ham._b0 : ham._b1, ham._a0 : ham._a1]
    local = np.asarray(ham._matvec(np.ascontiguousarray(tile).ravel()))
    pieces = comm.allgather(((ham._a0, ham._a1, ham._b0, ham._b1), local))
    result = assemble_tiles(pieces, len(alpha), len(beta))
    expected = hsub @ x
    assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 1e-12

    spin = 0.5 * (nelec[0] - nelec[1])
    ss = spin * (spin + 1)
    h1, h2 = chain_integrals(n_atoms)
    solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), shift=LAMBDA, ss=ss)
    solver.conv_tol = 1e-12
    e_pyscf, _ = solver.kernel(h1, h2, n_atoms, nelec)
    diag = np.asarray(ham.diag_local)
    lowest = comm.allreduce(float(diag.min()) if diag.size else np.inf, op=MPI.MIN)
    v0 = np.zeros(diag.size)
    if diag.size and diag.min() == lowest:
        v0[int(diag.argmin())] = 1.0
    values, _ = distributed_primme_eigsh(
        ham,
        k=1,
        tol=1e-9,
        which="SA",
        precond_diag_local=diag,
        v0_local=v0,
        energy_tol=1e-13,
        gpu=True,
    )
    # PySCF's linear penalty is lambda (S^2 - ss); the operator here is H + lambda S^2.
    assert abs(float(np.asarray(values)[0]) - LAMBDA * ss - e_pyscf) < 1e-8
    comm.Barrier()
