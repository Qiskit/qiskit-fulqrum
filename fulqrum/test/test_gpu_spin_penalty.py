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

"""The GPU Hamiltonians' ``spin_penalty`` option, against H + lam S^2 and PySCF."""

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.gpu.spin_penalty import penalized_hamiltonian
from fulqrum.test.test_gpu_spin_flip import (
    chain_integrals,
    penalised_hamiltonian,
    sector_strings,
)

pytest.importorskip("pyscf")

# (hydrogen atoms = orbitals, (N_alpha, N_beta)): singlet, doublet, triplet
SECTORS = [(4, (2, 2)), (5, (3, 2)), (6, (4, 2))]


def h_only(n_atoms, nelec):
    hsub, _, _ = penalised_hamiltonian(n_atoms, nelec, lam=0.0)
    return hsub


def h_with_constant(n_atoms, nelec, constant):
    hsub = h_only(n_atoms, nelec)
    identity = fq.QubitOperator(2 * n_atoms, [("", [], constant)])
    identity.set_type(2)
    operator = hsub.diag_H + hsub.off_H + identity
    return fq.SubspaceHamiltonian(operator, hsub.spmv.subspace)


def lowest_ss(nelec):
    spin = 0.5 * (nelec[0] - nelec[1])
    return spin * (spin + 1)


def pyscf_reference(n_atoms, nelec, lam):
    from pyscf import fci

    h1, h2 = chain_integrals(n_atoms)
    solver = fci.addons.fix_spin_(
        fci.direct_spin1.FCI(), shift=lam, ss=lowest_ss(nelec)
    )
    solver.conv_tol = 1e-12
    e_ref, c_ref = solver.kernel(h1, h2, n_atoms, nelec)
    return e_ref, fci.spin_op.spin_square(c_ref, n_atoms, nelec)[0]


# ----------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("lam", [0.1, 0.2])
@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_penalized_hamiltonian_equals_a_hand_built_penalty(n_atoms, nelec, lam):
    penalized, sector = penalized_hamiltonian(h_only(n_atoms, nelec), lam)
    assert sector == nelec
    # H + lam S^2, without constant terms
    by_hand, _, _ = penalised_hamiltonian(n_atoms, nelec, lam=lam)
    x = np.random.RandomState(5).uniform(-1.0, 1.0, by_hand.shape[0])
    expected = by_hand @ x - lam * lowest_ss(nelec) * x
    np.testing.assert_allclose(penalized @ x, expected, rtol=0, atol=1e-11)


def test_penalized_hamiltonian_keeps_the_constant_of_h():
    n_atoms, nelec, lam, constant = 4, (2, 2), 0.1, 0.7
    with_constant = h_with_constant(n_atoms, nelec, constant)
    assert with_constant.const_energy == pytest.approx(constant)
    penalized, _ = penalized_hamiltonian(with_constant, lam)
    x = np.random.RandomState(5).uniform(-1.0, 1.0, with_constant.shape[0])
    by_hand, _, _ = penalised_hamiltonian(n_atoms, nelec, lam=lam)
    expected = by_hand @ x + (constant - lam * lowest_ss(nelec)) * x
    np.testing.assert_allclose(penalized @ x, expected, rtol=0, atol=1e-11)


def test_a_target_that_needs_the_quadratic_penalty_is_refused():
    with pytest.raises(ValueError, match="quadratic"):
        penalized_hamiltonian(h_only(4, (2, 2)), 0.1, target_s=1.0)


# ----------------------------------------------------------------------------- GPU


@pytest.mark.gpu
@pytest.mark.parametrize("lam", [0.1, 0.2])
@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_gpu_spin_penalty_matches_hand_built_and_pyscf(n_atoms, nelec, lam):
    cupy = pytest.importorskip("cupy")
    from scipy.sparse.linalg import eigsh

    from fulqrum.gpu import GPUHamiltonian

    device = int(cupy.cuda.runtime.getDevice())
    gpu = GPUHamiltonian(h_only(n_atoms, nelec), gpu_ids=(device,), spin_penalty=lam)
    by_hand, _, _ = penalised_hamiltonian(n_atoms, nelec, lam=lam)
    x = np.random.RandomState(5).uniform(-1.0, 1.0, by_hand.shape[0])
    expected = by_hand @ x - lam * lowest_ss(nelec) * x
    got = gpu @ x
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-12

    values, vectors = eigsh(gpu, k=1, which="SA", tol=1e-12)
    out = gpu.spin_penalty_expectations(values[0], vectors[:, 0])
    e_ref, s2_ref = pyscf_reference(n_atoms, nelec, lam)
    assert abs(values[0] - e_ref) < 1e-9
    assert abs(out["s2"] - lowest_ss(nelec)) < 1e-6
    assert abs(out["s2"] - s2_ref) < 1e-6
    # the linear penalty vanishes at the target spin
    assert abs(out["energy"] - e_ref) < 1e-9


@pytest.mark.gpu
def test_gpu_spin_penalty_keeps_the_constant_of_h():
    cupy = pytest.importorskip("cupy")
    from fulqrum.gpu import GPUHamiltonian

    n_atoms, nelec, lam, constant = 5, (3, 2), 0.1, 0.7
    device = int(cupy.cuda.runtime.getDevice())
    gpu = GPUHamiltonian(
        h_with_constant(n_atoms, nelec, constant), gpu_ids=(device,), spin_penalty=lam
    )
    by_hand, _, _ = penalised_hamiltonian(n_atoms, nelec, lam=lam)
    x = np.random.RandomState(5).uniform(-1.0, 1.0, by_hand.shape[0])
    expected = by_hand @ x + (constant - lam * lowest_ss(nelec)) * x
    got = gpu @ x
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-12


@pytest.mark.gpu
@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_distributed_spin_penalty_matches_hand_built(n_atoms, nelec):
    pytest.importorskip("cupy")
    from fulqrum.gpu import DistributedMPIGPUHamiltonian, spin_penalty_expectations
    from fulqrum.test.conftest import assemble_tiles, run_local_ranks

    lam = 0.1
    hsub = h_only(n_atoms, nelec)
    alpha, beta = sector_strings(n_atoms, nelec[0]), sector_strings(n_atoms, nelec[1])
    by_hand, _, _ = penalised_hamiltonian(n_atoms, nelec, lam=lam)
    dense = np.array([by_hand @ e for e in np.eye(by_hand.shape[0])]).T
    w, v = np.linalg.eigh(dense)
    vector = v[:, 0]
    x = np.random.RandomState(5).uniform(-1.0, 1.0, by_hand.shape[0])

    def body(comm):
        ham = DistributedMPIGPUHamiltonian(
            hsub, comm=comm, comm_backend="inproc", spin_penalty=lam
        )
        tile = (slice(ham._b0, ham._b1), slice(ham._a0, ham._a1))
        local_x = np.ascontiguousarray(x.reshape(ham._n_b, ham._n_a)[tile]).ravel()
        y = np.asarray(ham._matvec(local_x))
        local_v = np.ascontiguousarray(vector.reshape(ham._n_b, ham._n_a)[tile]).ravel()
        # w[0] belongs to H + lam S^2; ham is H + lam (S^2 - ss)
        out = spin_penalty_expectations(ham, w[0] - lam * lowest_ss(nelec), local_v)
        return ((ham._a0, ham._a1, ham._b0, ham._b1), y), out

    results = run_local_ranks(2, body)
    got = assemble_tiles([r[0] for r in results], len(alpha), len(beta))
    expected = by_hand @ x - lam * lowest_ss(nelec) * x
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 1e-12
    h_dense = np.array([h_only(n_atoms, nelec) @ e for e in np.eye(len(vector))]).T
    for _, out in results:
        assert abs(out["s2"] - lowest_ss(nelec)) < 1e-8
        assert abs(out["energy"] - (vector @ h_dense @ vector)) < 1e-9  # <H>


@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize("n_atoms,nelec", SECTORS)
def test_mpi_spin_penalty_option_matches_pyscf(n_atoms, nelec):
    """One GPU per MPI rank (``mpirun -n 2`` or more)."""
    MPI = pytest.importorskip("mpi4py.MPI")
    cupy = pytest.importorskip("cupy")
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs at least 2 MPI ranks; run under mpirun")
    from fulqrum.gpu import (
        DistributedMPIGPUHamiltonian,
        distributed_primme_eigsh,
        spin_penalty_expectations,
    )
    from fulqrum.gpu.distributed import _collective

    _collective.reset_for_testing()
    local_rank = comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    device = local_rank % cupy.cuda.runtime.getDeviceCount()
    cupy.cuda.Device(device).use()
    lam = 0.1
    ham = DistributedMPIGPUHamiltonian(
        h_only(n_atoms, nelec), comm=comm, gpu_ids=(device,), spin_penalty=lam
    )
    diag = np.asarray(ham.diag_local)
    lowest = comm.allreduce(float(diag.min()) if diag.size else np.inf, op=MPI.MIN)
    v0 = np.zeros(diag.size)
    if diag.size and diag.min() == lowest:
        v0[int(diag.argmin())] = 1.0
    values, vectors = distributed_primme_eigsh(
        ham,
        k=1,
        tol=1e-9,
        which="SA",
        precond_diag_local=diag,
        v0_local=v0,
        energy_tol=1e-13,
        gpu=True,
    )
    value = float(np.asarray(values)[0])
    out = spin_penalty_expectations(ham, value, np.asarray(vectors)[:, 0])
    e_ref, s2_ref = pyscf_reference(n_atoms, nelec, lam)
    assert abs(value - e_ref) < 1e-8
    assert abs(out["s2"] - s2_ref) < 1e-6
    assert abs(out["energy"] - e_ref) < 1e-8
    comm.Barrier()
