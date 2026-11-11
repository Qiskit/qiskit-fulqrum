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
"""Ground-state energy of a molecule in a tensor-product subspace on many GPUs.

Run one MPI rank per GPU.

Determinant files hold one bitstring per line, with orbital 0 the
rightmost character. All strings in a file must have the same length (the
number of spatial orbitals) and the same number of ones (electrons of that
spin).

This is an example; adapt it to your problem and cluster.
"""

import argparse
import time

import numpy as np
from mpi4py import MPI

import fulqrum as fq
from fulqrum.convert import fcidump_to_fq_fermionic_op
from fulqrum.gpu import DistributedMPIGPUHamiltonian, distributed_primme_eigsh


def read_strings(path):
    with open(path) as handle:
        strings = [line.strip() for line in handle if line.strip()]
    if not strings:
        raise ValueError(f"{path} has no determinant strings")
    return strings


def parse_grid(text):
    if text is None:
        return None
    p_alpha, p_beta = (int(value) for value in text.lower().split("x"))
    return p_alpha, p_beta


def main():
    parser = argparse.ArgumentParser(
        description="Ground-state energy in a tensor-product subspace on many GPUs."
    )
    parser.add_argument("--fcidump", required=True, help="FCIDUMP file of the molecule")
    parser.add_argument(
        "--alpha-dets", required=True, help="alpha half strings, one per line"
    )
    parser.add_argument(
        "--beta-dets", help="beta half strings; defaults to the alpha strings"
    )
    parser.add_argument(
        "-r",
        "--residual-norm",
        type=float,
        default=1e-3,
        help="residual norm tolerance",
    )
    parser.add_argument("--nroots", type=int, default=1, help="number of eigenpairs")
    parser.add_argument(
        "--gpu-eigsh",
        type=int,
        choices=(0, 1),
        default=0,
        help="run the eigensolver's linear algebra on the GPU (1) or the host (0); "
        "matvecs run on the GPU either way",
    )
    parser.add_argument(
        "--grid",
        help="process grid PxQ (alpha x beta); P*Q must equal the number of ranks. "
        "Defaults to an automatic choice.",
    )
    parser.add_argument(
        "--rdm",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="also compute the ground state's 1-RDM (1) or 1- and 2-RDMs (2)",
    )
    parser.add_argument("--out", help="save energies (and RDMs) to this .npz file")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def report(text):
        if rank == 0:
            print(text, flush=True)

    alpha = read_strings(args.alpha_dets)
    beta = read_strings(args.beta_dets) if args.beta_dets else alpha
    report(
        f"ranks={comm.Get_size()} orbitals={len(alpha[0])} "
        f"alpha={len(alpha)} beta={len(beta)} dets={len(alpha) * len(beta):,}"
    )

    start = time.perf_counter()
    operator = fcidump_to_fq_fermionic_op(args.fcidump).extended_jw_transformation()
    operator, constant = operator.remove_constant_terms()
    operator.set_type(2)
    hsub = fq.SubspaceHamiltonian(operator, fq.Subspace([alpha, beta]))
    hamiltonian = DistributedMPIGPUHamiltonian(
        hsub, comm=comm, process_grid=parse_grid(args.grid)
    )
    p_alpha, p_beta = hamiltonian.process_grid
    report(
        f"build {time.perf_counter() - start:.1f} s, grid {p_alpha}x{p_beta}, "
        f"backend {hamiltonian.backend_name}"
    )

    # Start from the lowest diagonal element and precondition with the diagonal.
    diagonal = np.asarray(hamiltonian.diag_local)
    lowest = comm.allreduce(float(diagonal.min()), op=MPI.MIN)
    start_vector = np.zeros(diagonal.size)
    if diagonal.size and diagonal.min() == lowest:
        start_vector[int(diagonal.argmin())] = 1.0

    start = time.perf_counter()
    energies, vectors, stats = distributed_primme_eigsh(
        hamiltonian,
        k=args.nroots,
        tol=args.residual_norm,
        which="SA",
        method="PRIMME_GD_Olsen_plusK",
        ncv=4,
        minRestartSize=2,
        maxPrevRetain=1,
        precond_diag_local=diagonal,
        v0_local=start_vector,
        return_stats=True,
        gpu=bool(args.gpu_eigsh),
    )
    seconds = time.perf_counter() - start
    energies = np.asarray(energies) + constant
    for index, energy in enumerate(energies):
        report(f"E[{index}] = {energy:.12f}")
    report(f"solve {seconds:.1f} s, {stats.get('numMatvecs')} matvecs")

    results = {"energies": energies}
    if args.rdm:
        ground = np.ascontiguousarray(np.asarray(vectors)[:, 0])
        start = time.perf_counter()
        rdms = hamiltonian.compute_rdms(ground, rank=args.rdm)
        report(f"rank-{args.rdm} RDM {time.perf_counter() - start:.1f} s")
        if args.rdm == 1:
            results["gamma_a"], results["gamma_b"] = rdms
        else:
            (results["gamma_a"], results["gamma_b"]), blocks = rdms
            results["G_aa"], results["G_ab"], results["G_bb"] = blocks
    if args.out and rank == 0:
        np.savez(args.out, **results)
        report(f"saved {args.out}")


if __name__ == "__main__":
    main()
