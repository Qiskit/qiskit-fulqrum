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

"""The ``spin_penalty`` option of the GPU Hamiltonians.

``spin_penalty=lam`` turns the operator of a ``SubspaceHamiltonian`` into
``H + lam * (S^2 - s(s+1))`` for the lowest spin of its electron sector,
``s = |N_alpha - N_beta| / 2``: PySCF's linear ``fix_spin_`` penalty. The penalty
comes from ``fulqrum.utils.make_spin_penalized_fermionic_op``, so the CPU and GPU
operators are the same. Higher spins of a sector need the quadratic penalty,
which is four-body and available on the CPU only. The operator keeps every
constant, so its eigenvalues are those of ``H + lam * (S^2 - s(s+1))``.
"""

import numpy as np

from .common.wide import max_electrons


def _sector(subspace):
    return max_electrons(subspace.alpha_strs), max_electrons(subspace.beta_strs)


def penalized_hamiltonian(hsub, lam, target_s=None):
    """``hsub``'s operator plus ``lam * (S^2 - s(s+1))`` on the same subspace.

    Parameters:
        hsub (SubspaceHamiltonian): Type-2 Hamiltonian on an alpha x beta subspace.
        lam (float): Penalty weight.
        target_s (float or None): Target spin; ``None`` (default) is the lowest
            spin of the sector. A target that needs the quadratic penalty raises.

    Returns:
        tuple: ``(penalized, sector)``: a type-2 ``SubspaceHamiltonian`` with
        ``hsub``'s constant and the penalty's, and ``(N_alpha, N_beta)``.

    Raises:
        ValueError: The target needs the quadratic penalty.
    """
    from ..core.fermi_operator import FermionicOperator  # noqa: PLC0415
    from ..core.linear_operator import SubspaceHamiltonian  # noqa: PLC0415
    from ..core import QubitOperator  # noqa: PLC0415
    from ..utils.spin_operators import (  # noqa: PLC0415
        make_spin_penalized_fermionic_op,
        use_quadratic_penalty,
    )

    subspace = hsub.spmv.subspace
    num_a, num_b = _sector(subspace)
    spin = abs(num_a - num_b) * 0.5 if target_s is None else float(target_s)
    if use_quadratic_penalty(num_a, num_b, spin):
        raise ValueError(
            f"spin {spin} in the ({num_a}, {num_b}) sector needs the quadratic penalty "
            "(S^2 - s(s+1))^2, which is four-body; the GPU operators take the linear "
            "penalty only, for the lowest spin of the sector. Use the CPU operator "
            "with fulqrum.utils.make_spin_penalized_fermionic_op."
        )
    width = int(hsub.spmv.width)
    penalty = make_spin_penalized_fermionic_op(
        FermionicOperator(width), width // 2, num_a, num_b, spin, lam
    ).extended_jw_transformation()
    penalty.set_type(2)
    # SubspaceHamiltonian keeps the constant out of diag_H, in const_energy
    operator = hsub.diag_H + hsub.off_H + penalty
    if hsub.const_energy:
        identity = QubitOperator(width, [("", [], hsub.const_energy)])
        identity.set_type(2)
        operator = operator + identity
    operator.set_type(2)
    return SubspaceHamiltonian(operator, subspace), (num_a, num_b)


def spin_penalty_expectations(ham, eigenvalue, vector):
    """``<S^2>`` and ``<H>`` of an eigenvector of a ``spin_penalty`` GPU operator.

    Parameters:
        ham: A ``GPUHamiltonian``, or an operator from ``DistributedMPIGPUHamiltonian``,
            built with ``spin_penalty``.
        eigenvalue (float): Eigenvalue of ``ham`` for ``vector``.
        vector (ndarray): The eigenvector: the whole vector for ``GPUHamiltonian``, this
            rank's local vector for a distributed operator.

    Returns:
        dict: ``{"s2", "energy"}``, as ``fulqrum.utils.spin_penalty_expectations``;
        the same on every rank.
    """
    lam = getattr(ham, "spin_penalty", None)
    if not lam:
        raise ValueError("the operator was not built with spin_penalty")
    from ..utils.spin_operators import (  # noqa: PLC0415
        spin_penalty_expectations as cpu_expectations,
    )
    from .gpu_hamiltonian import GPUHamiltonian  # noqa: PLC0415

    num_a, num_b = ham.spin_penalty_sector
    if isinstance(ham, GPUHamiltonian):  # the whole vector is on this host
        return cpu_expectations(
            eigenvalue, vector, ham.spin_penalty_subspace, num_a, num_b, None, lam
        )
    s2 = _distributed_s_squared(ham, vector)
    spin = abs(num_a - num_b) * 0.5
    return {"s2": s2, "energy": float(eigenvalue) - lam * (s2 - spin * (spin + 1))}


def _distributed_s_squared(ham, vector_local):
    """<S^2> of a distributed vector, from an S^2 operator on the same ranks.

    The S^2 operator is built on first use and kept on ``ham``.
    """
    from ..core.linear_operator import SubspaceHamiltonian  # noqa: PLC0415
    from ..utils.spin_operators import s_squared_qubit_op  # noqa: PLC0415
    from .distributed.dist_matvec_tile_resident import (  # noqa: PLC0415
        TileGPUResidentHamiltonian,
    )

    s2_op = getattr(ham, "_spin_penalty_s2", None)
    if s2_op is None:
        subspace = ham.spin_penalty_subspace
        operator = s_squared_qubit_op(int(subspace.width) // 2)
        operator.set_type(2)
        s2_op = TileGPUResidentHamiltonian(
            SubspaceHamiltonian(operator, subspace),
            subspace.alpha_strs,
            subspace.beta_strs,
            **ham.spin_penalty_kwargs,
        )
        ham._spin_penalty_s2 = s2_op
    v = np.ascontiguousarray(vector_local, dtype=np.float64)
    y = np.asarray(s2_op._matvec(v))
    comm = ham._comm
    return comm.allreduce(float(np.dot(v, y))) / comm.allreduce(float(np.dot(v, v)))
