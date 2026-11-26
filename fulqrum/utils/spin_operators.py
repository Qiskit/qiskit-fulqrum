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

"""Spin operators for imposing spin constraints in SQD calculations.

Inspired by PySCF's ``fci.addons.fix_spin_``.

Two penalty regimes are supported, matching PySCF:

* Linear: H + lambda * (S^2 - s(s+1)).
* Quadratic: H + lambda * (S^2 - s(s+1))^2`.

The penalized operator can be built in either of the two forms used for
eigensolving in Fulqrum:

* `make_spin_penalized_operator` returns a
  `scipy.sparse.linalg.LinearOperator` (applies `S^2` twice
  for the quadratic case, so no operator squaring is needed). Both
  ``SubspaceHamiltonian`` and
  its ``to_csr_linearoperator_fast()`` CSR form work.
* `make_spin_penalized_csr` returns an explicit ``scipy.sparse`` CSR matrix;
  in the quadratic case it squares the `S^2` matrix.

``target_s=None`` (the default) targets the lowest spin of the electron
sector, ``|num_a - num_b| / 2``. `spin_penalty_expectations` gives `<S^2>`
and `<H>` of a penalized eigenvector.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from ..core.fermi_operator import FermionicOperator


def s_z_fermionic_op(num_orbitals):
    n = num_orbitals
    terms = [("1", [p], 0.5) for p in range(n)]  # + 1/2 n_{pa}
    terms += [("1", [p + n], -0.5) for p in range(n)]  # - 1/2 n_{pb}
    return FermionicOperator(2 * n, terms)


def s_plus_fermionic_op(num_orbitals):
    n = num_orbitals
    return FermionicOperator(2 * n, [("+-", [p, p + n], 1.0) for p in range(n)])


def s_minus_fermionic_op(num_orbitals):
    n = num_orbitals
    return FermionicOperator(2 * n, [("+-", [p + n, p], 1.0) for p in range(n)])


def s_squared_fermionic_op(num_orbitals):
    s_z = s_z_fermionic_op(num_orbitals)
    s_plus = s_plus_fermionic_op(num_orbitals)
    s_minus = s_minus_fermionic_op(num_orbitals)
    return s_minus @ s_plus + s_z @ s_z + s_z


def s_squared_qubit_op(num_orbitals, operator_type=2):
    """Build the `S^2` qubit operator on ``2 * num_orbitals`` modes.

    Parameters:
        num_orbitals (int): Number of spatial orbitals ``N``.
        operator_type (int): Fulqrum operator type, 1 or 2. Both give the same
            matrix on a subspace; the GPU operators need type 2 (the default).

    Returns:
        QubitOperator: `S^2` after the extended Jordan-Wigner transformation.
    """
    op = s_squared_fermionic_op(num_orbitals).extended_jw_transformation()
    op.set_type(operator_type)
    return op


def make_spin_penalized_fermionic_op(
    hamiltonian,
    num_orbitals,
    num_a,
    num_b,
    target_s=None,
    lam=0.2,
):
    """Build a spin-penalized Hamiltonian as a single `FermionicOperator`.

    Parameters:
        hamiltonian (FermionicOperator): Hamiltonian.
        num_orbitals (int): Number of spatial orbitals ``N``.
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float or None): Target total spin quantum number ``s``. ``None``
            (default) targets the lowest spin of the sector,
            ``s = |num_a - num_b| / 2``, which uses the linear penalty.
        lam (float): Penalty weight `lambda`. Defaults to ``0.2``.

    Returns:
        FermionicOperator: ``hamiltonian + lam * P(S^2)``.
    """
    target_s = _target_spin(num_a, num_b, target_s)
    num_qubits = 2 * num_orbitals
    s2 = s_squared_fermionic_op(num_orbitals)
    ss = target_s * (target_s + 1.0)
    # (S^2 - ss I) as a fermionic operator (identity is the empty-string term).
    dev = s2 + FermionicOperator(num_qubits, [("", [], -ss)])
    penalty = dev @ dev if use_quadratic_penalty(num_a, num_b, target_s) else dev
    return hamiltonian + lam * penalty


def use_quadratic_penalty(num_a, num_b, target_s):
    r"""Decide whether the quadratic spin penalty is required.

    Mirrors PySCF's ``fix_spin`` logic.

    Parameters:
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float): Target total spin quantum number ``s``.

    Returns:
        bool: ``True`` if the quadratic penalty is required.
    """
    sz = abs(num_a - num_b) * 0.5
    ss = target_s * (target_s + 1.0)
    # PySCF uses the linear form when ss < sz*(sz+1) + 0.1 (target is the lowest
    # spin of the sector); otherwise the quadratic form.
    return ss >= sz * (sz + 1.0) + 0.1


def make_spin_penalized_operator(
    h_operator,
    s2_operator,
    num_a,
    num_b,
    target_s=None,
    lam=0.2,
):
    r"""Compose a spin-penalized Hamiltonian as a ``LinearOperator``.

    Both ``h_operator`` and ``s2_operator`` must be projected onto the *same*
    subspace and expose a ``matvec`` and a matching ``shape``.

    Parameters:
        h_operator: Projected molecular Hamiltonian (has ``matvec`` / ``shape``).
        s2_operator: Projected `S^2` operator on the same subspace.
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float or None): Target total spin quantum number ``s``; ``None``
            (default) is the lowest spin of the sector (e.g. ``0`` for a
            singlet, ``0.5`` doublet, ``1`` triplet).
        lam (float): Penalty weight. Defaults to ``0.2`` (the
            value used in arXiv:2411.04827; PySCF's ``fix_spin_`` uses 0.1-0.2).
            A higher value enforces the constraint more strongly.

    Returns:
        scipy.sparse.linalg.LinearOperator: The spin-penalized operator.
    """
    target_s = _target_spin(num_a, num_b, target_s)
    if h_operator.shape != s2_operator.shape:
        raise ValueError(
            "h_operator and s2_operator must act on the same subspace; got shapes "
            f"{h_operator.shape} and {s2_operator.shape}."
        )

    ss = target_s * (target_s + 1.0)
    quadratic = use_quadratic_penalty(num_a, num_b, target_s)
    dim = h_operator.shape[0]
    # The penalty mixes a complex S^2 action into a possibly-real H; keep the
    # composite complex unless both inputs are real.
    is_real = getattr(h_operator, "dtype", np.dtype(complex)) == np.dtype(
        float
    ) and getattr(s2_operator, "dtype", np.dtype(complex)) == np.dtype(float)
    dtype = np.dtype(float) if is_real else np.dtype(complex)

    def matvec(x):
        x = np.asarray(x).reshape(dim)
        hv = h_operator.matvec(x)
        dev = s2_operator.matvec(x) - ss * x  # (S^2 - ss) v
        if quadratic:
            penalty = s2_operator.matvec(dev) - ss * dev  # (S^2 - ss)^2 v
        else:
            penalty = dev
        return np.asarray(hv).reshape(dim) + lam * penalty

    return LinearOperator((dim, dim), matvec=matvec, dtype=dtype)


def make_spin_penalized_csr(
    h_csr,
    s2_csr,
    num_a,
    num_b,
    target_s=None,
    lam=0.2,
):
    """Build a spin-penalized Hamiltonian as an explicit CSR matrix.

    The explicit form of `make_spin_penalized_operator`: ``H + lam * (S^2 - ss)``
    (linear) or ``H + lam * (S^2 - ss)^2`` (quadratic), on the subspace of both
    matrices. The quadratic form squares the `S^2` matrix, which can fill in
    many more entries than `S^2` itself; the matrix-free
    `make_spin_penalized_operator` avoids that.

    Parameters:
        h_csr: Projected Hamiltonian, a ``scipy.sparse`` matrix or a Fulqrum
            ``CSRLinearOperator`` (``SubspaceHamiltonian.to_csr_linearoperator()``).
        s2_csr: Projected `S^2` on the same subspace, in the same form.
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float or None): Target total spin quantum number ``s``. ``None``
            (default) targets the lowest spin of the sector,
            ``s = |num_a - num_b| / 2``, which uses the linear penalty.
        lam (float): Penalty weight `lambda`. Defaults to ``0.2``.

    Returns:
        scipy.sparse.csr_matrix: The penalized matrix.
    """
    target_s = _target_spin(num_a, num_b, target_s)
    # Accept either a raw scipy sparse matrix or a Fulqrum CSRLinearOperator.
    h_mat = h_csr.matrix if hasattr(h_csr, "matrix") else h_csr
    s2_mat = s2_csr.matrix if hasattr(s2_csr, "matrix") else s2_csr
    if h_mat.shape != s2_mat.shape:
        raise ValueError(
            "h_csr and s2_csr must act on the same subspace; got shapes "
            f"{h_mat.shape} and {s2_mat.shape}."
        )

    ss = target_s * (target_s + 1.0)
    quadratic = use_quadratic_penalty(num_a, num_b, target_s)

    identity = sp.identity(h_mat.shape[0], dtype=s2_mat.dtype, format="csr")
    dev = sp.csr_matrix(s2_mat) - ss * identity  # (S^2 - ss I)
    penalty = dev @ dev if quadratic else dev

    return (sp.csr_matrix(h_mat) + lam * penalty).tocsr()


def _target_spin(num_a, num_b, target_s):
    """The target spin: the given one, or the lowest spin of the sector."""
    return abs(num_a - num_b) * 0.5 if target_s is None else target_s


def spin_penalty_expectations(
    eigenvalue,
    vector,
    subspace,
    num_a,
    num_b,
    target_s=None,
    lam=0.2,
):
    r"""`<S^2>` and `<H>` of an eigenvector of a spin-penalized Hamiltonian.

    The eigenvalue of ``H + lam * P`` (``P`` the linear or quadratic penalty, as
    in `make_spin_penalized_fermionic_op`) is not the energy; this returns
    `<S^2>` and `<H> = eigenvalue - lam * <P>` of the eigenvector.

    Parameters:
        eigenvalue (float): Eigenvalue of the penalized operator for ``vector``.
        vector (ndarray): The eigenvector, in the order of ``subspace``.
        subspace (Subspace): The alpha x beta subspace of the calculation.
        num_a (int): Number of spin-up (alpha) electrons.
        num_b (int): Number of spin-down (beta) electrons.
        target_s (float or None): Target spin; ``None`` is the lowest spin of the sector.
        lam (float): Penalty weight `lambda`. Defaults to ``0.2``.

    Returns:
        dict: ``{"s2": <S^2>, "energy": <H>}``.
    """
    from ..core.linear_operator import SubspaceHamiltonian

    target_s = _target_spin(num_a, num_b, target_s)
    ss = target_s * (target_s + 1.0)
    s2 = SubspaceHamiltonian(s_squared_qubit_op(subspace.width // 2), subspace)
    v = np.asarray(vector).ravel()
    norm = np.vdot(v, v).real
    dev = np.asarray(s2.matvec(v)).ravel() - ss * v
    s2_value = np.vdot(v, dev).real / norm + ss
    if use_quadratic_penalty(num_a, num_b, target_s):
        penalty = np.vdot(dev, dev).real / norm
    else:
        penalty = s2_value - ss
    return {"s2": float(s2_value), "energy": float(eigenvalue - lam * penalty)}
