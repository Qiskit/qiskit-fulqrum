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

"""Spin operators and spin penalties (fulqrum.utils.spin_operators) against PySCF."""

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.convert import integrals_to_fq_fermionic_op
from fulqrum.utils import (
    build_spin_separated_half_strings,
    make_spin_penalized_csr,
    make_spin_penalized_fermionic_op,
    make_spin_penalized_operator,
    s_squared_fermionic_op,
    s_squared_qubit_op,
    spin_penalty_expectations,
    split_alpha_beta,
    use_quadratic_penalty,
)

pyscf = pytest.importorskip("pyscf")
from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from pyscf.fci import cistring, spin_op  # noqa: E402

LAMBDA = 0.2
# (orbitals, (N_alpha, N_beta)), S_z = 0 and S_z != 0
SECTORS = [(4, (2, 2)), (4, (3, 1)), (4, (2, 1)), (6, (3, 3)), (6, (4, 2)), (6, (3, 2))]


def sector_strings(norb, electrons):
    return [
        format(int(s), f"0{norb}b")
        for s in cistring.make_strings(range(norb), electrons)
    ]


def dense(qubit_op, norb, nelec):
    """The operator on the full sector; SubspaceHamiltonian.matvec includes identity terms."""
    subspace = fq.Subspace(
        [sector_strings(norb, nelec[0]), sector_strings(norb, nelec[1])]
    )
    hsub = fq.SubspaceHamiltonian(qubit_op, subspace)
    dim = hsub.shape[0]
    return np.array([np.asarray(hsub.matvec(e)).ravel() for e in np.eye(dim)]).T


def pyscf_s_squared(norb, nelec):
    na, nb = (cistring.num_strings(norb, k) for k in nelec)
    out = np.zeros((na * nb, na * nb))
    for k in range(na * nb):
        ci = np.zeros((na, nb))
        ci[k % na, k // na] = 1.0  # Fulqrum's index is i_beta * n_alpha + i_alpha
        out[:, k] = np.asarray(spin_op.contract_ss(ci, norb, nelec)).T.ravel()
    return out


def chain(n_atoms):
    mol = gto.M(
        atom="; ".join(f"H 0 0 {0.9 * k}" for k in range(n_atoms)),
        basis="sto-3g",
        spin=n_atoms % 2,
        verbose=0,
    )
    mf = (scf.ROHF if n_atoms % 2 else scf.RHF)(mol).run()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), mol.nao)
    return np.ascontiguousarray(h1), np.ascontiguousarray(h2)


def qubit(fermionic_op, op_type):
    out = fermionic_op.extended_jw_transformation()
    out.set_type(op_type)
    return out


@pytest.mark.parametrize("op_type", [1, 2])
@pytest.mark.parametrize("norb,nelec", SECTORS)
def test_s_squared_matches_pyscf(norb, nelec, op_type):
    np.testing.assert_allclose(
        dense(s_squared_qubit_op(norb, op_type), norb, nelec),
        pyscf_s_squared(norb, nelec),
        rtol=0,
        atol=1e-12,
    )


def test_operator_products_set_the_derived_term_fields():
    s2 = s_squared_fermionic_op(4)
    d = s2.to_dict()
    rebuilt = fq.FermionicOperator(
        d["width"], [(vals, inds, complex(*c)) for vals, inds, c in d["terms"]]
    )
    # offdiag_structure drives term grouping; the Python constructor sets it (but not
    # the projector fields, which the C++ string constructor and multiply also set).
    for name in ("offdiag_structures", "weights"):
        np.testing.assert_array_equal(getattr(s2, name)(), getattr(rebuilt, name)())
    assert np.count_nonzero(s2.offdiag_structures()) > 0


@pytest.mark.parametrize("coeff", [0.0, -0.0])
def test_zero_identity_terms_keep_their_coefficient(coeff):
    op = fq.FermionicOperator(4, [("", [], coeff)])
    assert op.to_dict()["terms"] == [["", [], (0.0, 0.0)]]


@pytest.mark.parametrize("lam", [0.1, 0.2])
@pytest.mark.parametrize("op_type", [1, 2])
@pytest.mark.parametrize(
    "n_atoms,nelec,target_s",
    [(4, (2, 2), 0.0), (5, (3, 2), 0.5), (4, (3, 1), 1.0), (6, (4, 2), 1.0)],
)
def test_linear_penalty_matches_pyscf(n_atoms, nelec, target_s, op_type, lam):
    assert not use_quadratic_penalty(nelec[0], nelec[1], target_s)
    ss = target_s * (target_s + 1)
    h1, h2 = chain(n_atoms)
    solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(), shift=lam, ss=ss)
    solver.conv_tol = 1e-12
    e_ref, c_ref = solver.kernel(h1, h2, n_atoms, nelec)
    penalised = make_spin_penalized_fermionic_op(
        integrals_to_fq_fermionic_op(h1, h2),
        n_atoms,
        nelec[0],
        nelec[1],
        target_s,
        lam,
    )
    w, v = np.linalg.eigh(dense(qubit(penalised, op_type), n_atoms, nelec))
    assert abs(w[0] - e_ref) < 1e-10
    s2 = v[:, 0] @ dense(s_squared_qubit_op(n_atoms, op_type), n_atoms, nelec) @ v[:, 0]
    assert abs(s2 - ss) < 1e-8
    assert abs(s2 - spin_op.spin_square(c_ref, n_atoms, nelec)[0]) < 1e-8


@pytest.mark.parametrize("op_type", [1, 2])
def test_quadratic_penalty_selects_the_triplet_at_sz_zero(op_type):
    assert use_quadratic_penalty(2, 2, 1.0)
    h1, h2 = chain(4)
    h_op = integrals_to_fq_fermionic_op(h1, h2)
    # the lowest S_z = 1 state is the lowest triplet
    triplet = np.linalg.eigvalsh(dense(qubit(h_op, op_type), 4, (3, 1)))[0]
    penalised = make_spin_penalized_fermionic_op(h_op, 4, 2, 2, 1.0, LAMBDA)
    w, v = np.linalg.eigh(dense(qubit(penalised, op_type), 4, (2, 2)))
    assert abs(w[0] - triplet) < 1e-10
    s2 = v[:, 0] @ dense(s_squared_qubit_op(4, op_type), 4, (2, 2)) @ v[:, 0]
    assert abs(s2 - 2.0) < 1e-8


@pytest.mark.parametrize(
    "target_s,nelec", [(0.0, (2, 2)), (1.0, (2, 2)), (1.0, (3, 1))]
)
def test_linear_operator_and_csr_forms_match_the_operator_form(target_s, nelec):
    h1, h2 = chain(4)
    h_op = integrals_to_fq_fermionic_op(h1, h2)
    subspace = fq.Subspace([sector_strings(4, nelec[0]), sector_strings(4, nelec[1])])
    h_sub = fq.SubspaceHamiltonian(qubit(h_op, 2), subspace)
    s2_sub = fq.SubspaceHamiltonian(s_squared_qubit_op(4), subspace)
    expected = dense(
        qubit(make_spin_penalized_fermionic_op(h_op, 4, *nelec, target_s, LAMBDA), 2),
        4,
        nelec,
    )
    linear = make_spin_penalized_operator(h_sub, s2_sub, *nelec, target_s, LAMBDA)
    dim = h_sub.shape[0]
    columns = np.array([linear.matvec(e) for e in np.eye(dim)]).T
    np.testing.assert_allclose(columns, expected, rtol=0, atol=1e-12)
    csr = make_spin_penalized_csr(
        h_sub.to_csr_linearoperator(),
        s2_sub.to_csr_linearoperator(),
        *nelec,
        target_s,
        LAMBDA,
    )
    np.testing.assert_allclose(csr.toarray(), expected, rtol=0, atol=1e-12)


def test_split_alpha_beta_keeps_first_appearance_order():
    alpha, beta = split_alpha_beta(["01100011", "10100101", "01100011"])
    assert list(alpha) == ["0011", "0101"]
    assert list(beta) == ["0110", "1010"]
    assert split_alpha_beta([]) == ({}, {})


@pytest.mark.parametrize("pool_by_weight", [True, False])
def test_half_strings_start_from_hartree_fock(pool_by_weight):
    alpha, beta = build_spin_separated_half_strings(
        ["01010011"], ["00111100"], 4, 2, 2, pool_by_weight=pool_by_weight
    )
    assert next(iter(alpha)) == "0011" and next(iter(beta)) == "0011"
    if pool_by_weight:
        # every half with the right weight goes to both spins
        assert set(alpha) == set(beta) == {"0011", "0101", "1100"}
    else:
        # halves stay on the side they appeared on (alpha = right, beta = left)
        assert set(alpha) == {"0011", "1100"}
        assert set(beta) == {"0011", "0101"}


def explicit_s_squared(norb):
    """S^2 term by term (any S_z), independent of the operator-product builder."""
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
    for i in range(n):
        for j in range(n):
            for p, sp in ((i, 1.0), (i + n, -1.0)):
                for q, sq in ((j, 1.0), (j + n, -1.0)):
                    vals, inds = ("1", [p]) if p == q else ("11", [p, q])
                    terms.append((vals, inds, 0.25 * sp * sq))
    return fq.FermionicOperator(2 * n, terms)


@pytest.mark.parametrize("lam", [0.1, 0.2])
@pytest.mark.parametrize(
    "n_atoms,nelec", [(4, (2, 2)), (5, (3, 2)), (4, (3, 1)), (6, (4, 2))]
)
def test_default_target_is_the_lowest_spin_and_matches_a_hand_built_penalty(
    n_atoms, nelec, lam
):
    spin = 0.5 * (nelec[0] - nelec[1])
    h_op = integrals_to_fq_fermionic_op(*chain(n_atoms))
    default = make_spin_penalized_fermionic_op(h_op, n_atoms, *nelec, lam=lam)
    explicit = make_spin_penalized_fermionic_op(h_op, n_atoms, *nelec, spin, lam)
    by_hand = (
        h_op
        + explicit_s_squared(n_atoms) * lam
        + fq.FermionicOperator(2 * n_atoms, [("", [], -lam * spin * (spin + 1))])
    )
    expected = dense(qubit(by_hand, 2), n_atoms, nelec)
    for op in (default, explicit):
        np.testing.assert_allclose(
            dense(qubit(op, 2), n_atoms, nelec), expected, rtol=0, atol=1e-12
        )


@pytest.mark.parametrize(
    "n_atoms,nelec,target_s",
    [(4, (2, 2), None), (5, (3, 2), None), (6, (4, 2), None), (4, (2, 2), 1.0)],
)
def test_expectations_give_the_energy_and_spin(n_atoms, nelec, target_s):
    h1, h2 = chain(n_atoms)
    h_op = integrals_to_fq_fermionic_op(h1, h2)
    spin = 0.5 * (nelec[0] - nelec[1]) if target_s is None else target_s
    penalised = make_spin_penalized_fermionic_op(
        h_op, n_atoms, *nelec, target_s, LAMBDA
    )
    w, v = np.linalg.eigh(dense(qubit(penalised, 2), n_atoms, nelec))
    subspace = fq.Subspace(
        [sector_strings(n_atoms, nelec[0]), sector_strings(n_atoms, nelec[1])]
    )
    out = spin_penalty_expectations(w[0], v[:, 0], subspace, *nelec, target_s, LAMBDA)
    h_dense = dense(qubit(h_op, 2), n_atoms, nelec)
    assert abs(out["s2"] - spin * (spin + 1)) < 1e-8
    assert abs(out["energy"] - v[:, 0] @ h_dense @ v[:, 0]) < 1e-10
    if target_s is None:
        solver = fci.addons.fix_spin_(
            fci.direct_spin1.FCI(), shift=LAMBDA, ss=spin * (spin + 1)
        )
        solver.conv_tol = 1e-12
        e_ref, c_ref = solver.kernel(h1, h2, n_atoms, nelec)
        assert abs(out["s2"] - spin_op.spin_square(c_ref, n_atoms, nelec)[0]) < 1e-8
        # linear: e = <H> at an exact spin state
        assert abs(out["energy"] - e_ref) < 1e-10
    else:
        # quadratic, triplet at S_z = 0: the energy is the lowest S_z = 1 energy
        triplet = np.linalg.eigvalsh(dense(qubit(h_op, 2), n_atoms, (3, 1)))[0]
        assert abs(out["energy"] - triplet) < 1e-10
