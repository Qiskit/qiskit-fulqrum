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


from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fulqrum.gpu.eigensolver._primme_mpi")
MPI = pytest.importorskip("mpi4py.MPI")

from fulqrum.gpu.eigensolver.primme_distributed import (  # noqa: E402
    distributed_primme_eigsh,
)

DIAGONAL = np.array(
    [-9.0, -4.0, -1.5, -0.25, 0.1, 0.4, 2.0, 3.0, 6.0, 10.0]
    + list(np.linspace(11.0, 40.0, 30))
)


def _operator(diagonal=DIAGONAL):
    n = diagonal.size
    return SimpleNamespace(
        shape=(n, n),
        n_local=n,
        _comm=MPI.COMM_SELF,
        _matvec=lambda vector: diagonal * vector,
    )


@pytest.mark.parametrize("which", ["SM", "LM"])
def test_magnitude_targets_match_numpy(which):
    evals, _ = distributed_primme_eigsh(
        _operator(), k=2, which=which, tol=1e-10, gpu=False
    )
    order = np.argsort(np.abs(DIAGONAL))
    picked = order[:2] if which == "SM" else order[-2:]
    np.testing.assert_allclose(
        np.sort(evals), np.sort(DIAGONAL[picked]), rtol=0, atol=1e-8
    )


@pytest.mark.parametrize(
    "which, maxiter, all_missing",
    [("SA", 12, False), ("SM", 12, True), ("SA", 4, True)],
    ids=["soft_locking", "locking", "budget_before_first_ritz_values"],
)
def test_unconverged_pairs_are_nan(which, maxiter, all_missing):
    evals, evecs, stats = distributed_primme_eigsh(
        _operator(),
        k=3,
        which=which,
        tol=1e-14,
        maxiter=maxiter,
        return_unconverged=True,
        return_stats=True,
        gpu=False,
    )
    assert stats["nconv"] < 3
    missing = np.isnan(evals)
    if all_missing:
        assert missing.all() and np.isnan(evecs).all()
    else:
        assert not missing.any()
        assert np.isfinite(evecs).all()
    assert missing.sum() == 3 - stats["nconv"] or not all_missing


class _Aborted(Exception):
    pass


class _Ranks:

    def __init__(self, size):
        self.size = size
        self.aborts = []

    def Get_rank(self):  # noqa: N802
        return 0

    def Get_size(self):  # noqa: N802
        return self.size

    def Abort(self, errorcode=1):  # noqa: N802
        self.aborts.append(errorcode)
        raise _Aborted()


@pytest.mark.parametrize("size, aborts", [(1, []), (2, [1])])
def test_a_failing_callback_aborts_only_a_distributed_solve(size, aborts, capsys):
    from fulqrum.gpu.eigensolver import _primme_mpi  # noqa: PLC0415

    comm = _Ranks(size)
    H = SimpleNamespace(n_local=1, _comm=comm)
    _primme_mpi._test_abort_if_distributed(H, "matvec")
    assert comm.aborts == aborts
    if aborts:
        assert "PRIMME matvec callback failed on rank 0" in capsys.readouterr().err
