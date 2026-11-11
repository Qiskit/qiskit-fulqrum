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

from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks
from fulqrum.gpu.eigensolver import primme_distributed as pd

N_LOCAL = 3


class _Stop(Exception):
    pass


def _operator(comm, device=False):
    size = comm.Get_size()
    return SimpleNamespace(
        shape=(N_LOCAL * size, N_LOCAL * size),
        n_local=N_LOCAL,
        _comm=comm,
        _matvec=lambda vector: vector,
        supports_device_matvec=device,
    )


def _stub_backend(monkeypatch, events=None):
    def eigsh(H, *args, **kwargs):
        if events is not None:
            events[H._comm.Get_rank()].append("solve")
        raise _Stop()

    backend = SimpleNamespace(eigsh=eigsh, gpu_backend_available=lambda: False)
    monkeypatch.setattr(pd, "_load_backend", lambda: backend)


def _solve(size, body):
    def run(comm):
        try:
            body(comm)
        except (ValueError, RuntimeError) as error:
            return type(error).__name__, str(error)
        except _Stop:
            return "stop", None
        return None

    return run_ranks(LocalComm.create(size, timeout=10.0), run)


def test_bad_preconditioner_on_one_rank_raises_on_every_rank(monkeypatch):
    _stub_backend(monkeypatch)

    def body(comm):
        length = N_LOCAL - 1 if comm.Get_rank() == 1 else N_LOCAL
        pd.distributed_primme_eigsh(
            _operator(comm), k=1, gpu=False, precond_diag_local=np.ones(length)
        )

    text = (
        "eigensolver setup failed: rank 1: precond_diag_local must have shape "
        f"({N_LOCAL},), got ({N_LOCAL - 1},)"
    )
    assert _solve(2, body) == [("ValueError", text)] * 2


def test_overlap_collective_runs_after_the_setup_agreement(monkeypatch):
    events = {0: [], 1: []}
    _stub_backend(monkeypatch, events)
    monkeypatch.setattr(pd, "_make_host_direct_matvec", lambda H, n: object())

    def overlap(H):
        events[H._comm.Get_rank()].append("overlap")
        H._comm.allgather(None)
        return False, "stubbed"

    monkeypatch.setattr(pd, "_host_overlap_mode", overlap)

    def healthy(comm):
        pd.distributed_primme_eigsh(_operator(comm, device=True), k=1, gpu=False)

    assert _solve(2, healthy) == [("stop", None)] * 2
    assert events == {0: ["overlap", "solve"], 1: ["overlap", "solve"]}

    events[0].clear()
    events[1].clear()

    def failing(comm):
        length = N_LOCAL - 1 if comm.Get_rank() == 1 else N_LOCAL
        pd.distributed_primme_eigsh(
            _operator(comm, device=True),
            k=1,
            gpu=False,
            precond_diag_local=np.ones(length),
        )

    results = _solve(2, failing)
    assert [kind for kind, _ in results] == ["ValueError"] * 2
    assert events == {0: [], 1: []}


def test_leg_disagreement_raises_on_every_rank(monkeypatch):
    _stub_backend(monkeypatch)
    monkeypatch.setattr(
        pd, "choose_leg", lambda H, gpu, backend: H._comm.Get_rank() == 0
    )
    monkeypatch.setattr(
        pd,
        "_gpu_leg_callbacks",
        lambda H, n, options, precond, v0: pd.SolverCallbacks(None, None),
    )

    def body(comm):
        pd.distributed_primme_eigsh(_operator(comm), k=1, gpu="auto")

    text = "eigensolver leg differs across ranks: rank 0: GPU leg; rank 1: host leg"
    assert _solve(2, body) == [("RuntimeError", text)] * 2


def test_native_setup_failure_on_one_rank_raises_on_every_rank(monkeypatch):
    native = pytest.importorskip("fulqrum.gpu.eigensolver._primme_mpi")
    pytest.importorskip("mpi4py")
    monkeypatch.setattr(pd, "_load_backend", lambda: native)

    def fault(H):
        if H._comm.Get_rank() == 1:
            return MemoryError("injected setup failure")
        return None

    monkeypatch.setattr(native, "_setup_fault_hook", fault)

    def body(comm):
        pd.distributed_primme_eigsh(_operator(comm), k=1, gpu=False)

    text = "eigensolver setup failed: rank 1: MemoryError: injected setup failure"
    assert _solve(2, body) == [("RuntimeError", text)] * 2
