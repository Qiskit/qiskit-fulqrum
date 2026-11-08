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


import logging
import threading
import time
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fulqrum.gpu.distributed.local_comm import LocalComm
from fulqrum.gpu.eigensolver import primme_distributed as pd


class _Duplicate:

    def __init__(self):
        self.freed = 0

    def Free(self):  # noqa: N802
        self.freed += 1


class _Comm:

    def __init__(self):
        self.duplicates = []

    def Get_rank(self):  # noqa: N802
        return 0

    def Get_size(self):  # noqa: N802
        return 2

    def allgather(self, value):
        return [value, value]

    def Dup(self):  # noqa: N802
        duplicate = _Duplicate()
        self.duplicates.append(duplicate)
        return duplicate


def _operator(comm):
    return SimpleNamespace(
        shape=(6, 6),
        n_local=3,
        _comm=comm,
        _matvec=lambda vector: vector,
        supports_device_matvec=True,
    )


@pytest.mark.parametrize("fails", [False, True], ids=["returns", "raises"])
def test_the_overlap_communicator_is_freed_once(monkeypatch, fails):
    comm = _Comm()

    def eigsh(H, *args, **kwargs):
        assert len(comm.duplicates) == 1 and comm.duplicates[0].freed == 0
        if fails:
            raise RuntimeError("solve failed")
        return np.zeros(1)

    backend = SimpleNamespace(eigsh=eigsh, gpu_backend_available=lambda: False)
    monkeypatch.setattr(pd, "_load_backend", lambda: backend)
    monkeypatch.setattr(pd, "_make_host_direct_matvec", lambda H, n: object())
    monkeypatch.setattr(pd, "_host_overlap_mode", lambda H: (True, "stubbed"))
    monkeypatch.setattr(pd, "_GuardedOverlap", lambda inner, comm: inner)
    monkeypatch.setattr(pd, "_abort_on_rank_error", lambda fn, comm, what: fn)
    H = _operator(comm)
    if fails:
        with pytest.raises(RuntimeError, match="solve failed"):
            pd.distributed_primme_eigsh(H, k=1, gpu=False)
    else:
        pd.distributed_primme_eigsh(H, k=1, gpu=False)
    assert [duplicate.freed for duplicate in comm.duplicates] == [1]


@pytest.mark.parametrize("setting, expected", [("on", True), ("0", False)])
def test_host_gpu_preconditioner_follows_the_config(
    monkeypatch, setting, expected
):
    seen = {}

    def precond(diagonal, n_local, tile):
        seen["tile"] = tile
        return "gpu precond"

    pool = SimpleNamespace(free_all_blocks=lambda: None)
    fake_cupy = SimpleNamespace(get_default_memory_pool=lambda: pool)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(pd, "_make_host_gpu_precond", precond)
    monkeypatch.setenv("FQ_PRIMME_GPU_PRECOND", setting)
    monkeypatch.setenv("FQ_PRIMME_GPU_PRECOND_TILE", "4096")
    H = SimpleNamespace(_comm=LocalComm.create(1)[0])
    callbacks = pd._host_leg_local(H, 3, np.ones(3), None)
    assert (callbacks.gpu_precond == "gpu precond") is expected
    assert seen == ({"tile": 4096} if expected else {})


def test_invalid_preconditioner_setting_is_rejected(monkeypatch):
    monkeypatch.setenv("FQ_PRIMME_GPU_PRECOND", "maybe")
    H = SimpleNamespace(_comm=LocalComm.create(1)[0])
    with pytest.raises(ValueError, match="FQ_PRIMME_GPU_PRECOND"):
        pd._host_leg_local(H, 3, np.ones(3), None)


def test_gpu_preconditioner_reports_that_it_disables_spec(monkeypatch):
    pool = SimpleNamespace(free_all_blocks=lambda: None)
    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(get_default_memory_pool=lambda: pool)
    )
    monkeypatch.setattr(pd, "_make_host_gpu_precond", lambda d, n, t: "gpu precond")
    monkeypatch.setattr(pd, "_make_host_direct_matvec", lambda H, n: object())
    monkeypatch.setattr(pd, "_host_overlap_mode", lambda H: (True, "stubbed"))
    monkeypatch.setenv("FQ_PRIMME_GPU_PRECOND", "1")
    monkeypatch.setenv("FQ_PRIMME_SPEC_MATVEC", "1")
    H = _operator(LocalComm.create(1)[0])
    callbacks = pd._host_leg_local(H, 3, np.ones(3), None)
    records = []
    handler = logging.Handler(level=logging.INFO)
    handler.emit = records.append
    level = pd.logger.level
    pd.logger.addHandler(handler)
    pd.logger.setLevel(logging.INFO)
    try:
        pd._host_leg_collective(H, callbacks)
    finally:
        pd.logger.removeHandler(handler)
        pd.logger.setLevel(level)
    messages = [record.getMessage() for record in records]
    assert any("disables speculative matvecs" in m for m in messages), messages
    assert any("spec_matvec=off (GPU preconditioner)" in m for m in messages)
    assert callbacks.spec_matvec is True


class _SlowMatvec:

    def __init__(self, seconds):
        self.seconds = seconds
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=time.sleep, args=(self.seconds,))
        self.thread.start()

    def close(self, timeout=None):
        self.thread.join(timeout)
        return not self.thread.is_alive()


class _AbortRecorder:

    def __init__(self, size):
        self.size = size
        self.aborts = []

    def Get_rank(self):  # noqa: N802
        return 0

    def Get_size(self):  # noqa: N802
        return self.size

    def Abort(self, errorcode=1):  # noqa: N802
        self.aborts.append(errorcode)


def _guarded(inner, comm):
    inner.input_done = inner.wait = inner.stats = inner.dot = lambda *a: None
    return pd._GuardedOverlap(inner, comm)


def test_a_stuck_overlap_thread_aborts_a_distributed_job(monkeypatch, capsys):
    monkeypatch.setenv("FQ_PRIMME_CLOSE_TIMEOUT_S", "0.2")
    inner, comm = _SlowMatvec(2.0), _AbortRecorder(2)
    guarded = _guarded(inner, comm)
    inner.start()
    started = time.perf_counter()
    guarded.close()
    assert time.perf_counter() - started < 1.5
    assert comm.aborts == [1]
    assert "did not finish on rank 0 within 0.2 s" in capsys.readouterr().err
    inner.thread.join()


def test_a_finished_overlap_thread_closes_without_aborting(monkeypatch):
    monkeypatch.setenv("FQ_PRIMME_CLOSE_TIMEOUT_S", "5")
    inner, comm = _SlowMatvec(0.05), _AbortRecorder(2)
    guarded = _guarded(inner, comm)
    inner.start()
    guarded.close()
    assert comm.aborts == []


def test_a_single_rank_close_waits_for_the_thread(monkeypatch):
    monkeypatch.setenv("FQ_PRIMME_CLOSE_TIMEOUT_S", "0.01")
    inner, comm = _SlowMatvec(0.3), _AbortRecorder(1)
    guarded = _guarded(inner, comm)
    inner.start()
    guarded.close()
    assert comm.aborts == [] and not inner.thread.is_alive()


def test_host_direct_close_reports_a_running_thread():
    direct = object.__new__(pd._HostDirectMatvec)
    direct._thread = threading.Thread(target=time.sleep, args=(0.5,))
    direct._thread.start()
    assert direct.close(timeout=0.01) is False and direct._thread is not None
    assert direct.close() is True and direct._thread is None
