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


import sys
import threading
from types import SimpleNamespace

import numpy as np

from fulqrum.gpu.distributed.backends import _ipc
from fulqrum.gpu.distributed.backends import host_mpi
from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks


def test_ring_shift_skips_zero_length_blocks(monkeypatch):
    seen = []
    backend = host_mpi._HostMpiBackend()
    monkeypatch.setattr(
        backend,
        "_transfer",
        lambda comm, sends, receives, stream, tag: seen.append((sends, receives)),
    )
    empty = np.zeros(0)
    full = np.zeros(4)
    backend.ring_shift_device(empty, full, 1, 2, comm=None)
    backend.ring_shift_device(full, empty, 1, 2, comm=None)
    backend.ring_shift_device(empty, empty, 1, 2, comm=None)
    assert [(len(s), len(r)) for s, r in seen] == [(0, 1), (1, 0), (0, 0)]
    assert seen[0][1][0][3] == 4 and seen[1][0][0][3] == 4


def test_ring_shift_with_no_lanes_does_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    backend = host_mpi._HostMpiBackend()
    backend._transfer(None, [], [], None, 0)


def test_a_bad_ipc_setting_on_one_rank_raises_on_every_rank(monkeypatch):
    from fulqrum.gpu import config  # noqa: PLC0415

    real = config.read

    current = threading.local()

    def read(name, environ=None, value=None):
        if name == "FQ_HOST_MPI_USE_CUDA_IPC" and current.rank == 1:
            raise ValueError("FQ_HOST_MPI_USE_CUDA_IPC='maybe': use one of ...")
        return real(name, environ, value)

    monkeypatch.setattr(config, "read", read)

    def body(comm):
        current.rank = comm.Get_rank()
        try:
            _ipc.IpcPeers.discover(comm)
        except ValueError as error:
            return str(error)
        return None

    comms = LocalComm.create(2, timeout=10.0)
    messages = run_ranks(comms, body)
    assert all(m and "rank 1: FQ_HOST_MPI_USE_CUDA_IPC" in m for m in messages)


def test_backend_prepare_defaults_to_a_no_op():
    from fulqrum.gpu.distributed._collective import _Backend  # noqa: PLC0415

    assert _Backend().prepare([object(), None]) is None


def test_nccl_prepare_creates_every_sub_communicator(monkeypatch):
    from fulqrum.gpu.distributed.backends import nccl  # noqa: PLC0415

    created = []
    backend = nccl._NcclBackend()
    monkeypatch.setattr(backend, "_get_nccl_for", created.append)
    first, second = object(), object()
    backend.prepare([first, None, second])
    assert created == [first, second]
