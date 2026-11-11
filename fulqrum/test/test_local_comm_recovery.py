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


import threading
import time

import pytest

from fulqrum.gpu.distributed.local_comm import (
    LocalComm,
    LocalCommAborted,
    run_ranks,
)


def test_a_take_timeout_releases_peers_waiting_in_a_barrier():
    comms = LocalComm.create(3, timeout=2.0)
    finished = {}
    started = time.perf_counter()

    def body(comm):
        rank = comm.Get_rank()
        try:
            if rank == 0:
                comm.take(1)
            else:
                time.sleep(1.0)
                comm.allgather(rank)
        finally:
            finished[rank] = time.perf_counter() - started

    with pytest.raises(LocalCommAborted):
        run_ranks(comms, body)
    assert max(finished.values()) < 2.5, finished


def _work(comm):
    child = comm.Split(color=comm.Get_rank() % 2, key=comm.Get_rank())
    return comm.allgather(comm.Get_rank()), child.allgather(child.Get_rank())


def test_comms_are_usable_after_a_failed_run():
    comms = LocalComm.create(3, timeout=10.0)
    run_ranks(comms, _work)

    def failing(comm):
        comm.allgather(None)
        if comm.Get_rank() == 1:
            raise ValueError("injected failure")
        comm.allgather(None)

    with pytest.raises(ValueError, match="injected failure"):
        run_ranks(comms, failing)
    results = run_ranks(comms, _work)
    assert [values for values, _ in results] == [[0, 1, 2]] * 3
    assert [child for _, child in results] == [[0, 1], [0], [0, 1]]


def test_held_sub_communicators_recover_too():
    comms = LocalComm.create(2, timeout=10.0)
    children = run_ranks(comms, lambda comm: comm.Split(color=0, key=comm.Get_rank()))

    def failing(index):
        def body(comm):
            child = children[comm.Get_rank()]
            child.allgather(None)
            if comm.Get_rank() == index:
                raise RuntimeError("child failure")
            child.allgather(None)

        return body

    with pytest.raises(RuntimeError, match="child failure"):
        run_ranks(comms, failing(0))
    values = run_ranks(
        comms, lambda comm: children[comm.Get_rank()].allgather(comm.Get_rank())
    )
    assert values == [[0, 1], [0, 1]]
    assert threading.active_count() >= 1
