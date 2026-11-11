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


import pytest

from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks
from fulqrum.gpu.distributed.rank_agreement import check_same_choice, run_on_every_rank


class _OutOfMemoryError(MemoryError):
    pass


class _FakeComm:
    def __init__(self, peer_reports):
        self.peer_reports = peer_reports

    def allgather(self, value):
        return [value, *self.peer_reports]


def _single():
    return LocalComm.create(1)[0]


def _raise(error):
    def function():
        raise error

    return function


def test_no_error_returns_result():
    assert run_on_every_rank(_single(), "step", lambda: 42) == 42


def test_value_error_unchanged():
    with pytest.raises(ValueError, match="step failed: rank 0: bad$"):
        run_on_every_rank(_single(), "step", _raise(ValueError("bad")))


def test_overflow_error_becomes_runtime_error():
    with pytest.raises(
        RuntimeError, match=r"rank 0: too big \(OverflowError\)"
    ) as info:
        run_on_every_rank(_single(), "step", _raise(OverflowError("too big")))
    assert type(info.value) is RuntimeError
    assert isinstance(info.value.__cause__, OverflowError)


def test_memory_error_subclass_becomes_memory_error():
    with pytest.raises(MemoryError, match=r"rank 0: oom \(_OutOfMemoryError\)") as info:
        run_on_every_rank(_single(), "step", _raise(_OutOfMemoryError("oom")))
    assert type(info.value) is MemoryError


def test_keyboard_interrupt_not_caught():
    comm = _FakeComm([])
    with pytest.raises(KeyboardInterrupt):
        run_on_every_rank(comm, "step", _raise(KeyboardInterrupt()))


def test_peer_report_raises_on_clean_rank():
    comm = _FakeComm([None, ("MemoryError", "oom (_OutOfMemoryError)")])
    with pytest.raises(MemoryError, match=r"rank 2: oom \(_OutOfMemoryError\)") as info:
        run_on_every_rank(comm, "step", lambda: 1)
    assert info.value.__cause__ is None


def test_first_reporting_rank_sets_type():
    comm = _FakeComm([("RuntimeError", "x (OverflowError)"), ("ValueError", "y")])
    with pytest.raises(RuntimeError) as info:
        run_on_every_rank(comm, "step", lambda: 1)
    assert "rank 1: x (OverflowError); rank 2: y" in str(info.value)


def test_three_ranks_one_overflow():
    def body(comm):
        def function():
            if comm.Get_rank() == 1:
                raise OverflowError("too big")
            return comm.Get_rank()

        try:
            run_on_every_rank(comm, "step", function)
        except RuntimeError as error:
            return str(error)
        return None

    messages = run_ranks(LocalComm.create(3, timeout=60.0), body)
    assert messages == ["step failed: rank 1: too big (OverflowError)"] * 3


def _choices(details):
    def body(comm):
        detail = details[comm.Get_rank()]
        try:
            check_same_choice(comm, detail is None, detail or "on", "arm table source")
        except RuntimeError as error:
            return str(error)
        return None

    return run_ranks(LocalComm.create(len(details), timeout=60.0), body)


def test_same_choice_all_on():
    assert _choices([None, None, None]) == [None, None, None]


def test_same_choice_one_rank_differs():
    text = (
        "arm table source differs across ranks: "
        "rank 0: on; rank 1: on; rank 2: needs FQ_AA_PAIR_TABLE"
    )
    assert _choices([None, None, "needs FQ_AA_PAIR_TABLE"]) == [text] * 3


def test_same_choice_all_off_with_different_reasons():
    assert _choices(["a", "b", "c"]) == [None, None, None]


def test_same_choice_single_rank():
    comm = LocalComm.create(1)[0]
    assert check_same_choice(comm, False, "reason", "arm table source") is None


def test_local_raise_before_choice_check_stays_collective():
    def body(comm):
        def groups():
            if comm.Get_rank() == 1:
                raise ValueError("bad groups")
            return None

        try:
            reason = run_on_every_rank(comm, "operator groups", groups)
            check_same_choice(comm, reason is None, reason or "on", "arm table source")
        except ValueError as error:
            return str(error)
        return None

    messages = run_ranks(LocalComm.create(3, timeout=10.0), body)
    assert messages == ["operator groups failed: rank 1: bad groups"] * 3


def _setup(fail_rank, failure):
    def body(comm):
        state = {}

        def setup():
            if comm.Get_rank() == fail_rank:
                if failure == "import":
                    state["available"] = False
                    state["error"] = "ImportError: No module named 'cupy'"
                    return
                raise RuntimeError("cudaErrorInvalidDevice: invalid device ordinal")
            state["available"], state["error"] = True, None

        try:
            run_on_every_rank(comm, "GPU stream setup", setup)
            check_same_choice(
                comm,
                state["available"],
                "cupy available" if state["available"] else state["error"],
                "CuPy availability",
            )
        except RuntimeError as error:
            return str(error)
        return None

    return run_ranks(LocalComm.create(2, timeout=10.0), body)


def test_cupy_missing_on_one_rank_raises_on_every_rank():
    text = (
        "CuPy availability differs across ranks: rank 0: cupy available; "
        "rank 1: ImportError: No module named 'cupy'"
    )
    assert _setup(1, "import") == [text] * 2


def test_stream_setup_error_on_one_rank_raises_on_every_rank():
    text = (
        "GPU stream setup failed: rank 1: "
        "cudaErrorInvalidDevice: invalid device ordinal"
    )
    assert _setup(1, "cuda") == [text] * 2


def test_device_selection_error_on_one_rank_raises_on_every_rank():
    def body(comm):
        def select():
            if comm.Get_rank() == 1:
                raise RuntimeError("cudaErrorInvalidDevice: invalid device ordinal")

        try:
            run_on_every_rank(comm, "GPU device selection", select)
        except RuntimeError as error:
            return str(error)
        return None

    text = (
        "GPU device selection failed: rank 1: "
        "cudaErrorInvalidDevice: invalid device ordinal"
    )
    assert run_ranks(LocalComm.create(2, timeout=10.0), body) == [text] * 2
