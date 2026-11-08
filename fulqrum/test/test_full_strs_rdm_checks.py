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
from types import SimpleNamespace

import numpy as np
import pytest

from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks


def test_a_bad_input_on_one_rank_raises_on_every_rank(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    from fulqrum.gpu.full_strs.rdm import compute_rdms_full_strs  # noqa: PLC0415

    def body(comm):
        ham = SimpleNamespace(_comm=comm, num_orbitals=4, n_local=3)
        x = np.zeros(2 if comm.Get_rank() == 1 else 3)
        try:
            compute_rdms_full_strs(ham, x, rank=2)
        except ValueError as error:
            return str(error)
        return None

    messages = run_ranks(LocalComm.create(2, timeout=10.0), body)
    assert messages == [
        "full-string RDM failed: rank 1: x must have shape (3,); got (2,)"
    ] * 2


@pytest.mark.parametrize(
    "rank, orbitals, message",
    [(3, 4, "rank must be 1 or 2, got 3"), (1, 65, "at most 64 orbitals per sector")],
)
def test_rank_and_width_checks_keep_their_messages(
    monkeypatch, rank, orbitals, message
):
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    from fulqrum.gpu.full_strs.rdm import compute_rdms_full_strs  # noqa: PLC0415

    comm = LocalComm.create(1)[0]
    ham = SimpleNamespace(_comm=comm, num_orbitals=orbitals, n_local=3)
    with pytest.raises(ValueError, match=message):
        compute_rdms_full_strs(ham, np.zeros(3), rank=rank)
