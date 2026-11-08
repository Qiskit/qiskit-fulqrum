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


import numpy as np
import pytest

from fulqrum.exceptions import FulqrumError
from fulqrum.gpu.common.wide import MAX_KEY_WORDS, check_width, key_words, width_error
from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks

from .conftest import gpu_available

requires_gpu = pytest.mark.skipif(
    not gpu_available(), reason="GPU build / CUDA device unavailable"
)

WIDE = 64 * MAX_KEY_WORDS + 1


def test_key_words():
    assert [key_words(n) for n in (1, 64, 65, 128, 129)] == [1, 1, 2, 2, 3]


def test_width_error():
    assert width_error((1, 64)) is None
    assert width_error(()) is None
    assert width_error((64, WIDE)) == (
        f"sector 1 has {WIDE} orbitals; "
        f"this build supports at most {WIDE - 1} per sector"
    )
    assert width_error((WIDE + 1, WIDE)).startswith(f"sector 0 has {WIDE + 1}")


def test_check_width_one_rank():
    comm = LocalComm.create(1)[0]
    assert check_width(comm, (64, 64), "op") is None
    with pytest.raises(FulqrumError, match=f"op failed: rank 0: sector 0 has {WIDE}"):
        check_width(comm, (WIDE, 64), "op")


def test_check_width_three_ranks_one_bad():
    def body(comm):
        widths = (64, WIDE) if comm.Get_rank() == 2 else (64, 64)
        try:
            check_width(comm, widths, "op")
        except FulqrumError as error:
            return error.message
        return None

    messages = run_ranks(LocalComm.create(3, timeout=60.0), body)
    assert messages == [f"op failed: rank 2: {width_error((64, WIDE))}"] * 3


def _wide_operator(num_orbitals=WIDE):
    import fulqrum as fq  # noqa: PLC0415

    terms = []
    for spin in (0, num_orbitals):
        for i in range(num_orbitals - 1):
            terms.append(("+-", [spin + i + 1, spin + i], -1.0))
            terms.append(("+-", [spin + i, spin + i + 1], -1.0))
    op = fq.FermionicOperator(2 * num_orbitals, terms).extended_jw_transformation()
    op.set_type(2)
    return op


def _wide_hsub(num_orbitals=WIDE):
    import fulqrum as fq  # noqa: PLC0415

    strings = []
    for ones in ((0, 1), (0, num_orbitals - 1), (1, num_orbitals - 1)):
        bits = ["0"] * num_orbitals
        for i in ones:
            bits[num_orbitals - 1 - i] = "1"
        strings.append("".join(bits))
    strings.sort()
    subspace = fq.Subspace([strings, list(strings)])
    return fq.SubspaceHamiltonian(_wide_operator(num_orbitals), subspace)


@requires_gpu
def test_tile_resident_rejects_wide_sector():
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (  # noqa: PLC0415
        TileGPUResidentHamiltonian,
    )

    hsub = _wide_hsub()
    subspace = hsub.spmv.subspace
    with pytest.raises(FulqrumError, match=f"has {WIDE} orbitals"):
        TileGPUResidentHamiltonian(
            hsub,
            subspace.alpha_strs,
            subspace.beta_strs,
            comm=LocalComm.create(1)[0],
            comm_backend="inproc",
        )


@requires_gpu
def test_gpu_hamiltonian_rejects_wide_sector():
    from fulqrum.gpu import GPUHamiltonian  # noqa: PLC0415

    with pytest.raises(FulqrumError, match=f"GPUHamiltonian failed: .*has {WIDE}"):
        GPUHamiltonian(_wide_hsub(), gpu_ids=(0,))


@requires_gpu
def test_full_strs_rejects_wide_sector():
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    operator = _wide_operator()
    keys = np.zeros((2, 2 * key_words(WIDE)), dtype=np.uint64)
    with pytest.raises(FulqrumError, match=f"FullStrsHamiltonian failed: .*has {WIDE}"):
        FullStrsHamiltonian(operator, keys, comm=LocalComm.create(1)[0])
