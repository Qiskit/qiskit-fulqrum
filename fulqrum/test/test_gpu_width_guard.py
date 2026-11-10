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
from fulqrum.gpu.common.wide import (
    MAX_KEY_WORDS,
    check_width,
    key_words,
    max_electrons,
    width_error,
)
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


FULL = "sector 1 is fully occupied with 64 orbitals"


def test_a_full_sector_of_64_orbitals_is_rejected():
    assert width_error((64, 64), (32, 64)).startswith(FULL)
    assert width_error((64, 64), (64, 64)).startswith("sector 0 is fully occupied")


@pytest.mark.parametrize(
    "widths, electrons",
    [((63, 63), (63, 63)), ((1, 1), (1, 1)), ((64, 64), (63, 1)), ((64, 64), None)],
)
def test_other_sectors_stay_legal(widths, electrons):
    assert width_error(widths, electrons) is None


def test_without_electron_counts_only_the_width_is_checked():
    assert width_error((64, 64)) is None
    assert width_error((64, 64), None) is None
    assert width_error((64, WIDE), (64, 64)) == width_error((64, WIDE))
    assert width_error((64, 64), (None, 64)).startswith(FULL)


def test_max_electrons_reads_strings_ints_and_words():
    assert max_electrons(["0011", "0111"]) == 3
    assert max_electrons([3, 7, 1]) == 3
    assert max_electrons(np.array([3, 15], np.uint64)) == 4
    assert max_electrons(np.array([[1, 3], [0, 0]], np.uint64)) == 3
    assert max_electrons([]) is None
    assert max_electrons(np.zeros(0, np.uint64)) is None


def test_a_colliding_key_overwrites_an_all_ones_string_in_the_hash_table():
    from fulqrum.gpu.common.hash_utils import _splitmix64  # noqa: PLC0415
    from fulqrum.gpu.distributed.hash_utils import (  # noqa: PLC0415
        _build_hash_table_simple,
        _ht_lookup_batch,
    )

    # The all-ones string equals the empty-slot marker, so a later key that
    # probes its slot treats it as free and replaces it.
    full = 2**64 - 1
    home = _splitmix64(full) & 7
    other = next(k for k in range(1, 1000) if _splitmix64(k) & 7 == home)
    keys = np.array([full, other], np.uint64)
    table, values, mask = _build_hash_table_simple(keys)
    assert int(mask) == 7
    assert _ht_lookup_batch(table, values, int(mask), keys).tolist() == [-1, 1]


def test_check_width_rejects_a_full_sector_on_every_rank():

    def body(comm):
        electrons = (32, 64) if comm.Get_rank() == 1 else (32, 32)
        try:
            check_width(comm, (64, 64), "op", electrons)
        except FulqrumError as error:
            return error.message
        return None

    messages = run_ranks(LocalComm.create(2, timeout=60.0), body)
    assert messages == [f"op failed: rank 1: {width_error((64, 64), (32, 64))}"] * 2


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
        TILE_MAX_KEY_WORDS,
        TileGPUResidentHamiltonian,
    )

    wide = 64 * TILE_MAX_KEY_WORDS + 1
    hsub = _wide_hsub(wide)
    subspace = hsub.spmv.subspace
    with pytest.raises(FulqrumError, match=f"has {wide} orbitals"):
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
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (  # noqa: PLC0415
        TILE_MAX_KEY_WORDS,
    )

    wide = 64 * TILE_MAX_KEY_WORDS + 1
    with pytest.raises(FulqrumError, match=f"GPUHamiltonian failed: .*has {wide}"):
        GPUHamiltonian(_wide_hsub(wide), gpu_ids=(0,))


@requires_gpu
def test_full_strs_rejects_a_sector_wider_than_two_words():
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    operator = _wide_operator(129)
    keys = np.zeros((2, 2 * key_words(129)), dtype=np.uint64)
    with pytest.raises(FulqrumError, match="FullStrsHamiltonian failed: .*has 129"):
        FullStrsHamiltonian(operator, keys, comm=LocalComm.create(1)[0])
