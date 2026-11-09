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


import itertools
import random

import numpy as np
import pytest

from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks
from fulqrum.test.test_full_strs_gpu import requires_gpu
from fulqrum.test.test_full_strs_two_words import (
    cpu_reference,
    from_words,
    random_sectors,
    wide_keys,
)

pytestmark = pytest.mark.gpu

NORB = 70
ACTIVE = list(range(58, 70))


def padded(seed, rows=800):
    from fulqrum.test.test_full_strs_coefficients import random_operator  # noqa: PLC0415

    rng = random.Random(seed)
    small = len(ACTIVE)
    alpha = random_sectors(rng, list(range(small)), 4, rows)
    beta = random_sectors(rng, list(range(small)), 3, rows)

    def place(value):
        return sum(1 << ACTIVE[k] for k in range(small) if (value >> k) & 1)

    return (
        random_operator(small, seed),
        wide_keys(alpha, beta, 1),
        random_operator(NORB, seed, active=ACTIVE),
        wide_keys([place(v) for v in alpha], [place(v) for v in beta], 2),
    )


def squeeze_rows(keys):
    position = {a: i for i, a in enumerate(ACTIVE)}

    def squeeze(value):
        return sum(1 << position[o] for o in ACTIVE if (value >> o) & 1)

    return [
        (squeeze(from_words(row[:2])), squeeze(from_words(row[2:]))) for row in keys
    ]


@requires_gpu
def test_two_word_matvec_equals_the_core_hamiltonian():
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    _op, _keys, op, keys = padded(31)
    H = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0])
    assert H.keys.shape[1] == 4
    reference = cpu_reference(op, H.keys, NORB)
    x = np.random.default_rng(3).normal(size=H.n_local)
    expected = reference.matvec(x)
    got = H.matvec(x)
    assert np.max(np.abs(got - expected)) <= 1e-12 * np.max(np.abs(expected))


@requires_gpu
@pytest.mark.parametrize("vectors", ["halo", "full"])
def test_two_word_ranks_equal_one_rank(vectors):
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    _op, _keys, op, keys = padded(37, rows=1500)
    one = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0])
    x = np.random.default_rng(5).normal(size=one.n_local)
    expected = dict(zip(map(tuple, one.keys.tolist()), one.matvec(x)))
    by_key = dict(zip(map(tuple, one.keys.tolist()), x))
    parts = np.array_split(keys, 3)
    groups = "by_key" if vectors == "full" else "by_rows"

    def body(comm):
        import cupy  # noqa: PLC0415

        cupy.cuda.Device(0).use()
        H = FullStrsHamiltonian(
            op, parts[comm.Get_rank()], comm=comm, groups=groups, vectors=vectors
        )
        local = np.array([by_key[tuple(row)] for row in H.keys.tolist()])
        return H.keys, H.matvec(local)

    results = run_ranks(LocalComm.create(3, timeout=300.0), body)
    worst = 0.0
    scale = max(abs(v) for v in expected.values())
    for rows, values in results:
        for row, value in zip(rows.tolist(), values):
            worst = max(worst, abs(value - expected[tuple(row)]))
    assert worst <= 1e-12 * scale


def group_multiset(H):
    from fulqrum.gpu.full_strs.host import CLASSES  # noqa: PLC0415

    host = H._host
    groups = []
    for chunk in range(host.num_chunks):
        for cls in range(len(CLASSES)):
            members, sizes, _starts = host.chunk_views(chunk, cls)
            offsets = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
            groups += [
                (cls, tuple(sorted(members[a:b].tolist())))
                for a, b in itertools.pairwise(offsets)
            ]
    return sorted(groups)


@requires_gpu
def test_forced_two_words_match_one_word(monkeypatch):
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    # Key slices hash every key word, so the zero high words move groups between
    # slices: the same groups are summed in a different order.
    op, keys, _wide_op, _wide_keys = padded(41, rows=2000)
    monkeypatch.delenv("FQ_FORCE_KEY_WORDS", raising=False)
    narrow = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0])
    monkeypatch.setenv("FQ_FORCE_KEY_WORDS", "2")
    wide = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0])
    assert wide.keys.shape[1] == 4 and not wide.keys[:, [1, 3]].any()
    np.testing.assert_array_equal(wide.keys[:, [0, 2]], narrow.keys)
    x = np.random.default_rng(7).normal(size=narrow.n_local)
    expected = narrow.matvec(x)
    got = wide.matvec(x)
    assert np.max(np.abs(got - expected)) <= 1e-13 * np.max(np.abs(expected))
    np.testing.assert_array_equal(wide.diag_local, narrow.diag_local)
    narrow_groups, wide_groups = group_multiset(narrow), group_multiset(wide)
    assert len(wide_groups) == len(narrow_groups) > 1000
    assert wide_groups == narrow_groups


@requires_gpu
def test_forced_two_words_with_one_slice_match_one_word_bit_for_bit(monkeypatch):
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    op, keys, _wide_op, _wide_keys = padded(41, rows=2000)
    monkeypatch.delenv("FQ_FORCE_KEY_WORDS", raising=False)
    narrow = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0], num_slices=1)
    monkeypatch.setenv("FQ_FORCE_KEY_WORDS", "2")
    wide = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0], num_slices=1)
    x = np.random.default_rng(7).normal(size=narrow.n_local)
    np.testing.assert_array_equal(wide.matvec(x), narrow.matvec(x))


@requires_gpu
def test_two_word_rdms_equal_the_unpadded_rdms():
    from fulqrum.gpu.full_strs import FullStrsHamiltonian  # noqa: PLC0415

    small_op, small_keys, op, keys = padded(43, rows=600)
    narrow = FullStrsHamiltonian(small_op, small_keys, comm=LocalComm.create(1)[0])
    wide = FullStrsHamiltonian(op, keys, comm=LocalComm.create(1)[0])
    rows = squeeze_rows(wide.keys)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(narrow.keys.tolist())}
    order = np.array([index[row] for row in rows])
    x = np.random.default_rng(9).normal(size=narrow.n_local)
    x /= np.linalg.norm(x)
    (ga, gb), blocks = narrow.compute_rdms(x, rank=2)
    (wa, wb), wide_blocks = wide.compute_rdms(x[order], rank=2)
    active = np.array(ACTIVE)
    rest = np.setdiff1d(np.arange(NORB), active)
    for small, big in [(ga, wa), (gb, wb)]:
        np.testing.assert_allclose(big[np.ix_(active, active)], small, atol=1e-12)
        assert not big[rest].any() and not big[:, rest].any()
    for small, big in zip(blocks, wide_blocks):
        np.testing.assert_allclose(
            big[np.ix_(active, active, active, active)], small, atol=1e-12
        )
        for axis in range(4):
            assert not np.take(big, rest, axis=axis).any()
