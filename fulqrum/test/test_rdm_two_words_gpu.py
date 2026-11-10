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


import random

import numpy as np
import pytest

from fulqrum.exceptions import FulqrumError
from fulqrum.test.test_cartesian_two_words import padded_hamiltonians

from .conftest import gpu_available

pytestmark = [
    pytest.mark.skipif(
        not gpu_available(), reason="GPU build / CUDA device unavailable"
    ),
    pytest.mark.gpu,
]

WIDTH = 70
ACTIVE = list(range(58, 70))


@pytest.fixture(scope="module")
def pair():
    return padded_hamiltonians(WIDTH, ACTIVE, (4, 3), 40, 11)


def state(op, seed=0):
    vector = np.random.default_rng(seed).standard_normal(op.shape[0])
    return vector / np.linalg.norm(vector)


def check_padded(narrow, wide):
    m = len(ACTIVE)
    rest = np.setdiff1d(np.arange(WIDTH), ACTIVE)
    scale = max(np.abs(narrow).max(), 1.0)
    block = wide[np.ix_(ACTIVE, ACTIVE)]
    assert np.abs(block - narrow).max() <= 1e-12 * scale
    assert not wide[rest].any() and not wide[:, rest].any()
    assert narrow.shape == (m, m)


@pytest.mark.parametrize("gpu_ids", [(0,), (0, 0, 0, 0)], ids=["1", "4"])
def test_two_word_one_body_rdms_equal_the_one_word_rdms(pair, gpu_ids):
    from fulqrum.gpu import GPUHamiltonian

    narrow, wide = pair
    small = GPUHamiltonian(narrow, gpu_ids=gpu_ids)
    large = GPUHamiltonian(wide, gpu_ids=gpu_ids)
    vector = state(small)
    gamma = small.compute_rdms(vector, rank=1)
    wide_gamma = large.compute_rdms(vector, rank=1)
    for a, b in zip(gamma, wide_gamma):
        check_padded(a, b)
    summed = large.compute_rdms(vector, rank=1, spin_summed=True)
    check_padded(gamma[0] + gamma[1], summed)
    assert abs(np.trace(wide_gamma[0]) - 4) < 1e-12
    assert abs(np.trace(wide_gamma[1]) - 3) < 1e-12


def test_two_word_rank_two_rdms_raise(pair):
    from fulqrum.gpu import GPUHamiltonian

    _, wide = pair
    op = GPUHamiltonian(wide, gpu_ids=(0, 0))
    with pytest.raises(FulqrumError, match="rank=2 RDMs support at most 64"):
        op.compute_rdms(state(op), rank=2)


def test_forced_two_word_rank_two_rdms_raise(pair, monkeypatch):
    from fulqrum.gpu import GPUHamiltonian

    narrow, _ = pair
    monkeypatch.setenv("FQ_FORCE_KEY_WORDS", "2")
    op = GPUHamiltonian(narrow, gpu_ids=(0,))
    with pytest.raises(FulqrumError, match="rank=2"):
        op.compute_rdms(state(op), rank=2)


def test_forced_two_word_one_body_rdms_equal_one_word(pair, monkeypatch):
    from fulqrum.gpu import GPUHamiltonian

    narrow, _ = pair
    small = GPUHamiltonian(narrow, gpu_ids=(0, 0))
    vector = state(small, 1)
    expected = small.compute_rdms(vector, rank=1)
    monkeypatch.setenv("FQ_FORCE_KEY_WORDS", "2")
    forced = GPUHamiltonian(narrow, gpu_ids=(0, 0)).compute_rdms(vector, rank=1)
    for a, b in zip(expected, forced):
        assert np.abs(a - b).max() <= 1e-12 * max(np.abs(a).max(), 1.0)


def test_same_diagonal_gemm_equals_the_kernel():
    import cupy

    from fulqrum.gpu.distributed import rdm_cart
    from fulqrum.gpu.rdm import kernel_launch

    rng = random.Random(4)
    m, ne = 40, 6
    values = sorted({sum(1 << o for o in rng.sample(range(m), ne)) for _ in range(500)})
    keys = cupy.asarray(np.array(values, dtype=np.uint64))
    weight = cupy.asarray(np.array([rng.random() for _ in values]))
    weight[::9] = 0.0
    lo, hi = 17, len(values) - 23
    kernel = cupy.zeros(m**4)
    kernel_launch.launch_same_diagonal(
        weight.data.ptr, keys.data.ptr, lo, hi - lo, kernel.data.ptr, m, 0
    )
    gemm = cupy.zeros(m**4)
    rdm_cart._same_diagonal_gemm(cupy, weight, keys.reshape(-1, 1)[lo:hi], gemm, m)
    cupy.cuda.Device().synchronize()
    scale = float(cupy.abs(kernel).max())
    assert scale > 0
    assert float(cupy.abs(gemm - kernel).max()) <= 1e-14 * scale


def test_padded_needed_tiles_equal_the_unpadded_tiles(pair):
    from fulqrum.gpu import GPUHamiltonian
    from fulqrum.gpu.distributed.rdm_cart import _needed_tiles

    narrow, wide = pair
    small = GPUHamiltonian(narrow, gpu_ids=(0, 0, 0, 0))
    large = GPUHamiltonian(wide, gpu_ids=(0, 0, 0, 0))
    for rank, (a, b) in enumerate(zip(small._ranks, large._ranks)):
        assert np.array_equal(_needed_tiles(a, rank, 1), _needed_tiles(b, rank, 1))
