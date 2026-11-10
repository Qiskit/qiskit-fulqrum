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


import re

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
LINE = re.compile(
    r"^arm plan: rank=\d+ source=\d+ key_words=(\d),(\d) "
    r"aabb=(clique|transposed_clique|direct|baseline|none) "
    r"singles=(pair|delta|kernel|none) diag=(stored|recompute)$"
)


@pytest.fixture(scope="module")
def pair():
    return padded_hamiltonians(WIDTH, ACTIVE, (4, 3), 40, 7)


def vector(op, seed=0):
    return np.random.default_rng(seed).standard_normal(op.shape[0])


def plan_lines(op):
    return [line for rank in op._ranks for line in rank._arm_plan_lines]


@pytest.mark.parametrize("gpu_ids", [(0,), (0, 0, 0, 0)], ids=["1", "4"])
def test_two_word_matvec_equals_the_one_word_matvec(pair, gpu_ids):
    from fulqrum.gpu import GPUHamiltonian

    narrow, wide = pair
    small = GPUHamiltonian(narrow, gpu_ids=gpu_ids)
    large = GPUHamiltonian(wide, gpu_ids=gpu_ids)
    assert large.shape == small.shape
    x = vector(small)
    expected = small @ x
    got = large @ x
    assert np.abs(got - expected).max() <= 1e-13 * np.abs(expected).max()
    lines = plan_lines(large)
    assert lines and all(LINE.match(line) for line in lines)
    assert all(LINE.match(line).group(1, 2) == ("2", "2") for line in lines)
    assert not any("aabb=baseline" in line for line in lines)
    assert not any(
        "singles=delta" in line or "singles=kernel" in line for line in lines
    )
    assert all("diag=stored" in line for line in lines)
    if len(gpu_ids) == 4:
        assert any("aabb=direct" in line for line in lines)


def test_two_word_matvec_equals_the_cpu_operator(pair):
    from fulqrum.gpu import GPUHamiltonian

    _, wide = pair
    op = GPUHamiltonian(wide, gpu_ids=(0,))
    dense = wide.spmv.to_csr_array().toarray()
    x = vector(op, 3)
    expected = dense @ x
    assert np.abs(op @ x - expected).max() <= 1e-12 * np.abs(expected).max()


def test_forced_two_words_match_one_word(pair, monkeypatch):
    from fulqrum.gpu import GPUHamiltonian

    narrow, _ = pair
    small = GPUHamiltonian(narrow, gpu_ids=(0, 0, 0, 0))
    x = vector(small, 1)
    expected = small @ x
    monkeypatch.setenv("FQ_FORCE_KEY_WORDS", "2")
    forced = GPUHamiltonian(narrow, gpu_ids=(0, 0, 0, 0))
    lines = plan_lines(forced)
    assert lines and all("key_words=2,2" in line for line in lines)
    assert not any("aabb=baseline" in line for line in lines)
    got = forced @ x
    assert np.abs(got - expected).max() <= 1e-13 * np.abs(expected).max()


def test_single_word_plans_report_one_word(pair):
    from fulqrum.gpu import GPUHamiltonian

    narrow, _ = pair
    lines = plan_lines(GPUHamiltonian(narrow, gpu_ids=(0, 0)))
    assert lines and all("key_words=1,1" in line for line in lines)
    assert not any("aabb=direct" in line for line in lines)


@pytest.mark.parametrize(
    "name,value",
    [
        ("FQ_ARM_TABLE_SOURCE", "kernel"),
        ("FQ_DIAG_RECOMPUTE", "1"),
        ("FQ_AA_PAIR_TABLE", "0"),
        ("FQ_BB_PAIR_TABLE", "0"),
    ],
)
def test_string_reading_overrides_raise_for_two_words(pair, monkeypatch, name, value):
    from fulqrum.gpu import GPUHamiltonian

    _, wide = pair
    monkeypatch.setenv(name, value)
    with pytest.raises((FulqrumError, ValueError), match=name):
        op = GPUHamiltonian(wide, gpu_ids=(0, 0, 0, 0))
        op @ vector(op)


@pytest.mark.parametrize(
    "name,value",
    [
        ("FQ_AABB_CLIQUE_ENABLE", "0"),
        ("FQ_AABB_REMOTE_BETA_METHOD", "baseline"),
        ("FQ_AABB_REMOTE_ALPHA_METHOD", "baseline"),
    ],
)
def test_clique_settings_route_to_the_direct_kernel(pair, monkeypatch, name, value):
    from fulqrum.gpu import GPUHamiltonian

    narrow, wide = pair
    small = GPUHamiltonian(narrow, gpu_ids=(0, 0, 0, 0))
    x = vector(small, 2)
    expected = small @ x
    monkeypatch.setenv(name, value)
    large = GPUHamiltonian(wide, gpu_ids=(0, 0, 0, 0))
    lines = plan_lines(large)
    assert not any("aabb=baseline" in line for line in lines)
    assert np.abs(large @ x - expected).max() <= 1e-13 * np.abs(expected).max()


def test_term_counts_widen_past_255_orbitals():
    import fulqrum as fq
    from fulqrum.gpu.common.operator_arrays import to_gpu_arrays
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import _pack_term_tables
    from fulqrum.test.test_full_strs_coefficients import random_operator

    width = 128
    active = [0, 1, 2, 125, 126, 127]
    op = random_operator(width, 5, active=active)
    op.set_type(2)
    alpha = [format(sum(1 << o for o in pair), f"0{width}b") for pair in ((0, 127),)]
    beta = [format(sum(1 << o for o in pair), f"0{width}b") for pair in ((1, 126),)]
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([alpha, beta]))
    data = to_gpu_arrays(hsub.spmv, skip_diag=True)
    counts = np.asarray(data["term_num_ops"])
    longest = int(counts.max())
    assert longest > 255 and counts.dtype == np.uint16
    assert np.asarray(data["term_inds"]).dtype == np.uint8
    with pytest.raises(ValueError, match="at most 255"):
        _pack_term_tables(data)


def test_single_word_term_arrays_keep_their_types(pair):
    from fulqrum.gpu.common.operator_arrays import to_gpu_arrays

    narrow, _ = pair
    data = to_gpu_arrays(narrow.spmv, skip_diag=True)
    for name in ("term_num_ops", "term_num_proj", "term_inds", "grp_offdiag_sizes"):
        assert np.asarray(data[name]).dtype == np.uint8, name
