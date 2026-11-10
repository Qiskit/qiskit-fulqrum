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


def separated_hamiltonians():
    """Return narrow and padded operators whose low and high beta thirds share no single move."""
    import itertools

    import fulqrum as fq
    from fulqrum.test.test_full_strs_coefficients import random_operator

    small = len(ACTIVE)

    def subsets(orbitals):
        return [sum(1 << o for o in c) for c in itertools.combinations(orbitals, 3)]

    low = sorted(subsets(range(6)))[:6]
    middle = sorted({sum(1 << o for o in (5, 6, x)) for x in range(5)})
    high = sorted(subsets(range(6, 12)))[:6]
    beta = sorted(set(low + middle + high))
    alpha = sorted(subsets(range(small)))[:30:3]

    def text(value, width, place):
        wide = sum(1 << ACTIVE[k] for k in range(small) if (value >> k) & 1)
        return format(wide if place else value, f"0{width}b")

    def build(width, place):
        operator = random_operator(width, 21, active=ACTIVE if place else None)
        operator.set_type(2)
        subspace = fq.Subspace(
            [
                [text(a, width, place) for a in alpha],
                [text(b, width, place) for b in beta],
            ]
        )
        return fq.SubspaceHamiltonian(operator, subspace)

    return build(small, False), build(WIDTH, True)


def assembled_products(hsub, grid):
    from fulqrum.gpu.distributed import _collective
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
        TileGPUResidentHamiltonian,
    )
    from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks

    _collective.reset_for_testing()
    subspace = hsub.spmv.subspace
    n_a, n_b = len(subspace.alpha_strs), len(subspace.beta_strs)
    x = np.random.default_rng(13).standard_normal(n_a * n_b)

    def body(comm):
        import cupy

        cupy.cuda.Device(0).use()
        op = TileGPUResidentHamiltonian(
            hsub,
            subspace.alpha_strs,
            subspace.beta_strs,
            comm=comm,
            process_grid=grid,
            comm_backend="inproc",
        )
        full = x.reshape(op._n_b, op._n_a)
        tile = np.ascontiguousarray(full[op._b0 : op._b1, op._a0 : op._a1]).ravel()
        y = np.asarray(op._matvec(tile))
        return (op._a0, op._a1, op._b0, op._b1), y, list(op._arm_plan_lines)

    out = np.zeros((n_b, n_a))
    lines = []
    for (a0, a1, b0, b1), y, plan in run_ranks(
        LocalComm.create(grid[0] * grid[1], timeout=300.0), body
    ):
        out[b0:b1, a0:a1] = y.reshape(b1 - b0, a1 - a0)
        lines += plan
    return out, lines


def test_a_source_without_mixed_spin_work_is_skipped_for_two_words():
    narrow, wide = separated_hamiltonians()
    expected, _ = assembled_products(narrow, (1, 3))
    got, lines = assembled_products(wide, (1, 3))
    assert any("key_words=2,2 aabb=none" in line for line in lines)
    assert any("aabb=direct" in line for line in lines)
    assert not any("aabb=baseline" in line for line in lines)
    assert np.abs(got - expected).max() <= 1e-13 * np.abs(expected).max()
