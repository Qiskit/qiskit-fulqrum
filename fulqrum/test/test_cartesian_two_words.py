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

import fulqrum as fq
from fulqrum.exceptions import FulqrumError
from fulqrum.gpu.common.wide import pack_strings, sort_strings, unpack_ints
from fulqrum.gpu.distributed.hash_utils import _pack_half_strs_to_uint64
from fulqrum.test.test_full_strs_coefficients import random_operator

WIDTHS = (10, 64, 65, 100, 128)


def random_strings(rng, width, count, electrons=None):
    out = []
    for _ in range(count):
        if electrons is None:
            value = rng.getrandbits(width)
        else:
            value = sum(1 << o for o in rng.sample(range(width), electrons))
        out.append(format(value, f"0{width}b"))
    return out


@pytest.mark.parametrize("width", WIDTHS)
def test_packing_equals_python_integers(width):
    rng = random.Random(width)
    alpha = random_strings(rng, width, 300)
    beta = random_strings(rng, width, 200)
    for packed, strings in zip(
        _pack_half_strs_to_uint64(alpha, beta, (width, width)), (alpha, beta)
    ):
        assert unpack_ints(packed) == [int(s, 2) for s in strings]
        assert packed.shape == ((len(strings),) if width <= 64 else (len(strings), 2))
        assert packed.flags.c_contiguous
    np.testing.assert_array_equal(
        pack_strings(alpha, width),
        _pack_half_strs_to_uint64(alpha, alpha, (width, width))[0],
    )


def test_default_packing_is_the_single_word_packing():
    strings = ["0" * 63 + "1", "1" + "0" * 63, "101"]
    alpha, beta = _pack_half_strs_to_uint64(strings, strings[:2])
    np.testing.assert_array_equal(
        alpha, np.array([int(s, 2) for s in strings], dtype=np.uint64)
    )
    assert alpha.dtype == np.uint64 and beta.shape == (2,)
    with pytest.raises(ValueError, match="at or above orbital 64"):
        _pack_half_strs_to_uint64(["1" + "0" * 64], strings)


@pytest.mark.parametrize("width", WIDTHS)
def test_subspace_words_equal_python_integers(width):
    rng = random.Random(100 + width)
    alpha = sorted(set(random_strings(rng, width, 40, electrons=3)))
    beta = sorted(set(random_strings(rng, width, 30, electrons=2)))
    space = fq.Subspace([alpha, beta])
    a, b = space._alpha_beta_words(2)
    assert unpack_ints(a) == [int(s, 2) for s in space.alpha_strs]
    assert unpack_ints(b) == [int(s, 2) for s in space.beta_strs]
    if width <= 64:
        assert a.shape == (len(alpha),)
        np.testing.assert_array_equal(a, space.to_alpha_beta_arrays()[0])
    else:
        assert a.shape == (len(alpha), 2)
        with pytest.raises(FulqrumError, match="at most 64 qubits"):
            space.to_alpha_beta_arrays()


def test_subspace_words_refuse_more_than_two_words():
    space = fq.Subspace([["1" + "0" * 128], ["1" + "0" * 128]])
    with pytest.raises(FulqrumError, match="at most 128 qubits"):
        space._alpha_beta_words(2)


def test_sort_strings_orders_by_integer_value():
    rng = random.Random(63)
    values = []
    for _ in range(10**5):
        value = rng.getrandbits(128)
        if rng.random() < 0.5:
            value &= (1 << 70) - 1
            value |= rng.getrandbits(10) << 58
        values.append(value)
    words = pack_strings([format(v, "0128b") for v in values], 128)
    assert unpack_ints(sort_strings(words)) == sorted(values)
    single = pack_strings([format(v & ((1 << 64) - 1), "064b") for v in values], 64)
    np.testing.assert_array_equal(sort_strings(single), np.sort(single))


def padded_hamiltonians(width, active, electrons, count, seed):
    rng = random.Random(seed)
    small = len(active)
    alpha = sorted(set(random_strings(rng, small, count, electrons[0])))
    beta = sorted(set(random_strings(rng, small, count, electrons[1])))

    def place(text):
        value = int(text, 2)
        wide = sum(1 << active[k] for k in range(small) if (value >> k) & 1)
        return format(wide, f"0{width}b")

    narrow_op = random_operator(small, seed)
    narrow_op.set_type(2)
    wide_op = random_operator(width, seed, active=active)
    wide_op.set_type(2)
    narrow = fq.SubspaceHamiltonian(narrow_op, fq.Subspace([alpha, beta]))
    wide = fq.SubspaceHamiltonian(
        wide_op, fq.Subspace([[place(s) for s in alpha], [place(s) for s in beta]])
    )
    return narrow, wide


@pytest.mark.parametrize(
    "width, active",
    [(70, list(range(58, 70))), (128, list(range(56, 67)) + [127])],
)
def test_two_word_diagonal_equals_the_one_word_diagonal(width, active):
    narrow, wide = padded_hamiltonians(width, active, (4, 3), 60, width)
    alpha, beta = narrow.spmv.subspace.to_alpha_beta_arrays()
    wide_alpha, wide_beta = wide.spmv.subspace._alpha_beta_words(2)
    assert wide_alpha.shape == (alpha.size, 2) and wide_beta.shape == (beta.size, 2)
    small = len(active)
    a0, a1 = 2, alpha.size - 3
    expected = narrow.spmv.diagonal_slice_product_space(alpha, beta, a0, a1, small)
    got = wide.spmv.diagonal_slice_product_space(wide_alpha, wide_beta, a0, a1, width)
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(got - expected)) <= 1e-13 * scale
    pieces = narrow.spmv.diag_separable_pieces(alpha, beta, a0, a1, small)
    wide_pieces = wide.spmv.diag_separable_pieces(wide_alpha, wide_beta, a0, a1, width)
    assert pieces[3] and wide_pieces[3]
    for piece, wide_piece in zip(pieces[:2], wide_pieces[:2]):
        np.testing.assert_allclose(wide_piece, piece, rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(
        wide_pieces[2][np.ix_(active, active)], pieces[2], rtol=0, atol=1e-13 * scale
    )
    rest = np.setdiff1d(np.arange(width), active)
    assert not wide_pieces[2][rest].any() and not wide_pieces[2][:, rest].any()
    core = wide.diagonal_vector().reshape(beta.size, alpha.size)[:, a0:a1].ravel()
    core = core - wide.spmv.const_energy
    assert np.max(np.abs(got - core)) <= 1e-12 * max(scale, 1.0)


def test_mixed_sector_widths_equal_the_core_diagonal():
    rng = random.Random(5)
    alpha_width, beta_width = 40, 100
    alpha = sorted(set(random_strings(rng, alpha_width, 30, 3)))
    beta = sorted(set(random_strings(rng, beta_width, 30, 2)))
    terms = []
    for _ in range(60):
        a = rng.randrange(alpha_width)
        b = alpha_width + rng.randrange(beta_width)
        kind = rng.choice(["ZZ", "11", "Z1", "1Z"])
        terms.append((kind, [a, b], rng.uniform(-1, 1)))
        terms.append(
            ("Z", [rng.randrange(alpha_width + beta_width)], rng.uniform(-1, 1))
        )
    op = fq.QubitOperator(alpha_width + beta_width, terms)
    op.set_type(2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([alpha, beta]))
    wide_a, wide_b = hsub.spmv.subspace._alpha_beta_words(2)
    assert wide_a.shape == (len(alpha),) and wide_b.shape == (len(beta), 2)
    got = hsub.spmv.diagonal_slice_product_space(
        wide_a, wide_b, 0, len(alpha), alpha_width
    )
    core = hsub.diagonal_vector() - hsub.spmv.const_energy
    np.testing.assert_allclose(got, core, rtol=0, atol=1e-12)
    _da, _db, coupling, ok = hsub.spmv.diag_separable_pieces(
        wide_a, wide_b, 0, len(alpha), alpha_width
    )
    assert ok and coupling.shape == (alpha_width, beta_width)


def test_single_word_diagonal_keeps_its_results():
    from fulqrum.test.test_gpu_contract import _half_strings, _lih_operator  # noqa: PLC0415

    op = _lih_operator()
    half = op.width // 2
    strings = _half_strings(half, 2)
    hsub = fq.SubspaceHamiltonian(op, fq.Subspace([strings, strings]))
    alpha, beta = hsub.spmv.subspace.to_alpha_beta_arrays()
    local = hsub.spmv.diagonal_slice_product_space(alpha, beta, 1, alpha.size - 1, half)
    pieces = hsub.spmv.diag_separable_pieces(alpha, beta, 1, alpha.size - 1, half)
    assert local.dtype == np.float64 and pieces[3]
    expected = hsub.diagonal_vector().reshape(beta.size, alpha.size)[:, 1:-1].ravel()
    np.testing.assert_allclose(
        local + hsub.spmv.const_energy, expected, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(
        hsub.spmv.diagonal_slice_product_space(
            alpha.astype(np.int64), beta, 1, alpha.size - 1, half
        ),
        local,
    )


def test_wrong_string_shapes_raise():
    narrow, wide = padded_hamiltonians(70, list(range(58, 70)), (4, 3), 20, 1)
    wide_alpha, wide_beta = wide.spmv.subspace._alpha_beta_words(2)
    with pytest.raises(
        ValueError,
        match=r"alpha strings must have shape \(N,\) for one word or \(N, 2\)",
    ):
        wide.spmv.diagonal_slice_product_space(wide_alpha[:, 0], wide_beta, 0, 1, 70)
    alpha, _beta = narrow.spmv.subspace.to_alpha_beta_arrays()
    with pytest.raises(ValueError, match="beta strings must have shape"):
        narrow.spmv.diag_separable_pieces(alpha, wide_beta, 0, 1, 12)
