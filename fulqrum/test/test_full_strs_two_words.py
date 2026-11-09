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
from fulqrum.gpu.common.wide import check_width, key_words, width_error
from fulqrum.gpu.distributed.local_comm import LocalComm, run_ranks
from fulqrum.gpu.full_strs import host as host_w1
from fulqrum.gpu.full_strs import host_w2
from fulqrum.gpu.full_strs.distribute import sort_unique
from fulqrum.gpu.full_strs.host_words import host_module
from fulqrum.test.test_full_strs_coefficients import random_operator

MASK = (1 << 64) - 1
OLD_CLASS_SHIFT, OLD_FIRST_SHIFT, OLD_SECOND_SHIFT, OLD_ORBITAL_MASK = 61, 54, 47, 0x7F
NEW_CLASS_SHIFT, NEW_FIRST_SHIFT, NEW_SECOND_SHIFT, NEW_ORBITAL_MASK = 61, 53, 45, 0xFF


def to_words(values, words):
    return np.array(
        [[(v >> (64 * w)) & MASK for w in range(words)] for v in values], np.uint64
    )


def from_words(row):
    return sum(int(word) << (64 * i) for i, word in enumerate(row))


def wide_keys(alpha, beta, words):
    return np.ascontiguousarray(
        np.concatenate([to_words(alpha, words), to_words(beta, words)], axis=1)
    )


def random_sectors(rng, orbitals, electrons, count):
    return [sum(1 << o for o in rng.sample(orbitals, electrons)) for _ in range(count)]


def test_the_selector_returns_the_module_for_each_width():
    assert host_module(1) is host_w1
    assert host_module(2) is host_w2
    assert (host_w1.KEY_WORDS, host_w1.ROW_WORDS, host_w1.RECORD_WORDS) == (1, 2, 3)
    assert (host_w2.KEY_WORDS, host_w2.ROW_WORDS, host_w2.RECORD_WORDS) == (2, 4, 5)
    with pytest.raises(ValueError, match="one or two words per sector, got 3"):
        host_module(3)


def decode(words, shifts, mask):
    cls_shift, first_shift, second_shift = shifts
    out = []
    for word in (int(w) for w in words):
        first = (word >> first_shift) & mask
        second = (word >> second_shift) & mask
        out.append(
            (
                word >> cls_shift,
                -1 if first == mask else first,
                -1 if second == mask else second,
                word & ((1 << second_shift) - 1),
            )
        )
    return out


def pack(tuples, shifts, mask):
    cls_shift, first_shift, second_shift = shifts
    return [
        (cls << cls_shift)
        | ((mask if first < 0 else first) << first_shift)
        | ((mask if second < 0 else second) << second_shift)
        | row
        for cls, first, second, row in tuples
    ]


@pytest.mark.parametrize(
    "words, norb, window",
    [(1, 10, range(10)), (1, 64, range(54, 64)), (2, 70, range(58, 70))],
)
def test_join_members_decode_to_the_same_tuples_in_either_layout(words, norb, window):
    rng = random.Random(norb)
    module = host_module(words)
    orbitals = list(window)
    keys = sort_unique(
        wide_keys(
            random_sectors(rng, orbitals, 3, 400),
            random_sectors(rng, orbitals, 2, 400),
            words,
        )
    )
    records, _counts = module.emit_records(keys, 0, norb, 0, 0, 1, 0, 1)
    headers, _hc, members, _mc = module.join_runs(
        records, np.array([0, keys.shape[0]], np.int64)
    )
    members = members.reshape(-1)
    assert members.size > 300
    decoded = [tuple(int(v) for v in row) for row in module.decode_members(members)]
    new = (NEW_CLASS_SHIFT, NEW_FIRST_SHIFT, NEW_SECOND_SHIFT)
    old = (OLD_CLASS_SHIFT, OLD_FIRST_SHIFT, OLD_SECOND_SHIFT)
    assert decoded == decode(members, new, NEW_ORBITAL_MASK)
    assert pack(decoded, new, NEW_ORBITAL_MASK) == [int(m) for m in members]
    if norb <= 127:
        repacked = pack(decoded, old, OLD_ORBITAL_MASK)
        assert decode(repacked, old, OLD_ORBITAL_MASK) == decoded
    assert all(0 <= row < keys.shape[0] for _c, _p, _q, row in decoded)
    assert all(-1 <= p < norb and -1 <= q < norb for _c, p, q, _r in decoded)
    if words == 2:
        assert any(p >= 64 or q >= 64 for _c, p, q, _r in decoded)
        assert any(0 <= p < 64 or 0 <= q < 64 for _c, p, q, _r in decoded)
    assert headers.shape[1] == module.RECORD_WORDS


def test_a_member_word_holds_every_orbital_up_to_255_and_rows_below_2_45():
    tuples = [
        (cls, first, second, row)
        for cls in range(5)
        for first in (-1, 0, 63, 64, 127, 128, 254)
        for second in (-1, 0, 64, 254)
        for row in (0, 1, 2**45 - 1)
    ]
    words = np.array(pack(tuples, (61, 53, 45), 0xFF), np.uint64)
    decoded = [tuple(int(v) for v in row) for row in host_w2.decode_members(words)]
    assert decoded == tuples
    assert [tuple(int(v) for v in r) for r in host_w1.decode_members(words)] == tuples


def test_two_word_sort_order_matches_the_key_store():
    rng = random.Random(63)
    orbitals = list(range(100))
    straddle = list(range(55, 75))

    def sector():
        occupied = set(rng.sample(straddle, 3)) | set(rng.sample(orbitals, 3))
        while len(occupied) < 6:
            occupied.add(rng.randrange(100))
        while len(occupied) > 6:
            occupied.pop()
        return sum(1 << o for o in occupied)

    pool_a = [sector() for _ in range(300)]
    pool_b = [sector() for _ in range(300)]
    rows = [(rng.choice(pool_a), rng.choice(pool_b)) for _ in range(10**5)]
    keys = wide_keys([a for a, _b in rows], [b for _a, b in rows], 2)
    ordered = sort_unique(keys)
    layout = host_w2.HostLayout(keys, 100)
    assert layout.num_rows == ordered.shape[0]
    expected = sorted(set(rows), key=lambda ab: (ab[1], ab[0]))
    assert [(from_words(r[:2]), from_words(r[2:])) for r in ordered] == expected
    op = random_operator(100, 3, active=list(range(56, 72)))
    store = host_w2.HostOperator(op, keys).sorted_keys()
    np.testing.assert_array_equal(store, ordered)


def padded_problem(norb, active, num_alpha, num_beta, count, seed):
    rng = random.Random(seed)
    alpha = random_sectors(rng, list(range(len(active))), num_alpha, count)
    beta = random_sectors(rng, list(range(len(active))), num_beta, count)

    def place(value):
        return sum(1 << active[k] for k in range(len(active)) if (value >> k) & 1)

    return alpha, beta, [place(v) for v in alpha], [place(v) for v in beta]


@pytest.mark.parametrize(
    "norb, active", [(70, list(range(58, 70))), (128, list(range(56, 68)) + [126, 127])]
)
def test_two_word_operator_equals_the_one_word_operator_on_the_active_orbitals(
    norb, active
):
    alpha, beta, wide_alpha, wide_beta = padded_problem(norb, active, 4, 3, 600, norb)
    small = len(active)
    narrow_op = random_operator(small, 11)
    narrow = host_w1.HostOperator(narrow_op, wide_keys(alpha, beta, 1))
    wide_op = random_operator(norb, 11, active=active)
    wide = host_w2.HostOperator(wide_op, wide_keys(wide_alpha, wide_beta, 2))
    assert wide.num_rows == narrow.num_rows
    position = {a: i for i, a in enumerate(active)}

    def squeeze(value):
        return sum(1 << position[o] for o in range(norb) if (value >> o) & 1)

    wide_rows = [
        (squeeze(from_words(r[:2])), squeeze(from_words(r[2:])))
        for r in wide.sorted_keys()
    ]
    narrow_rows = [(int(r[0]), int(r[1])) for r in narrow.sorted_keys()]
    index = {row: i for i, row in enumerate(narrow_rows)}
    order = np.array([index[row] for row in wide_rows])
    x = np.random.default_rng(norb).normal(size=wide.num_rows)
    narrow_x = np.empty_like(x)
    narrow_x[order] = x
    expected = narrow.matvec(narrow_x)[order]
    got = wide.matvec(x)
    assert np.max(np.abs(got - expected)) <= 1e-13 * np.max(np.abs(expected))
    np.testing.assert_allclose(
        wide.diagonal(), narrow.diagonal()[order], rtol=0, atol=1e-12
    )


def wide_labels(keys, norb):
    words = keys.shape[1] // 2
    return [
        format(from_words(row[words:]), f"0{norb}b")
        + format(from_words(row[:words]), f"0{norb}b")
        for row in keys
    ]


def cpu_reference(op, keys, norb):
    import fulqrum as fq  # noqa: PLC0415

    return fq.SubspaceHamiltonian(op, fq.Subspace([wide_labels(keys, norb)]))


def test_two_word_operator_equals_the_core_subspace_hamiltonian():
    rng = random.Random(70)
    norb = 70
    active = list(range(56, 72))[:14]
    op = random_operator(norb, 21, active=active)
    alpha = [
        sum(1 << o for o in rng.sample(active, 4)) | (1 << 3) for _ in range(500)
    ]
    beta = [sum(1 << o for o in rng.sample(active, 3)) for _ in range(500)]
    host = host_w2.HostOperator(op, wide_keys(alpha, beta, 2))
    keys = host.sorted_keys()
    reference = cpu_reference(op, keys, norb)
    x = np.random.default_rng(2).normal(size=host.num_rows)
    expected = reference.matvec(x)
    got = host.matvec(x)
    assert np.max(np.abs(got - expected)) <= 1e-12 * np.max(np.abs(expected))
    np.testing.assert_allclose(
        host.diagonal(),
        np.asarray(reference.diagonal_vector()).real,
        rtol=0,
        atol=1e-12,
    )


def test_key_rows_check_the_width_and_pad_when_forced(monkeypatch):
    from fulqrum.gpu.full_strs.hamiltonian import _key_rows  # noqa: PLC0415

    keys = np.array([[3, 5], [6, 9]], np.uint64)
    monkeypatch.delenv("FQ_FORCE_KEY_WORDS", raising=False)
    np.testing.assert_array_equal(_key_rows(keys, 10), keys)
    with pytest.raises(ValueError, match=r"keys must have shape \(n, 4\) for 70"):
        _key_rows(keys, 70)
    wide = np.array([[3, 1, 5, 2]], np.uint64)
    np.testing.assert_array_equal(_key_rows(wide, 70), wide)
    with pytest.raises(ValueError, match=r"keys must have shape \(n, 2\) for 10"):
        _key_rows(wide, 10)
    monkeypatch.setenv("FQ_FORCE_KEY_WORDS", "2")
    np.testing.assert_array_equal(
        _key_rows(keys, 10), np.array([[3, 0, 5, 0], [6, 0, 9, 0]], np.uint64)
    )
    np.testing.assert_array_equal(_key_rows(wide, 70), wide)


def test_forced_two_words_give_the_same_operator(monkeypatch):
    rng = random.Random(9)
    norb = 12
    alpha = random_sectors(rng, list(range(norb)), 5, 300)
    beta = random_sectors(rng, list(range(norb)), 4, 300)
    op = random_operator(norb, 13)
    narrow = host_w1.HostOperator(op, wide_keys(alpha, beta, 1))
    wide = host_w2.HostOperator(op, wide_keys(alpha, beta, 2))
    np.testing.assert_array_equal(wide.sorted_keys()[:, [0, 2]], narrow.sorted_keys())
    assert not wide.sorted_keys()[:, [1, 3]].any()
    x = np.random.default_rng(1).normal(size=narrow.num_rows)
    np.testing.assert_array_equal(wide.matvec(x), narrow.matvec(x))
    np.testing.assert_array_equal(wide.diagonal(), narrow.diagonal())


@pytest.mark.parametrize(
    "width, ok", [(64, True), (65, True), (128, True), (129, False)]
)
def test_full_string_width_limit(width, ok):
    error = width_error((width, width), max_words=2)
    assert (error is None) == ok
    if not ok:
        assert error == (
            f"sector 0 has {width} orbitals; this build supports at most 128 per sector"
        )


def test_cartesian_width_limit_is_unchanged():
    assert width_error((64, 64)) is None
    assert "at most 64 per sector" in width_error((65, 65))
    assert width_error((128, 128), (64, 128), max_words=2).startswith(
        "sector 1 is fully occupied with 128 orbitals"
    )


def test_check_width_with_two_words_on_every_rank():

    def body(comm):
        widths = (129, 129) if comm.Get_rank() == 1 else (100, 100)
        try:
            check_width(comm, widths, "op", max_words=2)
        except FulqrumError as error:
            return error.message
        return None

    messages = run_ranks(LocalComm.create(2, timeout=60.0), body)
    expected = width_error((129, 129), max_words=2)
    assert messages == [f"op failed: rank 1: {expected}"] * 2


@pytest.mark.parametrize("m", [64, 128])
def test_rdm_buffer_sizes_at_full_width(m):
    expected = 3 * m**4 + 4 * m**3 + 5 * m**2
    from fulqrum.gpu.full_strs.rdm import MAX_ORBITALS, _array_shapes  # noqa: PLC0415

    sizes = _array_shapes(np.int32(m))
    assert MAX_ORBITALS == 128
    assert all(type(v) is int for v in sizes.values())
    assert sum(sizes.values()) == expected


def test_key_words_matches_the_sector_width():
    assert [key_words(n) for n in (1, 64, 65, 100, 128)] == [1, 1, 2, 2, 2]
