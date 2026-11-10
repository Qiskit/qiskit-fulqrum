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
from types import SimpleNamespace

import numpy as np
import pytest

from fulqrum.gpu.common import wide
from fulqrum.gpu.full_strs import tile_dispatch


class Ranks:
    """A fake communicator over in-memory per-rank values, for one rank's view."""

    def __init__(self, shared):
        self.shared = shared
        self.gathers = 0

    def allgather(self, _value):
        self.gathers += 1
        return self.shared.pop(0)

    def Get_size(self):
        return 3


def key_rows(pairs, words):
    rows = np.zeros((len(pairs), 2 * words), dtype=np.uint64)
    for i, (a, b) in enumerate(pairs):
        for w in range(words):
            rows[i, w] = (a >> (64 * w)) & ((1 << 64) - 1)
            rows[i, words + w] = (b >> (64 * w)) & ((1 << 64) - 1)
    return rows


def blocks_of(pairs, words, parts):
    pairs = sorted(pairs, key=lambda ab: (ab[1], ab[0]))
    cuts = [len(pairs) * r // parts for r in range(parts + 1)]
    return [
        SimpleNamespace(keys=key_rows(pairs[c0:c1], words), num_rows=len(pairs))
        for c0, c1 in itertools.pairwise(cuts)
    ]


def detect(pairs, words, parts=3):
    blocks = blocks_of(pairs, words, parts)
    halves = [tile_dispatch.halves(b.keys, words) for b in blocks]
    alpha_sets = [wide.unique_w(a)[0] for a, _ in halves]
    beta_sets = [wide.unique_w(b)[0] for _, b in halves]
    counts = [(len(a), len(b)) for a, b in zip(alpha_sets, beta_sets)]
    comm = Ranks([counts, alpha_sets, beta_sets])
    return tile_dispatch.product_halves(comm, blocks[0], words)


def test_impossible_products_skip_the_string_gather():
    block = SimpleNamespace(keys=key_rows([(1, 2), (3, 4)], 1), num_rows=2)
    comm = Ranks([[(2, 2), (1, 1), (1, 1)]])
    assert tile_dispatch.product_halves(comm, block, 1) is None
    assert comm.gathers == 1
    comm = Ranks([[(1, 1), (1, 1)]])
    big = SimpleNamespace(keys=key_rows([(1, 2)], 1), num_rows=9)
    assert tile_dispatch.product_halves(comm, big, 1) is None
    assert comm.gathers == 1


@pytest.mark.parametrize("words,n", [(1, 20), (2, 100)])
def test_a_complete_product_is_detected(words, n):
    rng = random.Random(words)
    alpha = sorted({rng.getrandbits(n) | (1 << (n - 1)) for _ in range(7)})
    beta = sorted({rng.getrandbits(n) for _ in range(5)})
    pairs = list(itertools.product(alpha, beta))
    found = detect(pairs, words)
    assert found is not None
    got_alpha, got_beta = found
    assert wide.unpack_ints(got_alpha) == alpha
    assert wide.unpack_ints(got_beta) == beta


@pytest.mark.parametrize("words,n", [(1, 20), (2, 100)])
def test_a_product_missing_one_key_is_not_a_product(words, n):
    rng = random.Random(10 + words)
    alpha = sorted({rng.getrandbits(n) | (1 << (n - 1)) for _ in range(7)})
    beta = sorted({rng.getrandbits(n) for _ in range(5)})
    pairs = list(itertools.product(alpha, beta))
    pairs.pop(11)
    assert detect(pairs, words) is None


def test_beta_cuts_follow_the_tile_split():
    from fulqrum.gpu.distributed.dist_matvec import _split_ranges_even

    for n_a, n_b, size in ((5, 7, 3), (4, 8, 4), (3, 3, 3), (6, 10, 1)):
        cuts = tile_dispatch.beta_cuts(n_a, n_b, size)
        ranges = _split_ranges_even(n_b, size)
        assert cuts.tolist() == [n_a * s for s, _ in ranges] + [n_a * n_b]
        assert np.all(np.diff(cuts) >= n_a)


def test_the_cuts_give_each_rank_its_tile_rows_in_order():
    n_a, n_b, size = 4, 7, 3
    alpha = list(range(10, 10 + n_a))
    beta = list(range(100, 100 + n_b))
    rows = sorted(itertools.product(alpha, beta), key=lambda ab: (ab[1], ab[0]))
    cuts = tile_dispatch.beta_cuts(n_a, n_b, size)
    from fulqrum.gpu.distributed.dist_matvec import _split_ranges_even

    for rank, (b0, b1) in enumerate(_split_ranges_even(n_b, size)):
        tile = [(alpha[a], beta[b]) for b in range(b0, b1) for a in range(n_a)]
        assert rows[cuts[rank] : cuts[rank + 1]] == tile


def test_halves_and_bit_strings_round_trip():
    rng = random.Random(3)
    values = [(rng.getrandbits(100), rng.getrandbits(100)) for _ in range(9)]
    keys = key_rows(values, 2)
    alpha, beta = tile_dispatch.halves(keys, 2)
    assert alpha.shape == (9, 2) and beta.shape == (9, 2)
    assert wide.unpack_ints(alpha) == [a for a, _ in values]
    strings = tile_dispatch.bit_strings(beta, 100)
    assert [int(s, 2) for s in strings] == [b for _, b in values]
    assert all(len(s) == 100 for s in strings)
    narrow = tile_dispatch.halves(key_rows([(5, 9)], 1), 1)
    assert narrow[0].shape == (1,) and narrow[1].tolist() == [9]


class Config(dict):
    def value(self, name):
        defaults = {"FQ_ARM_TABLE_SOURCE": "auto", "FQ_DIAG_RECOMPUTE": False}
        if name == "FQ_DIAG_RECOMPUTE" and name in self:
            return self[name] == "1"
        return self.get(name, defaults[name])


@pytest.mark.parametrize(
    "settings,wide,refused",
    [
        ({}, True, False),
        ({"FQ_BB_PAIR_TABLE": "0"}, True, True),
        ({"FQ_AA_PAIR_TABLE": "0"}, True, True),
        ({"FQ_ARM_TABLE_SOURCE": "kernel"}, True, True),
        ({"FQ_DIAG_RECOMPUTE": "1"}, True, True),
        ({"FQ_BB_PAIR_TABLE": "0"}, False, False),
        ({"FQ_ARM_TABLE_SOURCE": "kernel"}, False, False),
        ({"FQ_DIAG_RECOMPUTE": "1"}, False, False),
    ],
)
def test_settings_the_tile_engine_refuses_for_two_words(settings, wide, refused):
    reason = tile_dispatch._setting_error(Config(settings), wide)
    assert (reason is not None) == refused


@pytest.mark.parametrize(
    "source,wide,reason,refused",
    [
        ("auto", True, "needs X", True),
        ("auto", False, "needs X", False),
        ("host", False, "needs X", True),
        ("kernel", False, "needs X", False),
        ("auto", True, None, False),
    ],
)
def test_host_arm_reasons_refuse_where_the_tile_engine_raises(
    monkeypatch, source, wide, reason, refused
):
    import sys
    import types

    seen = []

    def fake(spmv, num_alpha, alpha, beta, *flags):
        seen.append((alpha.ndim, flags))
        return reason

    stand_in = types.ModuleType("fulqrum.gpu.distributed.dist_matvec_tile_resident")
    stand_in.host_arm_reason = fake
    monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)
    strings = np.zeros((3, 2), np.uint64)
    got = tile_dispatch._arm_error(
        Config({"FQ_ARM_TABLE_SOURCE": source}), None, 60, strings, strings, wide, True
    )
    assert (got is not None) == refused
    if source != "kernel":
        assert seen and seen[0][0] == 1
