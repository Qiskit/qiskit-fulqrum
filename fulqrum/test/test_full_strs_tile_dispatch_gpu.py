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
from fulqrum.test.test_full_strs_two_words import wide_keys

pytestmark = pytest.mark.gpu

ACTIVE = list(range(58, 70))


def sector(rng, orbitals, electrons, count):
    out = set()
    while len(out) < count:
        out.add(sum(1 << o for o in rng.sample(orbitals, electrons)))
    return sorted(out)


def product_case(norb, seed, n_alpha=12, n_beta=10):
    from fulqrum.test.test_full_strs_coefficients import (
        random_operator,
    )

    rng = random.Random(seed)
    orbitals = list(range(norb)) if norb <= 64 else ACTIVE
    alpha = sector(rng, orbitals, 4, n_alpha)
    beta = sector(rng, orbitals, 3, n_beta)
    pairs = list(itertools.product(alpha, beta))
    rng.shuffle(pairs)
    words = 1 if norb <= 64 else 2
    keys = wide_keys([a for a, _ in pairs], [b for _, b in pairs], words)
    active = None if norb <= 64 else ACTIVE
    return random_operator(norb, seed, active=active), keys


def run(op, keys, ranks, engine, **options):
    parts = np.array_split(keys, ranks)
    rng = np.random.default_rng(9)
    state = {
        row: float(rng.normal()) for row in map(tuple, np.unique(keys, axis=0).tolist())
    }

    def compute(comm):
        import cupy

        from fulqrum.gpu.full_strs import FullStrsHamiltonian

        cupy.cuda.Device(0).use()
        H = FullStrsHamiltonian(
            op, parts[comm.Get_rank()], comm=comm, engine=engine, **options
        )
        rows = [tuple(row) for row in H.keys.tolist()]
        x = np.array([state[row] for row in rows])
        y = H.matvec(x)
        return {
            "engine": H.engine,
            "rows": rows,
            "row_start": H.row_start,
            "n_local": H.n_local,
            "y": dict(zip(rows, y.tolist())),
            "diag": dict(zip(rows, np.asarray(H.diag_local).tolist())),
            "rdm1": H.compute_rdms(x, rank=1),
            "rdm2": H.compute_rdms(x, rank=2),
            "tile": None
            if H._tile is None
            else (H._tile._alpha_bra.copy(), H._tile._beta_bra.copy()),
        }

    return run_ranks(LocalComm.create(ranks, timeout=600.0), compute)


def merged(results, field):
    out = {}
    for result in results:
        out.update(result[field])
    return out


def close(a, b, tolerance=1e-13):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() <= tolerance * max(np.abs(b).max(), 1.0)


def rows_of_words(row, words):
    alpha = sum(int(row[w]) << (64 * w) for w in range(words))
    beta = sum(int(row[words + w]) << (64 * w) for w in range(words))
    return alpha, beta


@requires_gpu
@pytest.mark.parametrize("norb", [20, 70], ids=["W1", "W2"])
@pytest.mark.parametrize("ranks", [1, 3])
def test_complete_products_use_the_tile_engine_with_equal_results(norb, ranks):
    op, keys = product_case(norb, norb + ranks)
    tile = run(op, keys, ranks, "auto")
    general = run(op, keys, ranks, "general")
    assert all(r["engine"] == "tile" for r in tile)
    assert all(r["engine"] == "general" for r in general)
    for results in (tile, general):
        rows = [row for result in results for row in result["rows"]]
        assert len(rows) == len(set(rows)) == keys.shape[0]
        starts = [r["row_start"] for r in results]
        assert starts == sorted(starts) and starts[0] == 0
        assert all(r["n_local"] == len(r["rows"]) for r in results)
    assert set(merged(tile, "y")) == set(merged(general, "y"))
    y_tile, y_general = merged(tile, "y"), merged(general, "y")
    rows = sorted(y_general)
    assert close([y_tile[r] for r in rows], [y_general[r] for r in rows])
    d_tile, d_general = merged(tile, "diag"), merged(general, "diag")
    assert close([d_tile[r] for r in rows], [d_general[r] for r in rows])
    for result in tile:
        g = general[0]
        assert all(close(a, b) for a, b in zip(result["rdm1"], g["rdm1"]))
        for a, b in zip(
            (*result["rdm2"][0], *result["rdm2"][1]), (*g["rdm2"][0], *g["rdm2"][1])
        ):
            assert close(a, b)


@requires_gpu
@pytest.mark.parametrize("norb", [20, 70], ids=["W1", "W2"])
def test_rank_keys_are_the_tile_beta_major_enumeration(norb):
    op, keys = product_case(norb, 5)
    words = 1 if norb <= 64 else 2
    for result in run(op, keys, 3, "tile"):
        alpha, beta = result["tile"]
        from fulqrum.gpu.common.wide import unpack_ints

        enumeration = [(a, b) for b in unpack_ints(beta) for a in unpack_ints(alpha)]
        assert [rows_of_words(row, words) for row in result["rows"]] == enumeration


@requires_gpu
def test_a_product_missing_one_key_uses_the_general_engine():
    op, keys = product_case(20, 8)
    results = run(op, keys[:-1], 2, "auto")
    assert all(r["engine"] == "general" for r in results)
    with pytest.raises(ValueError, match="not a complete product"):
        run(op, keys[:-1], 2, "tile")


@requires_gpu
def test_non_default_options_use_the_general_engine():
    op, keys = product_case(20, 9)
    results = run(op, keys, 2, "auto", num_slices=2)
    assert all(r["engine"] == "general" for r in results)
    with pytest.raises(ValueError, match="no general-engine options"):
        run(op, keys, 2, "tile", num_slices=2)


@requires_gpu
def test_an_unknown_engine_is_refused():
    op, keys = product_case(20, 10)
    with pytest.raises(ValueError, match="engine must be one of"):
        run(op, keys, 1, "fast")


@requires_gpu
@pytest.mark.parametrize(
    "name,value",
    [("FQ_BB_PAIR_TABLE", "0"), ("FQ_AA_PAIR_TABLE", "0"), ("FQ_DIAG_RECOMPUTE", "1")],
)
def test_two_word_settings_the_tile_engine_refuses_use_the_general_engine(
    monkeypatch, name, value
):
    op, keys = product_case(70, 12)
    monkeypatch.setenv(name, value)
    results = run(op, keys, 1, "auto")
    assert all(r["engine"] == "general" for r in results)
    with pytest.raises(ValueError, match=name):
        run(op, keys, 1, "tile")
