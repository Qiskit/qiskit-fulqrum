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

from fulqrum.gpu.common.wide import sort_strings
from fulqrum.test.test_cartesian_two_words import padded_hamiltonians

tile_tables = pytest.importorskip("fulqrum.gpu.distributed.tile_tables")

from fulqrum.gpu.distributed.tables import values_on_edges  # noqa: E402
from fulqrum.gpu.distributed.tile_discovery import (  # noqa: E402
    _build_tile_flip_csr_dense,
)

WIDTH = 70
ACTIVE = list(range(58, 70))


@pytest.fixture(scope="module")
def pair():
    return padded_hamiltonians(WIDTH, ACTIVE, (4, 3), 40, 11)


def discovered(rows, source, popcount):
    csr = _build_tile_flip_csr_dense(rows, source, popcount)
    count = int(csr[0][-1])
    positions = np.stack([p[:count] for p in csr[2 : 2 + popcount]], axis=1)
    return csr[0], csr[1][:count], positions


@pytest.mark.parametrize("which", [0, 1], ids=["one_word", "two_words"])
@pytest.mark.parametrize("sector", [0, 1])
@pytest.mark.parametrize("part", ["all", "middle"])
@pytest.mark.parametrize("threads", [1, 4])
def test_edge_values_equal_the_joined_sector_tiles(pair, which, sector, part, threads):
    spmv = pair[which].spmv
    strings = sort_strings(spmv.subspace._alpha_beta_words(2)[sector])
    count = len(strings)
    first, last = (0, count) if part == "all" else (count // 3, 2 * count // 3)
    singles = discovered(strings[first:last], strings, 2)
    doubles = discovered(strings[first:last], strings, 4)
    assert len(singles[1]) > 0 and len(doubles[1]) > 0
    tile = tile_tables.sector_tiles(spmv, sector, first, last, [0, count])[0]
    got = tile_tables.edge_values(
        spmv, sector, first, singles[0], singles[2], doubles[0], doubles[2], threads
    )
    for name, host in (
        ("single_values", tile["single_values"]),
        ("own_signs", tile["own_signs"].astype(np.float64)),
        ("jw_signs", tile["jw_signs"].astype(np.float64)),
    ):
        want = values_on_edges(
            singles[0], singles[1], tile["single_offsets"], tile["columns"], host
        )
        assert got[name].astype(np.float64).tobytes() == want.tobytes()
    want = values_on_edges(
        doubles[0],
        doubles[1],
        tile["double_offsets"],
        tile["double_columns"],
        tile["double_values"],
    )
    assert np.count_nonzero(want) > 0
    assert got["double_values"].tobytes() == want.tobytes()


def test_edge_values_refuse_mismatched_inputs(pair):
    spmv = pair[0].spmv
    strings = sort_strings(spmv.subspace._alpha_beta_words(2)[0])
    singles = discovered(strings, strings, 2)
    doubles = discovered(strings, strings, 4)
    with pytest.raises(ValueError, match="positions"):
        tile_tables.edge_values(
            spmv, 0, 0, singles[0], singles[2][:-1], doubles[0], doubles[2]
        )
    with pytest.raises(ValueError, match="same bra strings"):
        tile_tables.edge_values(
            spmv, 0, 0, singles[0], singles[2], doubles[0][:-1], doubles[2]
        )
    with pytest.raises(ValueError, match="bra range"):
        tile_tables.edge_values(
            spmv, 0, 1, singles[0], singles[2], doubles[0], doubles[2]
        )
