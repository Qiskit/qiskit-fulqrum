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

from fulqrum.gpu.distributed.term_tables import (
    _build_grp_pos_tables,
    _build_ht_u32_kv,
    _build_ht_u64_kv,
    _splitmix32_host,
)
from fulqrum.gpu.common.hash_utils import _EMPTY_KEY, _splitmix64

HIGH = [7, (1 << 23) << 8 | 255, 0xFFFFFFFF, ((1 << 24) - 2) << 8]


def _probe(keys, vals, mask, key, hashed, empty):
    slot = hashed(key) & int(mask)
    while keys[slot] != empty:
        if int(keys[slot]) == key:
            return int(vals[slot])
        slot = (slot + 1) & int(mask)
    return None


def test_int32_storage_would_turn_a_large_group_negative():
    assert np.array([0x800000FF], dtype=np.uint32).view(np.int32)[0] < 0


def test_pair_table_stores_packed_values_as_uint32():
    keys = [3 | (5 << 16), 1 | (2 << 16), 4 | (9 << 16), 6 | (7 << 16)]
    ht_keys, ht_vals, mask, _ = _build_ht_u32_kv(keys, HIGH)
    assert ht_vals.dtype == np.uint32
    assert int(ht_vals[ht_keys == np.uint32(0xFFFFFFFF)][0]) == 0xFFFFFFFF
    for key, val in zip(keys, HIGH):
        got = _probe(ht_keys, ht_vals, mask, key, _splitmix32_host, 0xFFFFFFFF)
        assert got == val


def test_quad_table_stores_packed_values_as_uint32():
    keys = [k | (k + 1) << 16 | (k + 2) << 32 | (k + 3) << 48 for k in (1, 5, 9, 13)]
    ht_keys, ht_vals, mask, _ = _build_ht_u64_kv(keys, HIGH)
    assert ht_vals.dtype == np.uint32
    for key, val in zip(keys, HIGH):
        assert _probe(ht_keys, ht_vals, mask, key, _splitmix64, _EMPTY_KEY) == val


def test_group_tables_are_uint32():
    inds = np.array([[0, 1, 0, 0], [0, 1, 2, 3]], dtype=np.uint8)
    sizes = np.array([2, 4], dtype=np.uint8)
    tables = _build_grp_pos_tables(inds, sizes, None, np.array([2, 3]))
    assert tables[1].dtype == np.uint32 and tables[5].dtype == np.uint32
