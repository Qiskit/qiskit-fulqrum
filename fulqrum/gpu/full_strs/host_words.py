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


from __future__ import annotations

import importlib

import numpy as np

from .. import config
from ..common.wide import key_words

MODULES = {1: ".host", 2: ".host_w2"}


def host_module(key_words):

    words = int(key_words)
    if words not in MODULES:
        raise ValueError(
            f"full-string keys support one or two words per sector, got {words}"
        )
    return importlib.import_module(MODULES[words], __package__)


def _forced_words():

    return int(config.read("FQ_FORCE_KEY_WORDS"))


def key_rows(keys, num_orbitals):

    words = key_words(num_orbitals)
    rows = np.asarray(keys)
    if rows.ndim != 2 or rows.shape[1] != 2 * words:
        raise ValueError(
            f"keys must have shape (n, {2 * words}) for {num_orbitals} orbitals per "
            f"sector ({words} alpha words, then {words} beta words); got {rows.shape}"
        )
    forced = _forced_words()
    if forced <= words:
        return keys
    padded = np.zeros((rows.shape[0], 2 * forced), dtype=rows.dtype)
    padded[:, :words] = rows[:, :words]
    padded[:, forced : forced + words] = rows[:, words:]
    return padded
