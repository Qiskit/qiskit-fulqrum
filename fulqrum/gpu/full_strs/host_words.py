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

MODULES = {1: ".host", 2: ".host_w2"}


def host_module(key_words):

    words = int(key_words)
    if words not in MODULES:
        raise ValueError(
            f"full-string keys support one or two words per sector, got {words}"
        )
    return importlib.import_module(MODULES[words], __package__)
