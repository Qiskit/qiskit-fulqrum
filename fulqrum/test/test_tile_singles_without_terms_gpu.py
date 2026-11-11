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

import numpy as np
import pytest

import fulqrum as fq
from fulqrum.test.test_full_strs_coefficients import random_operator

from .conftest import gpu_available

pytestmark = [
    pytest.mark.skipif(
        not gpu_available(), reason="GPU build / CUDA device unavailable"
    ),
    pytest.mark.gpu,
]

WIDTH = 20
ACTIVE = list(range(12))
OUTSIDE = 15


def strings():
    out = []
    for occupied in itertools.combinations(ACTIVE, 6):
        out.append(sum(1 << orbital for orbital in occupied))
    for dropped in ACTIVE[:6]:
        kept = [orbital for orbital in ACTIVE[:6] if orbital != dropped]
        out.append(sum(1 << orbital for orbital in kept) | (1 << OUTSIDE))
    return sorted(format(value, f"0{WIDTH}b") for value in out)


@pytest.mark.parametrize("gpu_ids", [(0,), (0, 0)], ids=["1", "2"])
def test_strings_joined_by_a_pair_without_terms(gpu_ids):
    from fulqrum.gpu import GPUHamiltonian

    half = strings()
    operator = random_operator(WIDTH, 3, active=ACTIVE)
    operator.set_type(2)
    hsub = fq.SubspaceHamiltonian(operator, fq.Subspace([half, half]))
    op = GPUHamiltonian(hsub, gpu_ids=gpu_ids)
    x = np.random.default_rng(1).standard_normal(op.shape[0])
    expected = hsub.matvec(x)
    scale = np.abs(expected).max()
    assert scale > 0
    assert np.abs(op @ x - expected).max() <= 1e-10 * scale
