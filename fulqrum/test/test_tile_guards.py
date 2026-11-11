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


from types import SimpleNamespace

import pytest

from fulqrum.gpu.distributed.dist_matvec_tile_resident import (
    TileGPUResidentHamiltonian,
)


@pytest.mark.parametrize("threads", [0, 16, 48, 288, 512, 1024])
def test_invalid_threads_per_block_is_rejected_before_any_work(threads):
    hsub = SimpleNamespace(spmv=SimpleNamespace(is_real=True))
    with pytest.raises(ValueError, match="threads_per_block"):
        TileGPUResidentHamiltonian(
            hsub, [], [], threads_per_block=threads, comm_backend="inproc"
        )


def test_slot_compaction_is_rejected_with_the_arm_name():
    op = object.__new__(TileGPUResidentHamiltonian)
    alpha, beta = object(), object()
    op._d_alpha_slot_persistent, op._d_beta_slot_persistent = alpha, beta
    op._identity_slots = (alpha, beta)
    op._check_identity_slots("source arms")
    op._d_alpha_slot_persistent = object()
    with pytest.raises(RuntimeError, match="source arms: the source slot arrays"):
        op._check_identity_slots("source arms")
