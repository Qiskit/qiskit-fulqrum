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

"""Grid, discovery, exchange-plan and rank-agreement checks on four in-process ranks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from .conftest import gpu_available, run_local_ranks

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not gpu_available(), reason="GPU build / CUDA device unavailable"
    ),
]

SPLIT_STRINGS = np.array([0b000_000, 0b000_111, 0b111_000, 0b111_111], dtype=np.uint64)


def _tiny_hsub(flip_masks=None):
    spmv = SimpleNamespace(is_real=True)
    if flip_masks is not None:
        spmv.flip_masks = flip_masks
    return SimpleNamespace(spmv=spmv)


def _singles_coupling():
    pair = [(np.uint64(0b11), (np.uint8(0), np.uint8(1)))]
    return _tiny_hsub({"alpha2": pair, "alpha4": [], "beta2": pair, "beta4": []})


def _no_coupling():
    return _tiny_hsub({"alpha2": [], "alpha4": [], "beta2": [], "beta4": []})


def _build(comm, n_a=4, n_b=4, hsub=None, alpha=None, beta=None, **kwargs):
    from fulqrum.gpu.distributed.dist_matvec_tile_resident import (  # noqa: PLC0415
        TileGPUResidentHamiltonian,
    )

    return TileGPUResidentHamiltonian(
        _tiny_hsub() if hsub is None else hsub,
        np.arange(n_a, dtype=np.uint64) if alpha is None else alpha,
        np.arange(n_b, dtype=np.uint64) if beta is None else beta,
        comm=comm,
        process_grid=(2, 2),
        comm_backend="inproc",
        **kwargs,
    )


def _on_four(body):
    return run_local_ranks(4, body)


def test_alpha_beta_subcomms_partition_the_2x2_grid():
    def body(comm):
        H = _build(comm)
        rank = comm.Get_rank()
        assert H.process_grid == (2, 2)
        assert (H.coord_alpha, H.coord_beta) == (rank // 2, rank % 2)
        assert H._alpha_comm.Get_size() == 2
        assert H._beta_comm.Get_size() == 2
        assert H._alpha_rank == H.coord_beta
        assert H._beta_rank == H.coord_alpha
        assert set(H._alpha_comm.allgather(H.coord_alpha)) == {H.coord_alpha}
        assert set(H._beta_comm.allgather(H.coord_beta)) == {H.coord_beta}
        return H.coord_alpha, H.coord_beta

    assert _on_four(body) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_uneven_split_pins_the_global_peer_max():
    def body(comm):
        H = _build(comm, n_a=5, n_b=6)
        assert H._alpha_ranges == [(0, 3), (3, 5)]
        assert H._beta_ranges == [(0, 3), (3, 6)]
        assert (H._n_a_peer_max, H._n_b_peer_max) == (3, 3)
        return H.n_a_local, H.n_b_local

    assert _on_four(body) == [(3, 3), (3, 3), (2, 3), (2, 3)]


def test_mixed_neighbors_at_2x2_with_singles_coupling():
    def body(comm):
        H = _build(comm)
        assert (H._mixed_indegree, H._mixed_outdegree) == (1, 1)
        assert H._mixed_sources == H._mixed_destinations
        return H._mixed_sources

    assert _on_four(body) == [[3], [2], [1], [0]]


def test_mixed_neighbors_empty_at_2x2_with_no_coupling():
    def body(comm):
        H = _build(comm, alpha=SPLIT_STRINGS, beta=SPLIT_STRINGS.copy())
        assert H._mixed_destinations == []
        assert (H._mixed_indegree, H._mixed_outdegree) == (0, 0)
        return H._mixed_sources

    assert _on_four(body) == [[], [], [], []]


def test_discovery_cross_check_sees_every_class_at_2x2():
    from fulqrum.gpu.distributed.dist_matvec import TileClass  # noqa: PLC0415

    def body(comm):
        H = _build(comm)
        assert H._sources_2d == H._source_set.reference
        return set(H._sources_2d.values())

    every = {TileClass.DIAGONAL, TileClass.ALPHA_ONLY, TileClass.BETA_ONLY}
    assert _on_four(body) == [every | {TileClass.MIXED}] * 4


def test_discovery_with_hand_rolled_flip_masks():
    from fulqrum.gpu.distributed.dist_matvec import TileClass  # noqa: PLC0415

    def body(comm):
        H = _build(comm, hsub=_singles_coupling())
        assert H._sources_2d == H._source_set.reference
        diagonal = [r for r, c in H._sources_2d.items() if c == TileClass.DIAGONAL]
        return len(H._sources_2d), set(H._sources_2d.values()), diagonal

    classes = {
        TileClass.DIAGONAL,
        TileClass.ALPHA_ONLY,
        TileClass.BETA_ONLY,
        TileClass.MIXED,
    }
    assert _on_four(body) == [(4, classes, [rank]) for rank in range(4)]


def test_discovery_without_coupling_returns_only_the_diagonal():
    from fulqrum.gpu.distributed.dist_matvec import TileClass  # noqa: PLC0415

    def body(comm):
        H = _build(
            comm, hsub=_no_coupling(), alpha=SPLIT_STRINGS, beta=SPLIT_STRINGS.copy()
        )
        assert H._sources_2d == H._source_set.reference
        return H._sources_2d

    assert _on_four(body) == [{rank: TileClass.DIAGONAL} for rank in range(4)]


def test_explicit_two_level_without_a_device_backend_raises_on_every_rank(
    monkeypatch,
):
    monkeypatch.setenv("FQ_EXCHANGE", "two_level")
    monkeypatch.delenv("_FULQRUM_TWO_LEVEL_ANY_BACKEND", raising=False)

    def body(comm):
        with pytest.raises(ValueError, match="needs the NCCL backend") as info:
            _build(comm)
        return "FQ_EXCHANGE=two_level cannot run" in str(info.value)

    assert _on_four(body) == [True] * 4


def test_exchange_mode_differing_across_ranks_raises_on_every_rank(monkeypatch):
    from fulqrum.gpu.config import GpuConfig  # noqa: PLC0415

    monkeypatch.delenv("FQ_EXCHANGE", raising=False)

    def body(comm):
        overrides = {"FQ_EXCHANGE": "ring"} if comm.Get_rank() == 1 else {}
        config = GpuConfig.from_environ(overrides=overrides)
        with pytest.raises(ValueError, match="differs? across ranks"):
            _build(comm, config=config)
        return True

    assert _on_four(body) == [True] * 4


def test_auto_exchange_on_inproc_runs_ring_with_one_digest(monkeypatch):
    monkeypatch.delenv("FQ_EXCHANGE", raising=False)

    def body(comm):
        timing = _build(comm).construction_timing
        return timing["exchange"], timing["exchange_reason"], timing["exchange_digest"]

    results = _on_four(body)
    assert {name for name, _, _ in results} == {"ring"}
    assert all("needs the NCCL backend" in reason for _, reason, _ in results)
    assert len({digest for _, _, digest in results}) == 1


def test_verify_hbm_fits_failing_on_one_rank_raises_on_every_rank():
    def body(comm):
        H = _build(comm)
        failing = comm.Get_rank() == 2
        H._hbm_fit_error = lambda: "too small" if failing else None
        with pytest.raises(RuntimeError, match="rank 2: too small"):
            H.verify_hbm_fits()
        return True

    assert _on_four(body) == [True] * 4


@pytest.mark.parametrize("double_buffer", ["1", "0"])
def test_exchange_steps_match_the_ring_and_window_plans(monkeypatch, double_buffer):
    """The ring steps visit every other member of the row or column in order, and the window steps follow the hops."""
    from fulqrum.gpu.distributed.exchange_plan import (  # noqa: PLC0415
        FORWARD,
        SEND_LOCAL,
    )

    monkeypatch.setenv("FQ_EXCHANGE_DOUBLE_BUFFER", double_buffer)

    def body(comm):
        H = _build(comm)
        for steps, members, index in (
            (H._beta_ring_steps, list(H._alpha_comm_world_ranks), H._alpha_rank),
            (H._alpha_ring_steps, list(H._beta_comm_world_ranks), H._beta_rank),
        ):
            size = len(members)
            assert size == 2
            peers = [members[(index + k) % size] for k in range(1, size)]
            send_peers = [members[(index - k) % size] for k in range(1, size)]
            left, right = members[(index - 1) % size], members[(index + 1) % size]
            assert [step.source for step in steps] == peers
            if double_buffer == "1":
                assert all(
                    (s.transfer, s.send_to, s.receive_from) == (FORWARD, left, right)
                    for s in steps
                )
            else:
                assert all(s.transfer == SEND_LOCAL for s in steps)
                assert [step.send_to for step in steps] == send_peers
        assert [(s.send_to, s.receive_from) for s in H._mixed_steps] == list(
            H._mixed_window_hops
        )
        return len(H._mixed_steps)

    assert all(count > 0 for count in _on_four(body))
