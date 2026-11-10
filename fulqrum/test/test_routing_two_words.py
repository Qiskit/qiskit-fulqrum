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


import dataclasses
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

from fulqrum.gpu.common.wide import width_error, with_words
from fulqrum.gpu.distributed import launch
from fulqrum.gpu.distributed.arm_plan import (
    ArmSettings,
    SourceFacts,
    plan_arms,
)

RECIPE = ArmSettings(
    factored=True, parent=True, remote_beta="clique", remote_alpha=True
)
WIDE = dataclasses.replace(RECIPE, wide=True)
SWITCHED_OFF = dataclasses.replace(
    RECIPE, parent=False, remote_beta="baseline", remote_alpha=False
)
FLAGS = (
    "is_bra",
    "same_alpha",
    "same_beta",
    "parent_built",
    "parent_plan",
    "parent_grouped_edges",
    "remote_beta_plan",
    "remote_beta_grouped_edges",
    "remote_alpha_plan",
)


def every_source():
    for values in itertools.product((False, True), repeat=len(FLAGS)):
        yield SourceFacts(
            singles_edges=True,
            singles_sell=True,
            doubles_edges=True,
            doubles_sell=True,
            **dict(zip(FLAGS, values)),
        )


@pytest.mark.parametrize("settings", [RECIPE, SWITCHED_OFF])
def test_two_word_sources_never_take_the_baseline_path(settings):
    wide = dataclasses.replace(settings, wide=True)
    for facts in every_source():
        narrow, plan = plan_arms(facts, settings), plan_arms(facts, wide)
        assert plan.aabb_path != "baseline"
        if narrow.aabb_path == "baseline":
            assert plan.aabb_path == "direct"
        else:
            assert plan == narrow


def test_single_word_plans_are_unchanged_by_the_default_setting():
    assert (
        ArmSettings(factored=True, parent=True, remote_beta="clique", remote_alpha=True)
        == RECIPE
    )
    assert not RECIPE.wide
    for facts in every_source():
        assert plan_arms(facts, RECIPE).aabb_path != "direct"


def test_sources_without_mixed_spin_work_have_no_aabb_arm():
    facts = next(every_source())
    facts = dataclasses.replace(facts, singles_edges=False, singles_sell=False)
    plan = plan_arms(facts, WIDE)
    assert plan.aabb_mode == 0


def test_two_word_sources_need_the_factored_tables():
    with pytest.raises(ValueError, match="factored"):
        plan_arms(next(every_source()), dataclasses.replace(WIDE, factored=False))


def test_cache_build_refuses_more_orbitals_than_its_buffer():
    for num_alpha, num_beta in ((65, 10), (10, 65), (128, 128)):
        with pytest.raises(ValueError, match="at most 64 orbitals"):
            launch.check_factor_orbitals((0, num_alpha, num_beta))
    launch.check_factor_orbitals((0, 64, 64))


def test_cache_build_limit_matches_the_kernel_buffer():
    header = (
        Path(launch.__file__).resolve().parents[1]
        / "kernels"
        / "arms"
        / "term_walk.cuh"
    )
    if not header.is_file():
        pytest.skip("the CUDA sources are not installed with the runtime package")
    found = re.search(r"kFactorMaxOrbitals\s*=\s*(\d+)", header.read_text())
    assert found and int(found.group(1)) == launch.FACTOR_MAX_ORBITALS


@pytest.mark.parametrize(
    "width,ok", [(64, True), (65, True), (128, True), (129, False)]
)
def test_cartesian_gates_accept_two_words(width, ok):
    assert (width_error((width, 10), max_words=2) is None) == ok
    assert (width_error((10, width), max_words=2) is None) == ok


def test_operator_strings_widen_to_the_forced_word_count():
    one = np.array([1, 2**63, 5], dtype=np.uint64)
    assert with_words(one, 1) is not None and with_words(one, 1).shape == (3,)
    two = with_words(one, 2)
    assert two.shape == (3, 2) and two[:, 0].tolist() == one.tolist()
    assert not two[:, 1].any()
    column = one.reshape(-1, 1)
    assert with_words(column, 2).tolist() == two.tolist()
    already = np.array([[1, 7]], dtype=np.uint64)
    assert with_words(already, 2) is not None
    assert with_words(already, 2).tolist() == [[1, 7]]
    with pytest.raises(ValueError, match="words"):
        with_words(np.zeros((2, 3), dtype=np.uint64), 2)


def test_direct_cross_launches_the_direct_kernel():
    calls = []

    class Kernels:
        def launch_cross_direct(self, *args):
            calls.append(("direct", args))

        def launch_cross_sell(self, *args):
            calls.append(("sell", args))

    class Device:
        def __init__(self, ptr, dtype="int64"):
            self.dtype = dtype
            self.data = type("Pointer", (), {"ptr": ptr})()
            self.ndim = 1

    view = {
        "d_alpha2_sell_off": Device(1),
        "d_alpha2_sell_len": Device(2, "int32"),
        "d_alpha2_sell_pack": Device(3, "uint64"),
        "d_beta2_ptr": Device(4),
        "d_beta2_j": Device(5, "int32"),
        "d_beta2_p0": Device(6, "uint8"),
        "d_beta2_p1": Device(7, "uint8"),
    }
    cross = launch.DirectCross(
        beta_words=Device(8, "uint64"), tasks=Device(9, "int32"), num_tasks=5
    )
    two_word = Device(10, "uint64")
    two_word.ndim = 2
    source = launch.SourceLaunch(
        geometry=(0,) * 14,
        src_in=11,
        out=12,
        alpha_slot=0,
        beta_slot=0,
        stream=13,
        same_alpha=False,
        same_beta=False,
        view=view,
        caches={},
        pair=None,
        rows={},
        num_alpha=6,
        num_beta=6,
        alpha_strings=two_word,
        beta_strings=two_word,
        coefficients=Device(14, "float64"),
        cross=cross,
    )
    launch._cross(Kernels(), source)
    assert [name for name, _ in calls] == ["direct"]
    args = calls[0][1]
    assert args[3:9] == (1, 2, 3, 8, 9, 14)
    assert args[9:12] == (15, 5, 13)
    source.cross = "sell"
    with pytest.raises(ValueError, match="single-word"):
        launch._cross(Kernels(), source)
