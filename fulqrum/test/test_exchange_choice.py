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


import pytest

from fulqrum.gpu.distributed.exchange_plan import choose_exchange, two_level_obstacle

TWO_DOMAINS = ((0, 1), (2, 3))


def test_two_level_without_nccl_or_host_mpi_names_nccl():
    obstacle = two_level_obstacle(False, TWO_DOMAINS)
    assert "needs the NCCL backend" in obstacle
    assert two_level_obstacle(False, TWO_DOMAINS, False, True) == obstacle


def test_two_level_on_host_mpi_needs_an_explicit_request():
    assert "FQ_EXCHANGE=two_level" in two_level_obstacle(False, TWO_DOMAINS, True)
    assert two_level_obstacle(False, TWO_DOMAINS, True, True) is None
    assert two_level_obstacle(True, TWO_DOMAINS) is None


@pytest.mark.parametrize(
    "domains, text",
    [
        (None, "at least two shared-memory domains"),
        (((0, 1, 2, 3),), "at least two shared-memory domains"),
        (((0,), (1,), (2,), (3,)), "two or more ranks"),
    ],
)
def test_two_level_domain_obstacles(domains, text):
    assert text in two_level_obstacle(True, domains)


def test_choose_exchange():
    assert choose_exchange("ring", True, TWO_DOMAINS) == ("ring", "FQ_EXCHANGE=ring")
    name, reason = choose_exchange("auto", False, TWO_DOMAINS)
    assert name == "ring"
    assert "needs the NCCL backend" in reason
    name, reason = choose_exchange("two_level", False, TWO_DOMAINS, True)
    assert (name, reason) == ("two_level", "2 shared-memory domains with host_mpi")
    with pytest.raises(ValueError, match="FQ_EXCHANGE=two_level cannot run"):
        choose_exchange("two_level", False, TWO_DOMAINS)
