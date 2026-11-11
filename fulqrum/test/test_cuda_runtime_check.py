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


import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from fulqrum.gpu import cuda_runtime

NVCC_12 = """nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
"""

NVCC_13 = """nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
"""


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    record = tmp_path / "cuda_build.json"
    monkeypatch.setattr(cuda_runtime, "RECORD", record)
    monkeypatch.setattr(cuda_runtime, "_checked", False)
    state = {"version": 12080}
    cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(runtimeGetVersion=lambda: state["version"])
    )
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(cuda=cuda))
    return record, state


def test_matching_major_passes(runtime):
    record, state = runtime
    record.write_text(json.dumps({"cuda_toolkit": "12.4"}))
    state["version"] = 12080
    cuda_runtime.check_cuda_runtime()
    assert cuda_runtime._checked


def test_mismatched_major_names_both_versions(runtime):
    record, state = runtime
    record.write_text(json.dumps({"cuda_toolkit": "12.8"}))
    state["version"] = 13000
    with pytest.raises(RuntimeError) as info:
        cuda_runtime.check_cuda_runtime()
    message = str(info.value)
    assert "CUDA 12.8" in message
    assert "CUDA 13.0" in message
    assert "cupy-cuda12x" in message
    assert "GPU_INSTALL.md" in message
    assert not cuda_runtime._checked


def test_missing_record_skips(runtime):
    _, state = runtime
    state["version"] = 13000
    cuda_runtime.check_cuda_runtime()
    assert cuda_runtime._checked


def test_unreadable_record_skips(runtime):
    record, state = runtime
    record.write_text("{not json")
    state["version"] = 13000
    cuda_runtime.check_cuda_runtime()
    assert cuda_runtime._checked


def test_check_runs_once(runtime):
    record, state = runtime
    record.write_text(json.dumps({"cuda_toolkit": "12.8"}))
    cuda_runtime.check_cuda_runtime()
    state["version"] = 13000
    cuda_runtime.check_cuda_runtime()


def test_runtime_version_format(runtime):
    _, state = runtime
    state["version"] = 12060
    assert cuda_runtime.runtime_version() == "12.6"
    state["version"] = 13010
    assert cuda_runtime.runtime_version() == "13.1"


@pytest.fixture
def build():
    path = Path(__file__).resolve().parents[2] / "gpu_build.py"
    if not path.is_file():
        pytest.skip("gpu_build.py is not in this tree")
    spec = importlib.util.spec_from_file_location("gpu_build_under_cuda_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nvcc_parser(build):
    assert build.parse_nvcc_version(NVCC_12) == "12.8"
    assert build.parse_nvcc_version(NVCC_13) == "13.0"
    with pytest.raises(RuntimeError):
        build.parse_nvcc_version("nvcc: command output without a release")


def test_build_check_stops_on_mismatch(build, monkeypatch):
    cuda = types.SimpleNamespace(
        runtime=types.SimpleNamespace(runtimeGetVersion=lambda: 13000)
    )
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match="CUDA 12.8 but CuPy uses CUDA 13.0"):
        build._check_build_cupy("12.8")
    build._check_build_cupy("13.0")


def test_build_check_skips_without_cupy(build, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    build._check_build_cupy("12.8")
