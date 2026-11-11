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

import json
from pathlib import Path

RECORD = Path(__file__).with_name("cuda_build.json")

_checked = False


def _major(version):

    return int(str(version).split(".")[0])


def mismatch_error(toolkit, runtime):

    if _major(toolkit) == _major(runtime):
        return None
    return (
        f"Fulqrum's GPU module was built with CUDA {toolkit} but CuPy uses CUDA "
        f"{runtime}. Install the matching CuPy (e.g. cupy-cuda{_major(toolkit)}x) "
        "and reinstall Fulqrum without the [gpu] extra, or use a CUDA "
        f"{_major(runtime)} toolkit. See GPU_INSTALL.md."
    )


def built_version(record=None):

    path = RECORD if record is None else Path(record)
    try:
        return str(json.loads(path.read_text())["cuda_toolkit"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def runtime_version():

    import cupy

    version = int(cupy.cuda.runtime.runtimeGetVersion())
    return f"{version // 1000}.{(version % 1000) // 10}"


def check_cuda_runtime():

    global _checked
    if _checked:
        return
    toolkit = built_version()
    if toolkit is not None:
        error = mismatch_error(toolkit, runtime_version())
        if error is not None:
            raise RuntimeError(error)
    _checked = True
