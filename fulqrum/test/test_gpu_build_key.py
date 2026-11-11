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
from pathlib import Path

import pytest


@pytest.fixture
def build(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "gpu_build.py"
    if not path.is_file():
        pytest.skip("gpu_build.py is not in this tree")
    spec = importlib.util.spec_from_file_location("gpu_build_under_key_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "root"
    (root / "patches/primme").mkdir(parents=True)
    (root / "patches/primme/a.patch").write_text("first\n")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *args, **kwargs: "cc 1.0\n"
    )
    blas = tmp_path / "libblas.so"
    blas.write_bytes(b"blas")
    return module, root, str(blas)


def _source(tmp_path, git):
    source = tmp_path / "primme"
    (source / "src").mkdir(parents=True)
    (source / "src/main.c").write_text("int main;\n")
    if git:
        (source / ".git").write_text("gitdir: elsewhere\n")
    return source


def _key(build, source):
    module, _, blas = build
    return module._primme_build_key(source, None, blas)


def test_git_checkout_key_ignores_submodule_edits(build, tmp_path):
    source = _source(tmp_path, git=True)
    before = _key(build, source)
    (source / "src/main.c").write_text("int edited;\n")
    (source / "src/new.c").write_text("int added;\n")
    assert _key(build, source) == before


def test_patch_changes_change_the_key(build, tmp_path):
    _, root, _ = build
    source = _source(tmp_path, git=True)
    before = _key(build, source)
    (root / "patches/primme/a.patch").write_text("edited\n")
    edited = _key(build, source)
    (root / "patches/primme/b.patch").write_text("added\n")
    added = _key(build, source)
    assert len({before, edited, added}) == 3


def test_sdist_key_follows_the_bundled_sources(build, tmp_path):
    source = _source(tmp_path, git=False)
    before = _key(build, source)
    (source / "src/main.c").write_text("int edited;\n")
    assert _key(build, source) != before
