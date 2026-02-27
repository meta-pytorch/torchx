#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import posixpath
import shutil
from tempfile import mkdtemp
from typing import Mapping

import fsspec
from torchx.specs import CfgVal, Role
from torchx.workspace.api import walk_workspace, WorkspaceMixin


class TmpDirWorkspaceMixin(WorkspaceMixin[None]):
    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """Copies *workspace* to a temp directory and sets ``role.image`` to it.

        Files matching ``.torchxignore`` patterns are skipped.
        """
        job_dir = mkdtemp(prefix="torchx_workspace")
        _copy_to_dir(workspace, job_dir)
        role.image = job_dir


class DirWorkspaceMixin(WorkspaceMixin[None]):
    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """Copies *workspace* into ``cfg["job_dir"]`` and sets ``role.image`` to it.

        No-op if ``job_dir`` is not set. Files matching ``.torchxignore``
        patterns are skipped.
        """
        job_dir = cfg.get("job_dir")
        if job_dir is None:
            return
        assert isinstance(job_dir, str), "job_dir must be str"

        os.mkdir(job_dir)
        _copy_to_dir(workspace, job_dir)
        role.image = job_dir


def _copy_to_dir(workspace: str, target: str) -> None:
    fs, path = fsspec.core.url_to_fs(workspace)
    assert isinstance(path, str), "path must be str"

    for dir, dirs, files in walk_workspace(fs, path):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        for file, info in files.items():
            filepath = posixpath.join(
                target,
                posixpath.join(relpath, file) if relpath != "." else file,
            )
            with fs.open(info["name"], "rb") as src, fsspec.open(filepath, "wb") as dst:
                shutil.copyfileobj(src, dst)
