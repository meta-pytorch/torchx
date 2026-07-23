#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os.path
import shutil
import tempfile
import unittest

import fsspec
from torchx.specs import Role
from torchx.workspace.dir_workspace import (
    _copy_to_dir,
    DirWorkspaceMixin,
    TmpDirWorkspaceMixin,
)


class DirWorkspaceTest(unittest.TestCase):
    def test_build_workspace_no_job_dir(self) -> None:
        w = DirWorkspaceMixin()
        role = Role(
            name="role",
            image="blah",
        )
        # should be noop
        w.build_workspace_and_update_role(role, workspace="invalid", cfg={})
        self.assertEqual(role.image, "blah")

    def test_build_workspace(self) -> None:
        w = DirWorkspaceMixin()
        role = Role(
            name="role",
            image="blah",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = os.path.join(tmpdir, "job")
            w.build_workspace_and_update_role(
                role,
                workspace="invalid",
                cfg={
                    "job_dir": job_dir,
                },
            )

    def test_torchxignore(self) -> None:
        fs = fsspec.filesystem("memory")
        files = [
            "ignoredir/bar",
            "ignoredir/recursive/bar",
            "three/part/ignore/foo/bar",
            "dir1/bar",
            "dir/ignorefileglob1",
            "dir/recursive/ignorefileglob2",
            "dir/ignorefile",
            "ignorefile",
            "ignorefilesuffix",
            "dir/file",
            "foo.sh",
            "unignore",
        ]
        for file in files:
            fs.touch("torchxignore/" + file)
        with fs.open("torchxignore/.torchxignore", "wt") as f:
            f.write(
                """
                # comment

                # dirs/files
                ignoredir
                ignorefile
                three/part/ignore

                # globs
                */ignorefileglo*1
                **/ignorefileglob2
                dir?

                # inverse patterns
                unignore
                !unignore

                # ignore .
                .
            """
            )

        _copy_to_dir("memory://torchxignore", "memory://torchxignoredest")

        files = fs.glob("torchxignoredest/*") + fs.glob("torchxignoredest/**/*")
        # strip prefix
        files = {
            os.path.normpath(file.partition("torchxignoredest/")[2]) for file in files
        }
        self.assertSetEqual(
            files,
            {
                ".torchxignore",
                "dir",
                "dir/file",
                "dir/ignorefile",
                "foo.sh",
                "ignorefilesuffix",
                "unignore",
            },
        )


class TmpDirWorkspaceTest(unittest.TestCase):
    def test_build_workspace(self) -> None:
        w = TmpDirWorkspaceMixin()
        role = Role(
            name="role",
            image="blah",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            w.build_workspace_and_update_role(
                role,
                workspace=f"file://{tmpdir}",
                cfg={},
            )
        self.assertIn(tempfile.gettempdir(), role.image)
        shutil.rmtree(role.image)
