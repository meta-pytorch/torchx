# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import shutil

from pathlib import Path
from typing import Mapping

from torchx.specs import CfgVal, Role
from torchx.test.fixtures import TestWithTmpDir

from torchx.workspace.api import Workspace, WorkspaceMixin


class TestWorkspace(WorkspaceMixin[None]):
    def __init__(self, tmpdir: Path) -> None:
        self.tmpdir = tmpdir

    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        role.image = "bar"
        role.metadata["workspace"] = workspace

        if not workspace.startswith("//"):
            # to validate the merged workspace dir copy its content to the tmpdir
            shutil.copytree(workspace, self.tmpdir)


class WorkspaceTest(TestWithTmpDir):

    def test_to_string_single_project_workspace(self) -> None:
        self.assertEqual(
            "/home/foo/bar",
            str(Workspace(projects={"/home/foo/bar": ""})),
        )

    def test_to_string_multi_project_workspace(self) -> None:
        workspace = Workspace(
            projects={
                "/home/foo/workspace/myproj": "",
                "/home/foo/github/torch": "torch",
            }
        )

        self.assertEqual(
            "/home/foo/workspace/myproj;/home/foo/github/torch:torch",
            str(workspace),
        )

    def test_is_unmapped_single_project_workspace(self) -> None:
        self.assertTrue(
            Workspace(projects={"/home/foo/bar": ""}).is_unmapped_single_project()
        )

        self.assertFalse(
            Workspace(projects={"/home/foo/bar": "baz"}).is_unmapped_single_project()
        )

        self.assertFalse(
            Workspace(
                projects={"/home/foo/bar": "", "/home/foo/torch": ""}
            ).is_unmapped_single_project()
        )

        self.assertFalse(
            Workspace(
                projects={"/home/foo/bar": "", "/home/foo/torch": "pytorch"}
            ).is_unmapped_single_project()
        )

    def test_from_str_single_project(self) -> None:
        self.assertDictEqual(
            {"/home/foo/bar": ""},
            Workspace.from_str("/home/foo/bar").projects,
        )

        self.assertDictEqual(
            {"/home/foo/bar": "baz"},
            Workspace.from_str("/home/foo/bar: baz").projects,
        )

    def test_from_str_multi_project(self) -> None:
        self.assertDictEqual(
            {
                "/home/foo/bar": "",
                "/home/foo/third-party/verl": "verl",
            },
            Workspace.from_str(
                """#
/home/foo/bar:
/home/foo/third-party/verl: verl
"""
            ).projects,
        )

    def test_build_and_update_role2_str_workspace(self) -> None:
        proj = self.tmpdir / "github" / "torch"
        proj.mkdir(parents=True)
        (proj / "torch.py").touch()

        role = Role(name="__IGNORED__", image="foo")
        out = self.tmpdir / "workspace-merged"
        TestWorkspace(out).build_workspace_and_update_role2(
            role,
            str(proj),
            cfg={},
        )

        # make sure build_workspace_and_update_role has been called
        # by checking that the image is updated from "foo" to "bar"
        self.assertEqual(role.image, "bar")
        self.assertTrue((out / "torch.py").exists())

    def test_build_and_update_role2_unmapped_single_project_workspace(self) -> None:
        proj = self.tmpdir / "github" / "torch"
        proj.mkdir(parents=True)
        (proj / "torch.py").touch()

        role = Role(name="__IGNORED__", image="foo")
        out = self.tmpdir / "workspace-merged"
        TestWorkspace(out).build_workspace_and_update_role2(
            role,
            Workspace(projects={str(proj): ""}),
            cfg={},
        )

        self.assertEqual(role.image, "bar")
        self.assertTrue((out / "torch.py").exists())

    def test_build_and_update_role2_unmapped_single_project_workspace_buck(
        self,
    ) -> None:
        buck_target = "//foo/bar:main"

        role = Role(name="__IGNORED__", image="foo")
        out = self.tmpdir / "workspace-merged"
        TestWorkspace(out).build_workspace_and_update_role2(
            role,
            Workspace(projects={buck_target: ""}),
            cfg={},
        )
        self.assertEqual(role.image, "bar")
        self.assertEqual(role.metadata["workspace"], buck_target)

    def test_build_and_update_role2_multi_project_workspace(self) -> None:
        proj1 = self.tmpdir / "github" / "torch"
        proj1.mkdir(parents=True)
        (proj1 / "torch.py").touch()

        proj2 = self.tmpdir / "github" / "verl"
        proj2.mkdir(parents=True)
        (proj2 / "verl.py").touch()

        file1 = self.tmpdir / ".torchxconfig"
        file1.touch()

        role = Role(name="__IGNORED__", image="foo")
        workspace = Workspace(
            projects={
                str(proj1): "",
                str(proj2): "verl",
                str(file1): "verl/.torchxconfig",
            }
        )

        out = self.tmpdir / "workspace-merged"
        TestWorkspace(out).build_workspace_and_update_role2(
            role,
            workspace,
            cfg={},
        )

        self.assertEqual(role.image, "bar")
        self.assertTrue((out / "torch.py").exists())
        self.assertTrue((out / "verl" / "verl.py").exists())
        self.assertTrue((out / "verl" / ".torchxconfig").exists())
