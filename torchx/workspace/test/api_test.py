# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import shutil
from pathlib import Path
from typing import Mapping

from torchx.specs import CfgVal, Role, Workspace
from torchx.test.fixtures import TestWithTmpDir
from torchx.workspace.api import WorkspaceMixin
from typing_extensions import override

IGNORED = "__IGNORED__"


class NonCachingWorkspace(WorkspaceMixin[None]):
    def __init__(self, tmpdir: Path) -> None:
        self.tmpdir = tmpdir
        self.version_counter: dict[str, int] = {}

    def _build_new_workspace_image(self, role: Role) -> str:
        version = self.version_counter.setdefault(role.image, 0)
        ephemeral_image = f"{role.image}:{version}"
        self.version_counter[role.image] += 1

        return ephemeral_image

    @override
    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        role.image = self._build_new_workspace_image(role)
        # copy the given workspace dir for assertions
        shutil.copytree(workspace, self.tmpdir / role.image)


class NonCachingWorkspaceTest(TestWithTmpDir):
    """Tests workspaces with `build_workspace_and_update_role` implemented"""

    def test_build_workspaces(self) -> None:
        workspace_dir = self.create_dir_tree(
            "workspace",
            {
                "proj_a": {
                    "a.py": "project a",
                },
                "proj_b": {
                    "b.py": "project b",
                },
            },
        )
        workspace = Workspace(
            projects={
                str(workspace_dir / "proj_a"): "",
                str(workspace_dir / "proj_b"): "b",
            }
        )
        roles = [
            Role(name=IGNORED, image="foo", workspace=None),
            Role(name=IGNORED, image="bar", workspace=workspace),
            Role(name=IGNORED, image="bar", workspace=workspace),
            Role(name=IGNORED, image="baz", workspace=workspace),
        ]

        outdir = self.tmpdir / "out"
        NonCachingWorkspace(outdir).build_workspaces(roles, cfg={})

        # check the updated images for each role
        self.assertListEqual(
            [
                Role(name=IGNORED, image="foo", workspace=None),
                Role(name=IGNORED, image="bar:0", workspace=workspace),
                Role(name=IGNORED, image="bar:1", workspace=workspace),
                Role(name=IGNORED, image="baz:0", workspace=workspace),
            ],
            roles,
        )

        merged_workspace = {
            "a.py": "project a",
            "b": {
                "b.py": "project b",
            },
        }
        self.assertDirTree(
            outdir,
            {
                "bar:0": merged_workspace,
                "bar:1": merged_workspace,
                "baz:0": merged_workspace,
            },
        )


class CachingWorkspace(WorkspaceMixin[None]):
    def __init__(self, tmpdir: Path) -> None:
        self.tmpdir = tmpdir
        self.version_counter: dict[str, int] = {}

    def _build_new_workspace_image(self, role: Role) -> str:
        version = self.version_counter.setdefault(role.image, 0)
        ephemeral_image = f"{role.image}:{version}"
        self.version_counter[role.image] += 1

        workspace = role.workspace
        assert workspace is not None

        workspace.merge_into(self.tmpdir / ephemeral_image)
        return ephemeral_image

    @override
    def caching_build_workspace_and_update_role(
        self,
        role: Role,
        cfg: Mapping[str, CfgVal],
        build_cache: dict[object, object],
    ) -> None:
        image = role.image
        workspace = role.workspace

        cache_key = (image, workspace)
        if (ephemeral_image := build_cache.get(cache_key)) is None:
            # cache miss, build new image
            role.image = self._build_new_workspace_image(role)
            build_cache[cache_key] = role.image
        else:
            assert isinstance(ephemeral_image, str)
            role.image = ephemeral_image


class CachingWorkspaceTest(TestWithTmpDir):
    """Tests workspaces with `caching_build_workspace_and_update_role` implemented"""

    def test_build_workspaces(self) -> None:
        workspace_dir = self.create_dir_tree(
            "workspace",
            {
                "proj_a": {
                    "a.py": "project a",
                },
                "proj_b": {
                    "b.py": "project b",
                },
                "proj_c": {
                    "c.py": "project c",
                },
            },
        )
        workspace1 = Workspace(
            projects={
                str(workspace_dir / "proj_a"): "",
                str(workspace_dir / "proj_b"): "b",
            }
        )
        workspace2 = Workspace(projects={str(workspace_dir / "proj_c"): "c"})

        roles = [
            Role(name=IGNORED, image="foo", workspace=None),
            Role(name=IGNORED, image="bar", workspace=workspace1),
            Role(name=IGNORED, image="bar", workspace=workspace1),  # cache hit
            Role(name=IGNORED, image="baz", workspace=workspace1),
            Role(name=IGNORED, image="baz", workspace=workspace2),  # cache miss
        ]

        outdir = self.tmpdir / "out"
        CachingWorkspace(outdir).build_workspaces(roles, cfg={})

        # check the updated images for each role
        self.assertListEqual(
            [
                Role(name=IGNORED, image="foo", workspace=None),
                Role(name=IGNORED, image="bar:0", workspace=workspace1),
                Role(name=IGNORED, image="bar:0", workspace=workspace1),  # cache hit
                Role(name=IGNORED, image="baz:0", workspace=workspace1),
                Role(name=IGNORED, image="baz:1", workspace=workspace2),  # cache miss
            ],
            roles,
        )

        merged_workspace1 = {
            "a.py": "project a",
            "b": {
                "b.py": "project b",
            },
        }
        merged_workspace2 = {
            "c": {
                "c.py": "project c",
            },
        }
        self.assertDirTree(
            outdir,
            {
                "bar:0": merged_workspace1,
                "baz:0": merged_workspace1,
                "baz:1": merged_workspace2,
            },
        )
