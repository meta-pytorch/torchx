# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import shutil
import unittest
from pathlib import Path
from typing import Any, Mapping

from typing_extensions import override

from torchx.specs import AppDef, CfgVal, Role, Workspace, runopts
from torchx.testing.fixtures import TestWithTmpDir
from torchx.workspace.api import (
    MultiWorkspaceMixin,
    WorkspaceMixin,
    pin_workspace_images,
)

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


class PinWorkspaceImagesTest(unittest.TestCase):
    @staticmethod
    def _role(name: str, image: str, workspace: Workspace | None) -> Role:
        return Role(name=name, image=image, workspace=workspace)

    def test_pins_and_clears_workspace(self) -> None:
        ws = Workspace.from_str("//ws")
        app = AppDef("app", roles=[self._role("a", "unbuilt", ws)])

        pin_workspace_images(app, {("unbuilt", ws): "built:abc"})

        self.assertEqual("built:abc", app.roles[0].image)
        self.assertIsNone(app.roles[0].workspace)

    def test_leaves_prebuilt_and_unknown_workspaces_alone(self) -> None:
        known, unknown = Workspace.from_str("//known"), Workspace.from_str("//unknown")
        app = AppDef(
            "app",
            roles=[
                self._role("prebuilt", "docker.io/img:1", None),
                self._role("unknown", "unbuilt", unknown),
            ],
        )

        pin_workspace_images(app, {("unbuilt", known): "built:abc"})

        self.assertEqual("docker.io/img:1", app.roles[0].image)
        self.assertEqual("unbuilt", app.roles[1].image)
        self.assertEqual(unknown, app.roles[1].workspace)

    def test_leaves_the_same_workspace_on_another_base_image_alone(self) -> None:
        ws = Workspace.from_str("//ws")
        app = AppDef("app", roles=[self._role("a", "other-base", ws)])

        pin_workspace_images(app, {("base", ws): "built:abc"})

        self.assertEqual("other-base", app.roles[0].image)
        self.assertEqual(ws, app.roles[0].workspace)


class RecordingBuilder(WorkspaceMixin[None]):
    """Prefixes ``role.image`` with its tag, so a test can tell which builder ran."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @override
    def workspace_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            f"{self.tag}_opt", type_=str, help=f"read only by the {self.tag} builder"
        )
        return opts

    @override
    def caching_build_workspace_and_update_role(
        self,
        role: Role,
        cfg: Mapping[str, CfgVal],
        build_cache: dict[object, object],
    ) -> None:
        role.image = f"{self.tag}:{role.image}"


class PushingBuilder(RecordingBuilder):
    """A builder with a push step; records what it was asked to push."""

    def __init__(self, tag: str) -> None:
        super().__init__(tag)
        self.pushed: list[list[str]] = []

    @override
    def dryrun_push_images(self, app: AppDef, cfg: Mapping[str, CfgVal]) -> list[str]:
        return [role.image for role in app.roles]

    @override
    def push_images(self, images_to_push: list[str]) -> None:
        self.pushed.append(images_to_push)


class TwoBuilders(MultiWorkspaceMixin):
    def __init__(self) -> None:
        self.local = RecordingBuilder("local")
        self.remote = PushingBuilder("remote")

    @override
    def workspace_builders(self) -> Mapping[str, WorkspaceMixin[Any]]:
        return {"local": self.local, "remote": self.remote}


class MultiWorkspaceMixinTest(unittest.TestCase):
    def test_default_is_the_first_builder(self) -> None:
        mixin = TwoBuilders()
        name, builder = mixin.workspace_builder({})
        self.assertEqual(name, "local")
        self.assertIs(builder, mixin.local)

    def test_run_option_selects_a_builder(self) -> None:
        mixin = TwoBuilders()
        name, builder = mixin.workspace_builder({"workspace_type": "remote"})
        self.assertEqual(name, "remote")
        self.assertIs(builder, mixin.remote)

    def test_unknown_type_names_the_offered_builders(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "unknown workspace_type `oci`; this scheduler offers: local, remote",
        ):
            TwoBuilders().workspace_builder({"workspace_type": "oci"})

    def test_opts_are_the_selector_plus_every_builders_options(self) -> None:
        opts = TwoBuilders().workspace_opts()
        self.assertEqual(
            sorted(opts._opts), ["local_opt", "remote_opt", "workspace_type"]
        )
        selector = opts.get("workspace_type")
        self.assertIsNotNone(selector)
        assert selector is not None  # for the type checker
        self.assertEqual(selector.default, "local")

    def test_build_workspaces_uses_the_selected_builder(self) -> None:
        mixin = TwoBuilders()
        roles = [
            Role(name="a", image="base", workspace=Workspace({"/ws": ""})),
            Role(name="b", image="prebuilt"),
        ]
        mixin.build_workspaces(roles, {"workspace_type": "remote"})
        self.assertEqual(roles[0].image, "remote:base")
        # a role without a workspace is left alone by every builder
        self.assertEqual(roles[1].image, "prebuilt")

    def test_push_goes_to_the_builder_that_built(self) -> None:
        mixin = TwoBuilders()
        app = AppDef(name="app", roles=[Role(name="a", image="img")])
        images = mixin.dryrun_push_images(app, {"workspace_type": "remote"})
        self.assertEqual(images, ("remote", ["img"]))
        mixin.push_images(images)
        self.assertEqual(mixin.remote.pushed, [["img"]])

    def test_a_builder_without_a_push_step_pushes_nothing(self) -> None:
        mixin = TwoBuilders()
        app = AppDef(name="app", roles=[Role(name="a", image="img")])
        images = mixin.dryrun_push_images(app, {})
        self.assertEqual(images, ("local", None))
        mixin.push_images(images)  # no NotImplementedError from the base class

    def test_no_builders_is_an_error(self) -> None:
        class NoBuilders(MultiWorkspaceMixin):
            @override
            def workspace_builders(self) -> Mapping[str, WorkspaceMixin[Any]]:
                return {}

        with self.assertRaisesRegex(ValueError, "returned no builders"):
            NoBuilders().workspace_opts()
