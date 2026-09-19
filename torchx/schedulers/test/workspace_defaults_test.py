# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx.schedulers import get_scheduler_factories
from torchx.workspace.api import MultiWorkspaceMixin, WorkspaceMixin
from torchx.workspace.dir_workspace import DirWorkspaceMixin
from torchx.workspace.docker_workspace import DockerWorkspaceMixin


class WorkspaceDefaultsTest(unittest.TestCase):
    """Selectable builders left every in-tree scheduler's default behaviour alone."""

    def test_local_cwd_builds_no_workspace(self) -> None:
        scheduler = get_scheduler_factories()["local_cwd"]("test")
        self.assertNotIsInstance(scheduler, WorkspaceMixin)

    def test_local_docker_and_kubernetes_keep_the_docker_builder(self) -> None:
        for name in ("local_docker", "kubernetes"):
            with self.subTest(scheduler=name):
                scheduler = get_scheduler_factories()[name]("test")
                self.assertIsInstance(scheduler, DockerWorkspaceMixin)
                self.assertNotIsInstance(scheduler, MultiWorkspaceMixin)
                self.assertNotIn("workspace_type", scheduler.run_opts()._opts)

    def test_slurm_defaults_to_the_job_dir_builder(self) -> None:
        scheduler = get_scheduler_factories()["slurm"]("test")
        self.assertIsInstance(scheduler, MultiWorkspaceMixin)
        assert isinstance(scheduler, MultiWorkspaceMixin)  # for the type checker
        name, builder = scheduler.workspace_builder({})
        self.assertEqual(name, "dir")
        self.assertIsInstance(builder, DirWorkspaceMixin)
        selector = scheduler.run_opts().get("workspace_type")
        self.assertIsNotNone(selector)
        assert selector is not None  # for the type checker
        self.assertEqual(selector.default, "dir")
