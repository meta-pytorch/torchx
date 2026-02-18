#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from unittest.mock import patch

from hydra.core.global_hydra import GlobalHydra
from torchx.components.utils import hydra
from torchx.specs import AppDef


class HydraComponentTest(unittest.TestCase):
    def tearDown(self) -> None:
        GlobalHydra.instance().clear()

    def test_hydra_simple(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_config.yaml"), "w") as f:
                f.write(
                    """
app:
  _target_: torchx.specs.AppDef
  name: test_job
  roles:
    - _target_: torchx.specs.Role
      name: test
      image: alpine:latest
      entrypoint: echo
      num_replicas: 1
      args:
        - hello
"""
                )

            app = hydra(config_name="test_config", config_dir=tmpdir)

            self.assertIsInstance(app, AppDef)
            self.assertEqual(app.name, "test_job")
            self.assertEqual(len(app.roles), 1)
            self.assertEqual(app.roles[0].name, "test")
            self.assertEqual(app.roles[0].image, "alpine:latest")
            self.assertEqual(app.roles[0].args, ["hello"])

    def test_hydra_with_resource(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_resource.yaml"), "w") as f:
                f.write(
                    """
app:
  _target_: torchx.specs.AppDef
  name: training_job
  roles:
    - _target_: torchx.specs.Role
      name: trainer
      image: alpine:latest
      entrypoint: python
      num_replicas: 2
      resource:
        _target_: torchx.specs.Resource
        cpu: 4
        gpu: 2
        memMB: 8192
"""
                )

            app = hydra(config_name="test_resource", config_dir=tmpdir)

            role = app.roles[0]
            self.assertEqual(role.num_replicas, 2)
            self.assertEqual(role.resource.cpu, 4)
            self.assertEqual(role.resource.gpu, 2)
            self.assertEqual(role.resource.memMB, 8192)

    def test_hydra_multiple_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_multi.yaml"), "w") as f:
                f.write(
                    """
app:
  _target_: torchx.specs.AppDef
  name: multi_role_job
  roles:
    - _target_: torchx.specs.Role
      name: trainer
      image: alpine:latest
      entrypoint: python
      num_replicas: 2
    - _target_: torchx.specs.Role
      name: worker
      image: alpine:latest
      entrypoint: python
      num_replicas: 4
"""
                )

            app = hydra(config_name="test_multi", config_dir=tmpdir)

            self.assertEqual(len(app.roles), 2)
            self.assertEqual(app.roles[0].name, "trainer")
            self.assertEqual(app.roles[0].num_replicas, 2)
            self.assertEqual(app.roles[1].name, "worker")
            self.assertEqual(app.roles[1].num_replicas, 4)

    @patch.dict(os.environ, {"TEST_IMAGE": "test:v1"})
    def test_hydra_with_env_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_env.yaml"), "w") as f:
                f.write(
                    """
app:
  _target_: torchx.specs.AppDef
  name: env_test
  roles:
    - _target_: torchx.specs.Role
      name: test
      image: ${oc.env:TEST_IMAGE}
      entrypoint: echo
      num_replicas: 1
"""
                )

            app = hydra(config_name="test_env", config_dir=tmpdir)
            self.assertEqual(app.roles[0].image, "test:v1")

    def test_hydra_with_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "role"))
            with open(os.path.join(tmpdir, "role", "python.yaml"), "w") as f:
                f.write(
                    """
_target_: torchx.specs.Role
name: python
image: alpine:latest
entrypoint: python
num_replicas: 1
"""
                )
            with open(os.path.join(tmpdir, "test_override.yaml"), "w") as f:
                f.write(
                    """
defaults:
  - role: python

app:
  _target_: torchx.specs.AppDef
  name: override_test
  roles:
    - ${role}
"""
                )

            app = hydra(
                "role.num_replicas=3", config_name="test_override", config_dir=tmpdir
            )
            self.assertEqual(app.roles[0].num_replicas, 3)

    def test_hydra_with_torchx_resolvers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_resolvers.yaml"), "w") as f:
                f.write(
                    """
app:
  _target_: torchx.specs.AppDef
  name: test_resolvers
  roles:
    - _target_: torchx.specs.Role
      name: test
      image: alpine:latest
      entrypoint: sh
      num_replicas: 1
      args:
        - -c
        - echo done
      env:
        APP_ID: ${torchx.app_id:}
        RANK0: ${torchx.rank0_env:}
        REPLICA: ${torchx.replica_id:}
        IMG_ROOT: ${torchx.img_root:}
"""
                )

            app = hydra(config_name="test_resolvers", config_dir=tmpdir)

            self.assertEqual(app.roles[0].env["APP_ID"], "${app_id}")
            self.assertEqual(app.roles[0].env["RANK0"], "${rank0_env}")
            self.assertEqual(app.roles[0].env["REPLICA"], "${replica_id}")
            self.assertEqual(app.roles[0].env["IMG_ROOT"], "${img_root}")

    def test_hydra_torchx_macros_are_replaceable(self) -> None:
        from torchx.specs.api import macros

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test_macros.yaml"), "w") as f:
                f.write(
                    """
app:
  _target_: torchx.specs.AppDef
  name: test
  roles:
    - _target_: torchx.specs.Role
      name: test
      image: alpine:latest
      entrypoint: echo
      num_replicas: 1
      env:
        APP_ID: ${torchx.app_id:}
        RANK0: ${torchx.rank0_env:}
"""
                )

            app = hydra(config_name="test_macros", config_dir=tmpdir)

            # Verify macros are in the env
            self.assertEqual(app.roles[0].env["APP_ID"], "${app_id}")
            self.assertEqual(app.roles[0].env["RANK0"], "${rank0_env}")

            # Verify TorchX can replace them using its machinery
            values = macros.Values(
                img_root="/tmp/img",
                app_id="test-job-123",
                replica_id="0",
                rank0_env="localhost",
            )
            replaced_role = values.apply(app.roles[0])

            self.assertEqual(replaced_role.env["APP_ID"], "test-job-123")
            self.assertEqual(replaced_role.env["RANK0"], "localhost")
