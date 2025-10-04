#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import posixpath
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import fsspec
import torchx
from docker.types import DeviceRequest, Mount
from torchx import specs
from torchx.components.dist import ddp
from torchx.schedulers.api import ListAppResponse, Stream
from torchx.schedulers.docker_scheduler import (
    create_scheduler,
    DockerContainer,
    DockerJob,
    DockerOpts,
    DockerScheduler,
    has_docker,
)
from torchx.schedulers.test.local_scheduler_test import LocalSchedulerTestUtil
from torchx.specs.api import AppDef, AppState, Role


def _test_app() -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer",
        image="pytorch/torchx:latest",
        entrypoint="main",
        args=[
            "--output-path",
            specs.macros.img_root,
            "--app-id",
            specs.macros.app_id,
            "--rank0-env",
            specs.macros.rank0_env,
        ],
        env={"FOO": "bar"},
        resource=specs.Resource(
            cpu=2,
            memMB=3000,
            gpu=4,
        ),
        port_map={"foo": 1234},
        num_replicas=1,
        max_retries=3,
        mounts=[
            specs.BindMount(src_path="/tmp", dst_path="/tmp", read_only=True),
            specs.DeviceMount(src_path="/dev/null", dst_path="/dev/null"),
        ],
    )

    return specs.AppDef("test", roles=[trainer_role])


class DockerSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler: DockerScheduler = create_scheduler(
            session_name="test_session",
        )

    def test_submit_dryrun(self) -> None:
        app = _test_app()
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app_name_42"
            info = self.scheduler.submit_dryrun(app, cfg={})

        want = DockerJob(
            "app_name_42",
            [
                DockerContainer(
                    image="pytorch/torchx:latest",
                    command=[
                        "main",
                        "--output-path",
                        "",
                        "--app-id",
                        "app_name_42",
                        "--rank0-env",
                        "TORCHX_RANK0_HOST",
                    ],
                    kwargs={
                        "device_requests": [
                            DeviceRequest(
                                count=4,
                                capabilities=[["compute", "utility"]],
                            )
                        ],
                        "devices": [
                            "/dev/null:/dev/null:rwm",
                        ],
                        "environment": {
                            "FOO": "bar",
                            "TORCHX_RANK0_HOST": "app_name_42-trainer-0",
                            "TORCHX_IMAGE": "pytorch/torchx:latest",
                        },
                        "labels": {
                            "torchx.pytorch.org/app-id": "app_name_42",
                            "torchx.pytorch.org/replica-id": "0",
                            "torchx.pytorch.org/role-name": "trainer",
                            "torchx.pytorch.org/version": torchx.__version__,
                        },
                        "mem_limit": "3000m",
                        "shm_size": "3000m",
                        "privileged": False,
                        "name": "app_name_42-trainer-0",
                        "hostname": "app_name_42-trainer-0",
                        "nano_cpus": int(2e9),
                        "restart_policy": {
                            "Name": "on-failure",
                            "MaximumRetryCount": 3,
                        },
                        "network": "torchx",
                        "mounts": [
                            Mount(
                                target="/tmp",
                                source="/tmp",
                                read_only=True,
                                type="bind",
                            ),
                        ],
                    },
                )
            ],
        )
        self.assertEqual(str(info), str(want))

    def test_volume_mounts(self) -> None:
        app = _test_app()
        app.roles[0].mounts = [
            specs.VolumeMount(src="name", dst_path="/tmp", read_only=True),
        ]

        info = self.scheduler.submit_dryrun(app, cfg={})
        want = [
            Mount(
                target="/tmp",
                source="name",
                read_only=True,
                type="volume",
            ),
        ]
        self.assertEqual(info.request.containers[0].kwargs["mounts"], want)

    def test_device_mounts(self) -> None:
        app = _test_app()
        app.roles[0].mounts = [
            specs.DeviceMount(src_path="foo", dst_path="bar"),
        ]

        info = self.scheduler.submit_dryrun(app, cfg={})
        self.assertEqual(info.request.containers[0].kwargs["devices"], ["foo:bar:rwm"])

    def test_resource_devices(self) -> None:
        app = _test_app()
        app.roles[0].mounts = []
        app.roles[0].resource.devices = {
            "vpc.amazonaws.com/efa": 1,
            "aws.amazon.com/neurondevice": 2,
        }

        info = self.scheduler.submit_dryrun(app, cfg={})
        self.assertEqual(
            info.request.containers[0].kwargs["devices"],
            [
                "/dev/infiniband/uverbs0:/dev/infiniband/uverbs0:rwm",
                "/dev/neuron0:/dev/neuron0:rwm",
                "/dev/neuron1:/dev/neuron1:rwm",
            ],
        )

    @patch("os.environ", {"FOO_1": "f1", "BAR_1": "b1", "FOOBAR_1": "fb1"})
    def test_copy_env(self) -> None:
        app = _test_app()
        cfg = DockerOpts({"copy_env": ["FOO_*", "BAR_*"]})
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app_name_42"
            info = self.scheduler.submit_dryrun(app, cfg)
        self.assertEqual(
            info.request.containers[0].kwargs["environment"],
            {
                "FOO": "bar",
                "FOO_1": "f1",
                "BAR_1": "b1",
                "TORCHX_RANK0_HOST": "app_name_42-trainer-0",
                "TORCHX_IMAGE": "pytorch/torchx:latest",
            },
        )

    def test_env(self) -> None:
        app = _test_app()
        cfg = DockerOpts({"env": {"FOO_1": "BAR_1"}})
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app_name_42"
            info = self.scheduler.submit_dryrun(app, cfg)
        self.assertEqual(
            info.request.containers[0].kwargs["environment"],
            {
                "FOO": "bar",
                "FOO_1": "BAR_1",
                "TORCHX_RANK0_HOST": "app_name_42-trainer-0",
                "TORCHX_IMAGE": "pytorch/torchx:latest",
            },
        )

    def test_privileged(self) -> None:
        app = _test_app()
        cfg = DockerOpts({"privileged": True})
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app_name_42"
            info = self.scheduler.submit_dryrun(app, cfg)
        self.assertTrue(info.request.containers[0].kwargs["privileged"])

    def test_long_hostname(self) -> None:
        app = _test_app()
        for role in app.roles:
            role.name = "ethology_explore_magic_calliope_divisive_whirl_dealt_lotus_oncology_facet_deerskin_blum_elective_spill_trammel_trainer"
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "ethology_explore_magic_calliope_divisive_whirl_dealt_lotus_oncology_facet_deerskin__.-_elective_spill_trammel_1234"
            info = self.scheduler.submit_dryrun(app, DockerOpts())
        for container in info.request.containers:
            assert "name" in container.kwargs
            name = container.kwargs["name"]
            assert isinstance(name, str)
            assert len(name) < 65
            # Assert match container name rules https://github.com/moby/moby/blob/master/daemon/names/names.go#L6
            self.assertRegex(name, r"[a-zA-Z0-9][a-zA-Z0-9_.-]")


if has_docker():
    # These are the live tests that require a local docker instance.

    class DockerSchedulerLiveTest(unittest.TestCase, LocalSchedulerTestUtil):
        def setUp(self) -> None:
            self.scheduler: DockerScheduler = create_scheduler(
                session_name="test_session",
            )

        def _docker_app(self, entrypoint: str, *args: str) -> AppDef:
            return AppDef(
                name="test-app",
                roles=[
                    Role(
                        name="image_test_role",
                        image="busybox",
                        entrypoint=entrypoint,
                        args=list(args),
                    ),
                ],
            )

        def test_docker_submit(self) -> None:
            app = self._docker_app("echo", "foo")
            app_id = self.scheduler.submit(app, cfg={})

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.SUCCEEDED, desc.state)
            self.assertEqual(len(desc.roles), 1)
            self.assertEqual(len(desc.roles_statuses), 1)
            self.assertEqual(len(desc.roles_statuses[0].replicas), 1)
            self.assertEqual(
                desc.roles_statuses[0].replicas[0].state, AppState.SUCCEEDED
            )

            self.assertEqual(desc.app_id, app_id)

        def test_docker_logs(self) -> None:
            app = self._docker_app("echo", "foo\nbar")
            start = datetime.utcnow()
            app_id = self.scheduler.submit(app, cfg={})
            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            # docker truncates to the second so pad out 1 extra second
            end = datetime.utcnow() + timedelta(seconds=1)

            self.assertEqual(AppState.SUCCEEDED, desc.state)

            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    since=start,
                    until=end,
                )
            )
            self.assertEqual(
                logs,
                [
                    "foo\n",
                    "bar\n",
                ],
            )
            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    regex="bar",
                )
            )
            self.assertEqual(
                logs,
                [
                    "bar\n",
                ],
            )

            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    since=end,
                )
            )
            self.assertEqual(logs, [])
            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    until=start,
                )
            )
            self.assertEqual(logs, [])
            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    should_tail=True,
                )
            )
            self.assertEqual(
                logs,
                [
                    "foo\n",
                    "bar\n",
                ],
            )

        def test_docker_logs_streams(self) -> None:
            app = self._docker_app("sh", "-c", "echo stdout; >&2 echo stderr")

            start = datetime.utcnow()
            app_id = self.scheduler.submit(app, cfg={})
            desc = self.wait(app_id)
            self.assertIsNotNone(desc)

            logs = set(
                self.scheduler.log_iter(app_id, "image_test_role", 0, streams=None)
            )
            self.assertEqual(
                logs,
                {
                    "stdout\n",
                    "stderr\n",
                },
            )

            logs = set(
                self.scheduler.log_iter(
                    app_id, "image_test_role", 0, streams=Stream.COMBINED
                )
            )
            self.assertEqual(
                logs,
                {
                    "stdout\n",
                    "stderr\n",
                },
            )

            logs = list(
                self.scheduler.log_iter(
                    app_id, "image_test_role", 0, streams=Stream.STDERR
                )
            )
            self.assertEqual(
                logs,
                [
                    "stderr\n",
                ],
            )

            logs = list(
                self.scheduler.log_iter(
                    app_id, "image_test_role", 0, streams=Stream.STDOUT
                )
            )
            self.assertEqual(
                logs,
                [
                    "stdout\n",
                ],
            )

        def test_docker_list(self) -> None:
            app = self._docker_app("echo", "bar")
            app_id = self.scheduler.submit(app, cfg={})

            self.wait(app_id)
            self.assertTrue(
                ListAppResponse(app_id=app_id, state=AppState.SUCCEEDED)
                in self.scheduler.list()
            )

        def test_docker_cancel(self) -> None:
            app = self._docker_app("sleep", "10000")
            app_id = self.scheduler.submit(app, cfg={})
            _ = self.scheduler.describe(app_id)

            self.wait(app_id, wait_for=lambda state: state == AppState.RUNNING)
            self.scheduler.cancel(app_id)

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(desc.state, AppState.FAILED)

        def test_docker_submit_error(self) -> None:
            app = self._docker_app("sh", "-c", "exit 1")
            app_id = self.scheduler.submit(app, cfg={})

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.FAILED, desc.state)
            self.assertEqual(len(desc.roles), 1)
            self.assertEqual(len(desc.roles_statuses), 1)
            self.assertEqual(len(desc.roles_statuses[0].replicas), 1)
            self.assertEqual(desc.roles_statuses[0].replicas[0].state, AppState.FAILED)

        def test_docker_submit_error_retries(self) -> None:
            app = self._docker_app("sh", "-c", "exit 1")
            app.roles[0].max_retries = 1
            app_id = self.scheduler.submit(app, cfg={})

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.FAILED, desc.state)

        def test_docker_submit_dist(self) -> None:
            workspace = "memory://docker_submit_dist/"
            with fsspec.open(posixpath.join(workspace, "main.py"), "wt") as f:
                f.write("print('hello world')\n")
            app = ddp(script="main.py", j="2x1")
            app_id = self.scheduler.submit(app, cfg={}, workspace=workspace)
            print(app_id)

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.SUCCEEDED, desc.state)
            self.assertEqual(len(desc.roles), 1)
            self.assertEqual(len(desc.roles_statuses), 1)
            self.assertEqual(len(desc.roles_statuses[0].replicas), 2)
            self.assertEqual(
                desc.roles_statuses[0].replicas[0].state, AppState.SUCCEEDED
            )
            self.assertEqual(
                desc.roles_statuses[0].replicas[1].state, AppState.SUCCEEDED
            )
