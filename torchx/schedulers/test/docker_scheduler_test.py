#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import posixpath
import sys
import unittest
from datetime import datetime, timedelta
from typing import cast
from unittest.mock import call, MagicMock, patch

import fsspec
import torchx
from docker.errors import APIError, DockerException
from docker.models.containers import Container
from docker.types import DeviceRequest, Mount
from torchx import specs
from torchx.components.dist import ddp
from torchx.schedulers.api import ListAppResponse, Scheduler, Stream
from torchx.schedulers.docker_scheduler import (
    create_scheduler,
    DockerContainer,
    DockerJob,
    DockerScheduler,
    ensure_network,
    has_docker,
    LABEL_APP_ID,
    LABEL_REPLICA_ID,
    LABEL_ROLE_NAME,
    Opts,
)
from torchx.schedulers.test.local_scheduler_test import LocalSchedulerTestUtil
from torchx.specs.api import AppDef, AppDryRunInfo, AppState, Role


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


def _mock_container(
    status: str = "running",
    exit_code: int = 0,
    role: str = "trainer",
    replica_id: int = 0,
    app_id: str = "app_id_1",
) -> MagicMock:
    container = MagicMock(spec=Container)
    container.status = status
    container.wait.return_value = {"StatusCode": exit_code}
    container.labels = {
        LABEL_APP_ID: app_id,
        LABEL_ROLE_NAME: role,
        LABEL_REPLICA_ID: str(replica_id),
    }
    # `name` is reserved in the MagicMock constructor so set it afterwards
    container.name = f"{app_id}-{role}-{replica_id}"
    return container


class DockerSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler: DockerScheduler = create_scheduler(
            session_name="test_session",
        )

    def test_submit_dryrun(self) -> None:
        app = _test_app()
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app_name_42"
            info = self.scheduler.submit_dryrun(app, cfg=Opts())

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

        info = self.scheduler.submit_dryrun(app, cfg=Opts())
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

        info = self.scheduler.submit_dryrun(app, cfg=Opts())
        self.assertEqual(info.request.containers[0].kwargs["devices"], ["foo:bar:rwm"])

    def test_resource_devices(self) -> None:
        app = _test_app()
        app.roles[0].mounts = []
        app.roles[0].resource.devices = {
            "vpc.amazonaws.com/efa": 1,
            "aws.amazon.com/neurondevice": 2,
        }

        info = self.scheduler.submit_dryrun(app, cfg=Opts())
        self.assertEqual(
            info.request.containers[0].kwargs["devices"],
            [
                "/dev/infiniband/uverbs0:/dev/infiniband/uverbs0:rwm",
                "/dev/neuron0:/dev/neuron0:rwm",
                "/dev/neuron1:/dev/neuron1:rwm",
            ],
        )

    def test_resource_devices_dryrun_idempotent(self) -> None:
        app = _test_app()
        app.roles[0].mounts = []
        app.roles[0].resource.devices = {"vpc.amazonaws.com/efa": 1}
        want = ["/dev/infiniband/uverbs0:/dev/infiniband/uverbs0:rwm"]

        info_1 = self.scheduler.submit_dryrun(app, cfg=Opts())
        info_2 = self.scheduler.submit_dryrun(app, cfg=Opts())

        # a dryrun must not mutate the AppDef; the second dryrun would
        # otherwise see (and re-add) the previously appended DeviceMounts
        self.assertEqual([], app.roles[0].mounts)
        self.assertEqual(want, info_1.request.containers[0].kwargs["devices"])
        self.assertEqual(want, info_2.request.containers[0].kwargs["devices"])

    def test_describe_no_containers(self) -> None:
        client = MagicMock()
        client.containers.list.return_value = []
        with patch.object(DockerScheduler, "_docker_client", client):
            self.assertIsNone(self.scheduler.describe("does-not-exist"))

    def test_has_docker_no_docker_module(self) -> None:
        # simulate docker-py not being installed
        with patch.dict(sys.modules, {"docker": None}):
            self.assertFalse(has_docker())

    @patch("os.environ", {"FOO_1": "f1", "BAR_1": "b1", "FOOBAR_1": "fb1"})
    def test_copy_env(self) -> None:
        app = _test_app()
        cfg = Opts(copy_env=["FOO_*", "BAR_*"])
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
        cfg = Opts(env={"FOO_1": "BAR_1"})
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
        cfg = Opts(privileged=True)
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
            info = self.scheduler.submit_dryrun(app, Opts())
        for container in info.request.containers:
            assert "name" in container.kwargs
            name = container.kwargs["name"]
            assert isinstance(name, str)
            assert len(name) < 65
            # Assert match container name rules https://github.com/moby/moby/blob/master/daemon/names/names.go#L6
            self.assertRegex(name, r"[a-zA-Z0-9][a-zA-Z0-9_.-]")

    def test_submit_dryrun_unknown_mount_type_raises(self) -> None:
        app = _test_app()
        app.roles[0].mounts = [cast(specs.BindMount, "not-a-mount")]

        with self.assertRaisesRegex(
            TypeError,
            "unknown mount type",
            msg="a mount that is not Bind/Volume/DeviceMount must be rejected",
        ):
            self.scheduler.submit_dryrun(app, cfg=Opts())

    def test_submit_dryrun_omits_unset_limits(self) -> None:
        app = specs.AppDef(
            "test",
            roles=[
                specs.Role(
                    name="worker",
                    image="pytorch/torchx:latest",
                    entrypoint="main",
                    resource=specs.Resource(cpu=-1, gpu=0, memMB=-1),
                )
            ],
        )

        info = self.scheduler.submit_dryrun(app, cfg=Opts())

        kwargs = info.request.containers[0].kwargs
        for key in (
            "restart_policy",
            "mem_limit",
            "shm_size",
            "nano_cpus",
            "device_requests",
        ):
            self.assertNotIn(
                key,
                kwargs,
                msg=f"unset retries/resources must not produce a docker `{key}` constraint",
            )
        self.assertEqual(
            {"TORCHX_RANK0_HOST", "TORCHX_IMAGE"},
            set(cast(dict[str, str], kwargs["environment"]).keys()),
            msg="a role without env vars must get exactly the torchx-injected ones",
        )

    def test_ensure_network_default_client(self) -> None:
        client = MagicMock()
        with (
            patch("docker.from_env", return_value=client) as from_env_ctx,
            patch("filelock.FileLock"),
        ):
            ensure_network()
        from_env_ctx.assert_called_once_with()
        client.networks.create.assert_called_once_with(
            name="torchx", driver="bridge", check_duplicate=True
        )

    def test_ensure_network_swallows_already_exists(self) -> None:
        client = MagicMock()
        client.networks.create.side_effect = APIError("network already exists")

        with patch("filelock.FileLock"):
            ensure_network(client)

        client.networks.create.assert_called_once()

    def test_ensure_network_raises_other_api_errors(self) -> None:
        client = MagicMock()
        client.networks.create.side_effect = APIError("permission denied")

        with (
            patch("filelock.FileLock"),
            self.assertRaisesRegex(
                APIError,
                "permission denied",
                msg="only the already-exists race is tolerated; other API errors must propagate",
            ),
        ):
            ensure_network(client)

    def test_schedule_pulls_each_image_once_and_skips_digests(self) -> None:
        job = DockerJob(
            app_id="app_id_1",
            containers=[
                DockerContainer(
                    image="sha256:0123", command=["echo"], kwargs={"name": "c0"}
                ),
                DockerContainer(
                    image="pytorch/torchx:latest",
                    command=["echo"],
                    kwargs={"name": "c1"},
                ),
                DockerContainer(
                    image="pytorch/torchx:latest",
                    command=["echo"],
                    kwargs={"name": "c2"},
                ),
            ],
        )
        client = MagicMock()

        with (
            patch.object(DockerScheduler, "_docker_client", client),
            patch(
                "torchx.schedulers.docker_scheduler.ensure_network"
            ) as ensure_network_ctx,
        ):
            app_id = self.scheduler.schedule(AppDryRunInfo(job, repr))

        self.assertEqual(
            "app_id_1", app_id, msg="schedule must return the request's app_id"
        )
        client.images.pull.assert_called_once_with("pytorch/torchx:latest")
        ensure_network_ctx.assert_called_once_with(client)
        self.assertEqual(
            [
                call("sha256:0123", ["echo"], detach=True, name="c0"),
                call("pytorch/torchx:latest", ["echo"], detach=True, name="c1"),
                call("pytorch/torchx:latest", ["echo"], detach=True, name="c2"),
            ],
            client.containers.run.call_args_list,
            msg="every container must be started detached with its dryrun kwargs",
        )

    def test_schedule_pull_failure_falls_back_to_local_image(self) -> None:
        job = DockerJob(
            app_id="app_id_1",
            containers=[
                DockerContainer(
                    image="pytorch/torchx:latest", command=["echo"], kwargs={}
                )
            ],
        )
        client = MagicMock()
        client.images.pull.side_effect = RuntimeError("registry unreachable")

        with (
            patch.object(DockerScheduler, "_docker_client", client),
            patch("torchx.schedulers.docker_scheduler.ensure_network"),
            self.assertLogs("torchx.schedulers.docker_scheduler", level="WARNING"),
        ):
            app_id = self.scheduler.schedule(AppDryRunInfo(job, repr))

        self.assertEqual(
            "app_id_1", app_id, msg="a failed pull must not abort scheduling"
        )
        client.containers.run.assert_called_once()

    def test_has_docker_with_healthy_daemon(self) -> None:
        with patch("docker.from_env", return_value=MagicMock()):
            self.assertTrue(
                has_docker(),
                msg="an importable docker module with a reachable daemon means docker is available",
            )

    def test_has_docker_daemon_unreachable(self) -> None:
        with patch("docker.from_env", side_effect=DockerException("daemon down")):
            self.assertFalse(
                has_docker(),
                msg="an unreachable docker daemon means docker is not available",
            )

    def test_validate_accepts_role_without_resource(self) -> None:
        app = _test_app()
        app.roles[0].resource = specs.NULL_RESOURCE

        with self.assertRaisesRegex(
            ValueError,
            "No resource for role",
            msg="the base Scheduler must reject a role without a resource,"
            " otherwise this test cannot discriminate the override",
        ):
            Scheduler._validate(self.scheduler, app, "local_docker", Opts())
        self.scheduler._validate(app, "local_docker", Opts())

    def test_describe_queries_docker_by_app_id_label(self) -> None:
        client = MagicMock()
        client.containers.list.return_value = [_mock_container()]

        with patch.object(DockerScheduler, "_docker_client", client):
            desc = self.scheduler.describe("app_id_1")

        self.assertIsNotNone(
            desc, msg="a docker listing with containers must yield a description"
        )
        client.containers.list.assert_called_once_with(
            all=True, filters={"label": f"{LABEL_APP_ID}=app_id_1"}
        )

    def test_log_iter_queries_docker_by_replica_labels(self) -> None:
        container = _mock_container()
        container.logs.return_value = b"foo\n"
        client = MagicMock()
        client.containers.list.return_value = [container]

        with patch.object(DockerScheduler, "_docker_client", client):
            logs = list(self.scheduler.log_iter("app_id_1", "trainer", 0))

        self.assertEqual(
            ["foo\n"],
            logs,
            msg="the sole label-matched container's logs must be returned",
        )
        client.containers.list.assert_called_once_with(
            all=True,
            filters={
                "label": [
                    f"{LABEL_APP_ID}=app_id_1",
                    f"{LABEL_ROLE_NAME}=trainer",
                    f"{LABEL_REPLICA_ID}=0",
                ]
            },
        )

    def test_log_iter_no_matching_container_raises(self) -> None:
        client = MagicMock()
        client.containers.list.return_value = []

        with patch.object(DockerScheduler, "_docker_client", client):
            with self.assertRaisesRegex(
                RuntimeError,
                "failed to find container",
                msg="a missing replica container must be a hard error",
            ):
                self.scheduler.log_iter("app_id_1", "trainer", 0)

    def test_log_iter_ambiguous_container_match_raises(self) -> None:
        client = MagicMock()
        client.containers.list.return_value = [_mock_container(), _mock_container()]

        with patch.object(DockerScheduler, "_docker_client", client):
            with self.assertRaisesRegex(
                RuntimeError,
                "found multiple containers",
                msg="an ambiguous replica label match must be a hard error",
            ):
                self.scheduler.log_iter("app_id_1", "trainer", 0)

    def test_cancel_stops_every_replica_container(self) -> None:
        containers = [_mock_container(status="running", replica_id=i) for i in range(2)]
        client = MagicMock()
        client.containers.list.return_value = containers

        with patch.object(DockerScheduler, "_docker_client", client):
            self.scheduler.cancel("app_id_1")

        for container in containers:
            container.stop.assert_called_once_with()

    def test_cancel_missing_app_is_noop(self) -> None:
        client = MagicMock()
        client.containers.list.return_value = []

        with patch.object(DockerScheduler, "_docker_client", client):
            self.scheduler.cancel("does-not-exist")

        self.assertEqual(
            1,
            client.containers.list.call_count,
            msg="cancel on a nonexistent app must stop after the exists() probe;"
            " stopping containers would query the docker listing a second time",
        )

    def test_describe_reports_first_non_terminal_state(self) -> None:
        containers = [
            _mock_container(status="exited", exit_code=0, replica_id=0),
            _mock_container(status="running", replica_id=1),
        ]

        client = MagicMock()
        client.containers.list.return_value = containers

        with patch.object(DockerScheduler, "_docker_client", client):
            desc = self.scheduler.describe("app_id_1")

        assert desc is not None, "an app with containers must have a description"
        self.assertEqual(
            AppState.RUNNING,
            desc.state,
            msg="an app with a non-terminal replica must report that replica's state",
        )
        self.assertEqual(
            1,
            len(desc.roles),
            msg="replicas of one role must collapse into a single Role entry",
        )
        self.assertEqual(
            2,
            desc.roles[0].num_replicas,
            msg="the reconstructed Role must count every replica container",
        )
        self.assertEqual(
            [(0, AppState.SUCCEEDED), (1, AppState.RUNNING)],
            [(r.id, r.state) for r in desc.roles_statuses[0].replicas],
            msg="per-replica ids and states must be reported in container order",
        )

    def test_describe_publishes_image_as_str(self) -> None:
        tagged = _mock_container()
        tagged.image.tags = ["pytorch/torchx:latest"]
        untagged = _mock_container()
        untagged.image.tags = []
        untagged.image.id = "sha256:0123"
        gone = _mock_container()
        gone.image = None

        for container, want, why in (
            (
                tagged,
                "pytorch/torchx:latest",
                "a tagged image must publish its first repo tag",
            ),
            (
                untagged,
                "sha256:0123",
                "an untagged image must fall back to the image id",
            ),
            (
                gone,
                specs.UNKNOWN,
                "a deleted image record must publish the UNKNOWN sentinel",
            ),
        ):
            with self.subTest(why=why):
                client = MagicMock()
                client.containers.list.return_value = [container]
                with patch.object(DockerScheduler, "_docker_client", client):
                    desc = self.scheduler.describe("app_id_1")
                assert (
                    desc is not None
                ), "an app with containers must have a description"
                self.assertEqual(want, desc.roles[0].image, msg=why)

    def test_describe_all_replicas_succeeded(self) -> None:
        containers = [
            _mock_container(status="exited", exit_code=0, replica_id=i)
            for i in range(2)
        ]

        client = MagicMock()
        client.containers.list.return_value = containers

        with patch.object(DockerScheduler, "_docker_client", client):
            desc = self.scheduler.describe("app_id_1")

        assert desc is not None, "an app with containers must have a description"
        self.assertEqual(
            AppState.SUCCEEDED,
            desc.state,
            msg="the app succeeds only when every replica succeeded",
        )

    def test_describe_terminal_with_failed_replica(self) -> None:
        containers = [
            _mock_container(status="exited", exit_code=0, replica_id=0),
            _mock_container(status="exited", exit_code=1, replica_id=1),
        ]

        client = MagicMock()
        client.containers.list.return_value = containers

        with patch.object(DockerScheduler, "_docker_client", client):
            desc = self.scheduler.describe("app_id_1")

        assert desc is not None, "an app with containers must have a description"
        self.assertEqual(
            AppState.FAILED,
            desc.state,
            msg="any failed replica must fail the terminal app state",
        )

    def test_log_iter_splits_byte_payload(self) -> None:
        container = _mock_container()
        container.logs.return_value = b"foo\nbar\n"

        client = MagicMock()
        client.containers.list.return_value = [container]

        with patch.object(DockerScheduler, "_docker_client", client):
            logs = list(self.scheduler.log_iter("app_id_1", "trainer", 0))

        self.assertEqual(
            ["foo\n", "bar\n"],
            logs,
            msg="a non-streaming byte payload must be decoded and split into lines",
        )
        container.logs.assert_called_once_with(
            since=None, until=None, stream=False, stderr=True, stdout=True
        )

    def test_log_iter_empty_payload(self) -> None:
        container = _mock_container()
        container.logs.return_value = b""

        client = MagicMock()
        client.containers.list.return_value = [container]

        with patch.object(DockerScheduler, "_docker_client", client):
            logs = list(self.scheduler.log_iter("app_id_1", "trainer", 0))

        self.assertEqual([], logs, msg="an empty log payload must yield no lines")

    def test_log_iter_stream_with_regex(self) -> None:
        container = _mock_container()
        container.logs.return_value = iter([b"foo\n", b"bar\n"])

        client = MagicMock()
        client.containers.list.return_value = [container]

        with patch.object(DockerScheduler, "_docker_client", client):
            logs = list(
                self.scheduler.log_iter(
                    "app_id_1",
                    "trainer",
                    0,
                    regex="bar",
                    should_tail=True,
                    streams=Stream.STDOUT,
                )
            )

        self.assertEqual(
            ["bar\n"],
            logs,
            msg="streamed byte chunks must be decoded and regex-filtered",
        )
        container.logs.assert_called_once_with(
            since=None, until=None, stream=True, stderr=False, stdout=True
        )

    def test_list_dedupes_replicas_of_same_app(self) -> None:
        client = MagicMock()
        client.containers.list.return_value = [
            _mock_container(status="running", replica_id=0),
            _mock_container(status="running", replica_id=1),
        ]

        with patch.object(DockerScheduler, "_docker_client", client):
            apps = self.scheduler.list()

        self.assertEqual(
            [ListAppResponse(app_id="app_id_1", state=AppState.RUNNING)],
            apps,
            msg="containers sharing an app-id must dedupe to one ListAppResponse",
        )


if has_docker():
    # These are the live tests that require a local docker instance.

    class DockerSchedulerLiveTest(unittest.TestCase, LocalSchedulerTestUtil):
        def setUp(self) -> None:
            # pyrefly: ignore [bad-override-mutable-attribute]
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
            app_id = self.scheduler.submit(app, cfg=Opts())

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
            app_id = self.scheduler.submit(app, cfg=Opts())
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

            app_id = self.scheduler.submit(app, cfg=Opts())
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
            app_id = self.scheduler.submit(app, cfg=Opts())

            self.wait(app_id)
            self.assertTrue(
                ListAppResponse(app_id=app_id, state=AppState.SUCCEEDED)
                in self.scheduler.list()
            )

        def test_docker_cancel(self) -> None:
            app = self._docker_app("sleep", "10000")
            app_id = self.scheduler.submit(app, cfg=Opts())
            _ = self.scheduler.describe(app_id)

            self.wait(app_id, wait_for=lambda state: state == AppState.RUNNING)
            self.scheduler.cancel(app_id)

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(desc.state, AppState.FAILED)

        def test_docker_submit_error(self) -> None:
            app = self._docker_app("sh", "-c", "exit 1")
            app_id = self.scheduler.submit(app, cfg=Opts())

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
            app_id = self.scheduler.submit(app, cfg=Opts())

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.FAILED, desc.state)

        def test_docker_submit_dist(self) -> None:
            workspace = "memory://docker_submit_dist/"
            with fsspec.open(posixpath.join(workspace, "main.py"), "wt") as f:
                f.write("print('hello world')\n")
            app = ddp(script="main.py", j="2x1")
            app_id = self.scheduler.submit(app, cfg=Opts(), workspace=workspace)
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
