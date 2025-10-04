#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import datetime
import os
from contextlib import contextmanager
from typing import cast, Generator, List, Mapping, Optional
from unittest.mock import MagicMock, patch

from torchx.runner import get_runner, Runner
from torchx.schedulers import SchedulerFactory
from torchx.schedulers.api import DescribeAppResponse, ListAppResponse, Scheduler
from torchx.schedulers.local_scheduler import (
    create_scheduler,
    LocalDirectoryImageProvider,
)
from torchx.specs import AppDryRunInfo, CfgVal
from torchx.specs.api import (
    AppDef,
    AppHandle,
    AppState,
    parse_app_handle,
    Resource,
    Role,
    runopts,
    UnknownAppException,
)
from torchx.specs.finder import ComponentNotFoundException
from torchx.test.fixtures import TestWithTmpDir
from torchx.tracker.api import ENV_TORCHX_JOB_ID, ENV_TORCHX_PARENT_RUN_ID
from torchx.util.types import none_throws
from torchx.workspace import WorkspaceMixin

GET_SCHEDULER_FACTORIES = "torchx.runner.api.get_scheduler_factories"


class resource:
    SMALL = Resource(cpu=1, gpu=0, memMB=1024)
    MEDIUM = Resource(cpu=4, gpu=0, memMB=(4 * 1024))
    LARGE = Resource(cpu=16, gpu=0, memMB=(16 * 1024))


SESSION_NAME = "test_session"


def get_full_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "resource", name)


@patch("torchx.runner.events.record")
class RunnerTest(TestWithTmpDir):
    def setUp(self) -> None:
        super().setUp()
        self.write_shell_script("touch.sh", ["touch $1"])
        self.write_shell_script("fail.sh", ["exit 1"])
        self.write_shell_script("sleep.sh", ["sleep $1"])

        self.cfg = {}

    @contextmanager
    def get_runner(self) -> Generator[Runner, None, None]:
        with Runner(
            SESSION_NAME,
            scheduler_factories={"local_dir": cast(SchedulerFactory, create_scheduler)},
            scheduler_params={
                "image_provider_class": LocalDirectoryImageProvider,
            },
        ) as runner:
            yield runner

    def test_validate_no_roles(self, _) -> None:
        with self.get_runner() as runner:
            app = AppDef("no roles")
            with self.assertRaises(ValueError):
                runner.run(app, scheduler="local_dir")

    def test_validate_no_resource(self, _) -> None:
        with self.get_runner() as runner:
            role = Role(
                "no resource",
                image="no_image",
                entrypoint="echo",
                args=["hello_world"],
            )
            app = AppDef("no resource", roles=[role])
            with self.assertRaises(ValueError):
                runner.run(app, scheduler="local_dir")

    def test_validate_invalid_replicas(self, _) -> None:
        with self.get_runner() as runner:
            role = Role(
                "invalid replicas",
                image="torch",
                entrypoint="echo",
                args=["hello_world"],
                num_replicas=0,
                resource=Resource(cpu=1, gpu=0, memMB=500),
            )
            app = AppDef("invalid replicas", roles=[role])
            with self.assertRaises(ValueError):
                runner.run(app, scheduler="local_dir")

    def test_session_id(self, record_mock: MagicMock) -> None:
        test_file = self.tmpdir / "test_file"

        with self.get_runner() as runner:
            self.assertEqual(1, len(runner.scheduler_backends()))
            role = Role(
                name="touch",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="touch.sh",
                args=[str(test_file)],
            )
            app = AppDef("name", roles=[role])

            app_handle_1 = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            none_throws(runner.wait(app_handle_1, wait_interval=0.1))

            app_handle_2 = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            none_throws(runner.wait(app_handle_2, wait_interval=0.1))

            from torchx.util.session import CURRENT_SESSION_ID

            self.assertIsNotNone(CURRENT_SESSION_ID)
            record_mock.assert_called()
            for i in range(record_mock.call_count):
                event = record_mock.call_args_list[i].args[0]
                self.assertEqual(event.session, CURRENT_SESSION_ID)

    def test_empty_session_id(self, _: MagicMock) -> None:
        empty_session_name = ""
        with Runner(
            empty_session_name,
            {"local": cast(SchedulerFactory, create_scheduler)},
        ) as runner:
            app = AppDef(
                "__unused_for_test__",
                roles=[
                    Role(
                        name="echo",
                        image=str(self.tmpdir),
                        resource=resource.SMALL,
                        entrypoint="echo",
                        args=["__unused_for_test__"],
                    )
                ],
            )

            app_handle = runner.run(app, "local", self.cfg)
            runner.wait(app_handle, wait_interval=0.1)

            scheduler, session_name, app_id = parse_app_handle(app_handle)
            self.assertEqual(scheduler, "local")
            self.assertEqual(empty_session_name, session_name)
            self.assertEqual(app_handle, f"local:///{app_id}")

    def test_run(self, record_mock: MagicMock) -> None:
        test_file = self.tmpdir / "test_file"

        with self.get_runner() as runner:
            self.assertEqual(1, len(runner.scheduler_backends()))
            role = Role(
                name="touch",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="touch.sh",
                args=[str(test_file)],
            )
            app = AppDef("name", roles=[role])

            app_handle = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            app_status = none_throws(runner.wait(app_handle, wait_interval=0.1))
            self.assertEqual(AppState.SUCCEEDED, app_status.state)
            from torchx.util.session import CURRENT_SESSION_ID

            self.assertIsNotNone(CURRENT_SESSION_ID)
            record_mock.assert_called()
            for i in range(record_mock.call_count):
                event = record_mock.call_args_list[i].args[0]
                self.assertEqual(event.session, CURRENT_SESSION_ID)

    def test_dryrun(self, record_mock: MagicMock) -> None:
        scheduler_mock = MagicMock()
        scheduler_mock.run_opts.return_value.resolve.return_value = {
            **self.cfg,
            "foo": "bar",
        }
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            role = Role(
                name="touch",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role])
            runner.dryrun(app, "local_dir", cfg=self.cfg)
            scheduler_mock.submit_dryrun.assert_called_once_with(
                app, {**self.cfg, "foo": "bar"}
            )
            scheduler_mock._validate.assert_called_once()
            from torchx.util.session import CURRENT_SESSION_ID

            self.assertIsNotNone(CURRENT_SESSION_ID)
            record_mock.assert_called()
            for i in range(record_mock.call_count):
                event = record_mock.call_args_list[i].args[0]
                self.assertEqual(event.session, CURRENT_SESSION_ID)

    def test_dryrun_env_variables(self, _) -> None:
        scheduler_mock = MagicMock()
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(app, "local_dir", cfg=self.cfg)
            for role in app.roles:
                self.assertEqual(
                    role.env[ENV_TORCHX_JOB_ID],
                    "local_dir://" + SESSION_NAME + "/${app_id}",
                )

    def test_dryrun_trackers_parent_run_id_as_paramenter(self, _) -> None:
        scheduler_mock = MagicMock()
        expected_parent_run_id = "123"
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(
                app, "local_dir", cfg=self.cfg, parent_run_id=expected_parent_run_id
            )
            for role in app.roles:
                self.assertEqual(
                    role.env[ENV_TORCHX_PARENT_RUN_ID],
                    expected_parent_run_id,
                )

    @patch("torchx.runner.api.get_configured_trackers")
    def test_dryrun_setup_trackers(self, config_trackers_mock: MagicMock, _) -> None:
        config_trackers_mock.return_value = {
            "my_tracker1": "manifold://config1.txt",
            "my_tracker2": "manifold://config2.txt",
        }
        scheduler_mock = MagicMock()
        expected_trackers = "my_tracker1,my_tracker2"
        expected_tracker1_config = "manifold://config1.txt"
        expected_tracker2_config = "manifold://config2.txt"

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(app, "local_dir", cfg=self.cfg, parent_run_id="999")
            for role in app.roles:
                self.assertEqual(
                    role.env["TORCHX_TRACKERS"],
                    expected_trackers,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER1_CONFIG"],
                    expected_tracker1_config,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER2_CONFIG"],
                    expected_tracker2_config,
                )

    @patch.dict(
        os.environ,
        {
            "TORCHX_TRACKERS": "my_tracker1,my_tracker2",
            "TORCHX_TRACKER_MY_TRACKER1_CONFIG": "manifold://config1.txt",
            "TORCHX_TRACKER_MY_TRACKER2_CONFIG": "manifold://config2.txt",
        },
    )
    def test_dryrun_setup_trackers_as_env_variable(self, _) -> None:
        scheduler_mock = MagicMock()
        expected_trackers = "my_tracker1,my_tracker2"
        expected_tracker1_config = "manifold://config1.txt"
        expected_tracker2_config = "manifold://config2.txt"

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(app, "local_dir", cfg=self.cfg, parent_run_id="999")
            for role in app.roles:
                self.assertEqual(
                    role.env["TORCHX_TRACKERS"],
                    expected_trackers,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER1_CONFIG"],
                    expected_tracker1_config,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER2_CONFIG"],
                    expected_tracker2_config,
                )

    def test_dryrun_with_workspace(self, _) -> None:
        class TestScheduler(WorkspaceMixin[None], Scheduler):
            def __init__(self, build_new_img: bool):
                super().__init__(backend="ignored", session_name="ignored")
                self.build_new_img = build_new_img

            def schedule(self, dryrun_info: AppDryRunInfo) -> str:
                pass

            def _submit_dryrun(
                self, app: AppDef, cfg: Mapping[str, CfgVal]
            ) -> AppDryRunInfo[AppDef]:
                return AppDryRunInfo(app, lambda s: s)

            def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
                pass

            def list(self) -> List[DescribeAppResponse]:
                pass

            def _cancel_existing(self, app_id: str) -> None:
                pass

            def build_workspace_and_update_role(
                self,
                role: Role,
                workspace: str,
                cfg: Mapping[str, CfgVal],
            ) -> None:
                if self.build_new_img:
                    role.image = f"{role.image}_new"

        with Runner(
            name=SESSION_NAME,
            # pyre-fixme[6]: scheduler factory type
            scheduler_factories={
                "no-build-img": lambda name, **kwargs: TestScheduler(
                    build_new_img=False
                ),
                "builds-img": lambda name, **kwargs: TestScheduler(build_new_img=True),
            },
        ) as runner:
            app = AppDef(
                "ignored",
                roles=[
                    Role(
                        name="sleep",
                        image="foo",
                        resource=resource.SMALL,
                        entrypoint="sleep",
                        args=["1"],
                    ),
                    Role(
                        name="sleep",
                        image="bar",
                        resource=resource.SMALL,
                        entrypoint="sleep",
                        args=["1"],
                    ),
                ],
            )
            dryruninfo = runner.dryrun(app, "no-build-img", workspace="//workspace")
            self.assertEqual("foo", dryruninfo.request.roles[0].image)
            self.assertEqual("bar", dryruninfo.request.roles[1].image)

            dryruninfo = runner.dryrun(app, "builds-img", workspace="//workspace")
            # workspace is attached to role[0] by default
            self.assertEqual("foo_new", dryruninfo.request.roles[0].image)
            self.assertEqual("bar", dryruninfo.request.roles[1].image)

    def test_describe(self, _) -> None:
        with self.get_runner() as runner:
            metadata = {"a": "b", "c": "d"}
            role = Role(
                name="sleep",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="sleep",
                args=["60"],
            )
            app = AppDef("sleeper", roles=[role], metadata=metadata)

            app_handle = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            self.assertEqual(app, runner.describe(app_handle))
            # unknown app should return None
            self.assertIsNone(runner.describe("local_dir://session1/unknown_app"))

    def test_status(self, _) -> None:
        with self.get_runner() as runner:
            role = Role(
                name="sleep",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="sleep",
                args=["60"],
            )
            app = AppDef("sleeper", roles=[role])
            app_handle = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            app_status = none_throws(runner.status(app_handle))
            self.assertEqual(AppState.RUNNING, app_status.state)
            runner.cancel(app_handle)
            app_status = none_throws(runner.status(app_handle))
            self.assertEqual(AppState.CANCELLED, app_status.state)

    def test_status_unknown_app(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(runner.status("local_dir://test_session/unknown_app_id"))

    @patch("json.dumps")
    def test_status_ui_url(self, json_dumps_mock: MagicMock, _) -> None:
        app_id = "test_app"
        json_dumps_mock.return_value = "{}"
        mock_scheduler = MagicMock()
        resp = DescribeAppResponse()
        resp.ui_url = "https://foobar"
        mock_scheduler.submit.return_value = app_id
        mock_scheduler.describe.return_value = resp

        with Runner(
            name="test_ui_url_session",
            scheduler_factories={"local_dir": lambda name, **kwargs: mock_scheduler},
        ) as runner:
            role = Role(
                "ignored",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="/bin/echo",
            )
            app_handle = runner.run(AppDef(app_id, roles=[role]), scheduler="local_dir")
            status = none_throws(runner.status(app_handle))
            self.assertEqual(resp.ui_url, status.ui_url)

    @patch("json.dumps")
    def test_status_structured_msg(self, json_dumps_mock: MagicMock, _) -> None:
        app_id = "test_app"
        json_dumps_mock.return_value = "{}"
        mock_scheduler = MagicMock()
        resp = DescribeAppResponse()
        resp.structured_error_msg = '{"message": "test error"}'
        mock_scheduler.submit.return_value = app_id
        mock_scheduler.describe.return_value = resp

        with Runner(
            name="test_structured_msg",
            scheduler_factories={"local_dir": lambda name, **kwargs: mock_scheduler},
        ) as runner:
            role = Role(
                "ignored",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="/bin/echo",
            )
            app_handle = runner.run(AppDef(app_id, roles=[role]), scheduler="local_dir")
            status = none_throws(runner.status(app_handle))
            self.assertEqual(resp.structured_error_msg, status.structured_error_msg)

    def test_wait_unknown_app(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(
                runner.wait(
                    "local_dir://test_session/unknown_app_id", wait_interval=0.1
                )
            )
            self.assertIsNone(
                runner.wait("local_dir://another_session/some_app", wait_interval=0.1)
            )

    def test_cancel(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(runner.cancel("local_dir://test_session/unknown_app_id"))

    def test_stop(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(runner.stop("local_dir://test_session/unknown_app_id"))

    def test_log_lines_unknown_app(self, _) -> None:
        with self.get_runner() as runner:
            with self.assertRaises(UnknownAppException):
                runner.log_lines("local_dir://test_session/unknown", "trainer")

    def test_log_lines(self, _) -> None:
        app_id = "mock_app"

        scheduler_mock = MagicMock()
        scheduler_mock.describe.return_value = DescribeAppResponse(
            app_id, AppState.RUNNING
        )
        scheduler_mock.log_iter.return_value = iter(["hello", "world"])

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            role_name = "trainer"
            replica_id = 2
            regex = "QPS.*"
            since = datetime.datetime.now()
            until = datetime.datetime.now()
            lines = list(
                runner.log_lines(
                    f"local_dir://test_session/{app_id}",
                    role_name,
                    replica_id,
                    regex,
                    since,
                    until,
                )
            )

            self.assertEqual(["hello", "world"], lines)
            scheduler_mock.log_iter.assert_called_once_with(
                app_id, role_name, replica_id, regex, since, until, False, streams=None
            )

    def test_list(self, _) -> None:
        scheduler_mock = MagicMock()
        sched_list_return = [
            ListAppResponse(app_id="app_id1", state=AppState.RUNNING),
            ListAppResponse(app_id="app_id2", state=AppState.SUCCEEDED),
        ]
        scheduler_mock.list.return_value = sched_list_return
        apps_expected = [
            ListAppResponse(
                app_id="app_id1",
                app_handle="kubernetes://test_session/app_id1",
                state=AppState.RUNNING,
            ),
            ListAppResponse(
                app_id="app_id2",
                app_handle="kubernetes://test_session/app_id2",
                state=AppState.SUCCEEDED,
            ),
        ]
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"kubernetes": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            apps = runner.list("kubernetes")
            self.assertEqual(apps, apps_expected)
            scheduler_mock.list.assert_called_once()

    @patch("json.dumps")
    def test_get_schedulers(self, json_dumps_mock: MagicMock, _) -> None:
        local_dir_sched_mock = MagicMock()
        json_dumps_mock.return_value = "{}"
        local_sched_mock = MagicMock()
        scheduler_factories = {
            "local_dir": lambda name, **kwargs: local_dir_sched_mock,
            "local": lambda name, **kwargs: local_sched_mock,
        }
        with Runner(
            name="test_session", scheduler_factories=scheduler_factories
        ) as runner:
            role = Role(
                name="sleep",
                image=str(self.tmpdir),
                resource=resource.SMALL,
                entrypoint="sleep.sh",
                args=["60"],
            )
            app = AppDef("sleeper", roles=[role])
            runner.run(app, scheduler="local")
            local_sched_mock.schedule.assert_called_once()

    def test_run_from_module(self, _: str) -> None:
        runner = get_runner(name="test_session")
        app_args = ["--image", "dummy_image", "--script", "test.py"]
        with patch.object(
            runner, "schedule", return_value=AppHandle("sample://torchx/app_handle")
        ), patch.object(runner, "dryrun") as dryrun_mock:
            _ = runner.run_component("dist.ddp", app_args, "local")
            args, kwargs = dryrun_mock.call_args
            actual_app = args[0]

        print(actual_app)
        self.assertEqual(actual_app.name, "test")
        self.assertEqual(1, len(actual_app.roles))
        self.assertEqual("test", actual_app.roles[0].name)

    def test_run_from_file_no_function_found(self, _) -> None:
        local_sched_mock = MagicMock()
        schedulers = {
            "local_dir": lambda name, **kwargs: local_sched_mock,
            "local": lambda name, **kwargs: local_sched_mock,
        }
        with Runner(name="test_session", scheduler_factories=schedulers) as runner:
            component_path = get_full_path("distributed.py")
            with patch.object(runner, "run"):
                with self.assertRaises(ComponentNotFoundException):
                    runner.run_component(
                        f"{component_path}:unknown_function", [], "local"
                    )

    def test_runner_context_manager(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(
            GET_SCHEDULER_FACTORIES,
            return_value={"local_dir": lambda name, **kwargs: mock_scheduler},
        ):
            with get_runner() as runner:
                # force schedulers to load
                runner.scheduler_run_opts("local_dir")
        mock_scheduler.close.assert_called_once()

    def test_runner_context_manager_with_error(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(
            GET_SCHEDULER_FACTORIES,
            return_value={"local_dir": lambda name, **kwargs: mock_scheduler},
        ):
            with self.assertRaisesRegex(RuntimeError, "foobar"):
                with get_runner():
                    raise RuntimeError("foobar")

    def test_runner_manual_close(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(
            GET_SCHEDULER_FACTORIES,
            return_value={"local_dir": lambda name, **kwargs: mock_scheduler},
        ):
            runner = get_runner()
            # force schedulers to load
            runner.scheduler_run_opts("local_dir")
            runner.close()

        mock_scheduler.close.assert_called_once()

        # can call close twice
        runner.close()

    def test_get_default_runner(self, _) -> None:
        runner = get_runner()
        self.assertEqual("torchx", runner._name)

    def test_cfg_from_str(self, _) -> None:
        scheduler_mock = MagicMock()
        opts = runopts()
        opts.add("foo", type_=str, default="", help="")
        opts.add("test_key", type_=str, default="", help="")
        opts.add("default_time", type_=int, default=0, help="")
        opts.add("enable", type_=bool, default=True, help="")
        opts.add("disable", type_=bool, default=True, help="")
        opts.add("complex_list", type_=List[str], default=[], help="")
        scheduler_mock.run_opts.return_value = opts

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name, **kwargs: scheduler_mock},
        ) as runner:
            self.assertDictEqual(
                {
                    "foo": "bar",
                    "test_key": "test_value",
                    "default_time": 42,
                    "enable": True,
                    "disable": False,
                    "complex_list": ["v1", "v2", "v3"],
                },
                runner.cfg_from_str(
                    "local_dir",
                    "foo=bar",
                    "test_key=test_value",
                    "default_time=42",
                    "enable=True,disable=False,complex_list=v1;v2;v3",
                ),
            )
