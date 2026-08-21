#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import concurrent
import copy
import os
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import cast, Dict, List, Mapping, Union
from unittest import mock
from unittest.mock import MagicMock

from torchx import specs
from torchx.specs import named_resources, named_resources_aws, resource
from torchx.specs.api import (
    _OVERRIDES_LOCK_KEY,
    _OverridesLock,
    _TERMINAL_STATES,
    AppDef,
    AppDryRunInfo,
    AppState,
    AppStatus,
    AppStatusError,
    cases,
    CfgVal,
    get_type_name,
    InvalidRunConfigException,
    macros,
    MalformedAppHandleException,
    MISSING,
    NULL_RESOURCE,
    parse_app_handle,
    ReplicaStatus,
    Resource,
    RetryPolicy,
    Role,
    RoleStatus,
    runopt,
    runopts,
    TORCHX_HOME,
    UNKNOWN,
    Workspace,
)
from torchx.test.fixtures import TestWithTmpDir


class TorchXHomeTest(unittest.TestCase):
    # guard against TORCHX_HOME set outside the test
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_TORCHX_HOME_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            user_home = Path(tmpdir) / "sally"
            with mock.patch("pathlib.Path.home", return_value=user_home):
                torchx_home = TORCHX_HOME()
                self.assertEqual(torchx_home, user_home / ".torchx")
                self.assertTrue(torchx_home.exists())

    def test_TORCHX_HOME_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            override_torchx_home = Path(tmpdir) / "test" / ".torchx"
            with mock.patch.dict(
                os.environ, {"TORCHX_HOME": str(override_torchx_home)}
            ):
                torchx_home = TORCHX_HOME()
                conda_pack_out = TORCHX_HOME("conda-pack", "out")

                self.assertEqual(override_torchx_home, torchx_home)
                self.assertEqual(torchx_home / "conda-pack" / "out", conda_pack_out)

                self.assertTrue(torchx_home.is_dir())
                self.assertTrue(conda_pack_out.is_dir())


class WorkspaceTest(TestWithTmpDir):

    def test_bool(self) -> None:
        self.assertFalse(Workspace(projects={}))
        self.assertFalse(Workspace.from_str(""))

        self.assertTrue(Workspace(projects={"/home/foo/bar": ""}))
        self.assertTrue(Workspace.from_str("/home/foo/bar"))

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

    def test_merge(self) -> None:
        self.touch("workspace/myproj/README.md")
        self.touch("workspace/myproj/bin/cli")

        self.touch("workspace/torch/setup.py")
        self.touch("workspace/torch/torch/__init__.py")

        w = Workspace(
            projects={
                str(self.tmpdir / "workspace/myproj"): "",
                str(self.tmpdir / "workspace/torch"): "torch",
            }
        )

        outdir = self.tmpdir / "out"
        w.merge_into(outdir)

        self.assertTrue((outdir / "README.md").is_file())
        self.assertTrue((outdir / "bin/cli").is_file())
        self.assertTrue((outdir / "torch/setup.py").is_file())
        self.assertTrue((outdir / "torch/torch/__init__.py").is_file())


class AppDryRunInfoTest(unittest.TestCase):
    def test_repr(self) -> None:
        request_mock = MagicMock()
        to_string_mock = MagicMock()
        info = AppDryRunInfo(request_mock, to_string_mock)
        info.__repr__()
        self.assertEqual(request_mock, info.request)

        to_string_mock.assert_called_once_with(request_mock)

    def test_app_and_cfg_accessors(self) -> None:
        info = AppDryRunInfo(MagicMock(), repr)

        self.assertIsNone(info.app, "app should be None until set by dryrun")
        self.assertEqual({}, dict(info.cfg), "cfg should default to empty")

        app = AppDef(name="test_app")
        info._app = app
        info._cfg = {"cluster": "foo", "priority": 1}

        self.assertIs(app, info.app, "app property should return the AppDef")
        self.assertEqual(
            {"cluster": "foo", "priority": 1},
            dict(info.cfg),
            "cfg property should reflect the resolved cfg",
        )

    def test_cfg_is_read_only(self) -> None:
        info = AppDryRunInfo(MagicMock(), repr)
        info._cfg = {"cluster": "foo"}

        # cast defeats the static Mapping protocol (no `__setitem__`) so the
        # RUNTIME read-only guarantee (MappingProxyType) is what's exercised
        cfg = cast(dict[str, CfgVal], info.cfg)
        with self.assertRaises(TypeError, msg="mutating cfg view must raise"):
            cfg["cluster"] = "bar"


class AppDefStatusTest(unittest.TestCase):
    def test_is_terminal(self) -> None:
        for s in AppState:
            is_terminal = AppStatus(state=s).is_terminal()
            if s in _TERMINAL_STATES:
                self.assertTrue(is_terminal)
            else:
                self.assertFalse(is_terminal)

    def test_serialize(self) -> None:
        status = AppStatus(AppState.FAILED)
        serialized = repr(status)
        self.assertEqual(
            serialized,
            """AppStatus:
  msg: ''
  num_restarts: 0
  roles: []
  state: FAILED (5)
  structured_error_msg: <NONE>
  ui_url: null
""",
        )

    def test_serialize_embed_json(self) -> None:
        status = AppStatus(
            AppState.FAILED, structured_error_msg='{"message": "test error"}'
        )
        serialized = repr(status)
        self.assertEqual(
            serialized,
            """AppStatus:
  msg: ''
  num_restarts: 0
  roles: []
  state: FAILED (5)
  structured_error_msg:
    message: test error
  ui_url: null
""",
        )

    def test_raise_on_status(self) -> None:
        AppStatus(state=AppState.SUCCEEDED).raise_for_status()

        with self.assertRaisesRegex(
            AppStatusError, r"(?s)job did not succeed:.*FAILED.*"
        ):
            AppStatus(state=AppState.FAILED).raise_for_status()

        with self.assertRaisesRegex(
            AppStatusError, r"(?s)job did not succeed:.*CANCELLED.*"
        ):
            AppStatus(state=AppState.CANCELLED).raise_for_status()

        with self.assertRaisesRegex(
            AppStatusError, r"(?s)job did not succeed:.*RUNNING.*"
        ):
            AppStatus(state=AppState.RUNNING).raise_for_status()

    def test_format_error_message(self) -> None:
        rpc_error_message = """RuntimeError('On WorkerInfo(id=1, name=trainer:0:0):
RuntimeError(ShardingError('Table of size 715.26GB cannot be added to any rank'))
Traceback (most recent call last):
..
')
Traceback (most recent call last):
  File "/dev/shm/uid-0/360e3568-seed-nspid4026541870-ns-4026541866/torch/distributed/rpc/internal.py", line 190, in _run_function
"""
        expected_error_message = """RuntimeError('On WorkerInfo(id=1, name=trainer:0:0):
RuntimeError(ShardingError('Table
 of size 715.26GB cannot be added to any rank'))
Traceback (most recent call last):
..
')"""
        status = AppStatus(state=AppState.FAILED)
        actual_message = status._format_error_message(
            rpc_error_message, header="", width=80
        )
        self.assertEqual(expected_error_message, actual_message)

    def _get_test_app_status(self) -> AppStatus:
        error_msg = '{"message":{"message":"error","errorCode":-1,"extraInfo":{"timestamp":1293182}}}'
        replica1 = ReplicaStatus(
            id=0,
            state=AppState.FAILED,
            role="worker",
            hostname="localhost",
            structured_error_msg=error_msg,
        )

        replica2 = ReplicaStatus(
            id=1,
            state=AppState.RUNNING,
            role="worker",
            hostname="localhost",
        )

        role_status = RoleStatus(role="worker", replicas=[replica1, replica2])
        return AppStatus(state=AppState.RUNNING, roles=[role_status])

    def test_format_app_status(self) -> None:
        os.environ["TZ"] = "Europe/London"
        time.tzset()

        app_status = self._get_test_app_status()
        actual_message = app_status.format()
        expected_message = """AppStatus:
    State: RUNNING
    Num Restarts: 0
    Roles:
 *worker[0]:FAILED (exitcode: -1)
        timestamp: 1970-01-16 00:13:02
        hostname: localhost
        error_msg: error
  worker[1]:RUNNING
    Msg:
    Structured Error Msg: <NONE>
    UI URL: None
    """
        # Split and compare to aviod AssertionError.
        self.assertEqual(expected_message.split(), actual_message.split())

    def _get_test_app_status_with_error_msg(self, error_msg: str) -> AppStatus:
        replica = ReplicaStatus(
            id=0,
            state=AppState.FAILED,
            role="worker",
            hostname="localhost",
            structured_error_msg=error_msg,
        )
        role_status = RoleStatus(role="worker", replicas=[replica])
        return AppStatus(state=AppState.RUNNING, roles=[role_status])

    def test_format_app_status_flat_error_schema(self) -> None:
        # Flat reply-file schema (e.g. MAST's MastReplyFileMessage): "message"
        # is a string and timestamp/errorCode live at the top level.
        os.environ["TZ"] = "Europe/London"
        time.tzset()

        app_status = self._get_test_app_status_with_error_msg(
            '{"message": "InjectedFailure: Injected failure exception",'
            ' "timestamp": 1293182, "timestamp_us": 1293182000000,'
            ' "errorService": "Mast", "errorCode": 1,'
            ' "pyCallStack": "Traceback (most recent call last): ..."}'
        )
        actual_message = app_status.format()
        expected_message = """AppStatus:
    State: RUNNING
    Num Restarts: 0
    Roles:
 *worker[0]:FAILED (exitcode: 1)
        timestamp: 1970-01-16 00:13:02
        hostname: localhost
        error_msg: InjectedFailure: Injected failure exception
    Msg:
    Structured Error Msg: <NONE>
    UI URL: None
    """
        self.assertEqual(expected_message.split(), actual_message.split())

    def test_format_app_status_flat_error_schema_missing_fields(self) -> None:
        app_status = self._get_test_app_status_with_error_msg(
            '{"message": "InjectedFailure: Injected failure exception"}'
        )
        actual_message = app_status.format()
        self.assertIn("FAILED (exitcode: <N/A>)", actual_message)
        self.assertIn("timestamp: <N/A>", actual_message)
        self.assertIn(
            "error_msg: InjectedFailure: Injected failure exception", actual_message
        )

    def test_format_app_status_nested_error_schema_missing_fields(self) -> None:
        app_status = self._get_test_app_status_with_error_msg(
            '{"message":{"message":"test error"}}'
        )
        actual_message = app_status.format()
        self.assertIn("FAILED (exitcode: <N/A>)", actual_message)
        self.assertIn("timestamp: <N/A>", actual_message)
        self.assertIn("error_msg: test error", actual_message)

    def test_format_app_status_unrecognized_error_schema(self) -> None:
        # Valid JSON that matches neither known reply-file schema renders
        # verbatim instead of raising.
        for error_msg in (
            '["not", "a", "dict"]',
            '{"error": "no message key"}',
            '{"message": {"message": 5}}',
            '{"message": null}',
            '"just a quoted string"',
        ):
            with self.subTest(error_msg=error_msg):
                app_status = self._get_test_app_status_with_error_msg(error_msg)
                self.assertIn(error_msg, app_status.format())

    def test_serialize_non_json_error(self) -> None:
        status = AppStatus(
            AppState.FAILED, structured_error_msg="worker terminated by SIGKILL"
        )
        self.assertIn("worker terminated by SIGKILL", repr(status))

        with self.assertRaisesRegex(
            AppStatusError, r"(?s)job did not succeed:.*FAILED.*"
        ):
            status.raise_for_status()

    def test_app_status_in_json(self) -> None:
        app_status = self._get_test_app_status()
        result = app_status.to_json()
        error_msg = '{"message":{"message":"error","errorCode":-1,"extraInfo":{"timestamp":1293182}}}'
        self.assertDictEqual(
            result,
            {
                "state": "RUNNING",
                "num_restarts": 0,
                "roles": [
                    {
                        "role": "worker",
                        "replicas": [
                            {
                                "id": 0,
                                "state": 5,
                                "role": "worker",
                                "hostname": "localhost",
                                "structured_error_msg": error_msg,
                                "hostaddr": "localhost",
                            },
                            {
                                "id": 1,
                                "state": 3,
                                "role": "worker",
                                "hostname": "localhost",
                                "structured_error_msg": "<NONE>",
                                "hostaddr": "localhost",
                            },
                        ],
                    }
                ],
                "msg": "",
                "structured_error_msg": "<NONE>",
                "url": None,
            },
        )


class ResourceTest(unittest.TestCase):
    def test_copy_resource(self) -> None:
        old_capabilities = {"test_key": "test_value", "old_key": "old_value"}
        resource = Resource(1, 2, 3, old_capabilities)
        new_resource = Resource.copy(
            resource, test_key="test_value_new", new_key="new_value"
        )
        self.assertEqual(new_resource.cpu, 1)
        self.assertEqual(new_resource.gpu, 2)
        self.assertEqual(new_resource.memMB, 3)

        self.assertEqual(len(new_resource.capabilities), 3)
        self.assertEqual(new_resource.capabilities["old_key"], "old_value")
        self.assertEqual(new_resource.capabilities["test_key"], "test_value_new")
        self.assertEqual(new_resource.capabilities["new_key"], "new_value")
        self.assertEqual(resource.capabilities["test_key"], "test_value")

    def test_copy_resource_copies_devices_and_tags(self) -> None:
        resource = Resource(
            1,
            2,
            3,
            devices={"vpc.amazonaws.com/efa": 4},
            tags={"resource_name": "test"},
        )
        new_resource = Resource.copy(resource)

        self.assertEqual(
            new_resource.devices,
            {"vpc.amazonaws.com/efa": 4},
            "copy must carry the original's devices",
        )
        self.assertEqual(
            new_resource.tags,
            {"resource_name": "test"},
            "copy must carry the original's tags",
        )

        new_resource.devices["nvidia.com/gpu"] = 1
        new_resource.tags["extra"] = "x"
        self.assertEqual(
            resource.devices,
            {"vpc.amazonaws.com/efa": 4},
            "mutating the copy's devices must not leak into the original",
        )
        self.assertEqual(
            resource.tags,
            {"resource_name": "test"},
            "mutating the copy's tags must not leak into the original",
        )

    def test_is_fractional_default(self) -> None:
        """Resource with no tags is not fractional."""
        res = Resource(cpu=4, gpu=1, memMB=1024)
        self.assertFalse(
            res.is_fractional(),
            "resource with no tags should not be fractional",
        )

    def test_is_fractional_true(self) -> None:
        """Resource tagged as fractional returns True."""
        from torchx.plugins._registration import resource_tags

        res = Resource(
            cpu=4,
            gpu=1,
            memMB=1024,
            tags={resource_tags.IS_FRACTIONAL: True},
        )
        self.assertTrue(
            res.is_fractional(),
            "resource tagged IS_FRACTIONAL=True should be fractional",
        )

    def test_is_fractional_false(self) -> None:
        """Resource explicitly tagged IS_FRACTIONAL=False is not fractional."""
        from torchx.plugins._registration import resource_tags

        res = Resource(
            cpu=4,
            gpu=1,
            memMB=1024,
            tags={resource_tags.IS_FRACTIONAL: False},
        )
        self.assertFalse(
            res.is_fractional(),
            "resource tagged IS_FRACTIONAL=False should not be fractional",
        )

    def test_get_resource_name_none(self) -> None:
        """Resource with no RESOURCE_NAME tag returns None."""
        res = Resource(cpu=4, gpu=1, memMB=1024)
        self.assertIsNone(
            res.get_resource_name(),
            "resource with no tags should return None for get_resource_name()",
        )

    def test_get_resource_name(self) -> None:
        """Resource tagged with RESOURCE_NAME returns the name as a string."""
        from torchx.plugins._registration import resource_tags

        res = Resource(
            cpu=4,
            gpu=1,
            memMB=1024,
            tags={resource_tags.RESOURCE_NAME: "gpu_4"},
        )
        self.assertEqual(
            res.get_resource_name(),
            "gpu_4",
            "should return the registered resource name",
        )

    def test_named_resources_iterator(self) -> None:
        registered_named_resources = set()
        for resource_name in named_resources:
            # just make sure we can create the resource using the name
            self.assertIsInstance(resource(h=resource_name), Resource)
            registered_named_resources.add(resource_name)

        # validate that the for-loop was not vacuous
        self.assertTrue(registered_named_resources)

    def test_named_resources(self) -> None:
        self.assertEqual(
            named_resources_aws.aws_m5_2xlarge(), named_resources["aws_m5.2xlarge"]
        )
        self.assertEqual(
            named_resources_aws.aws_t3_medium(), named_resources["aws_t3.medium"]
        )
        self.assertEqual(
            named_resources_aws.aws_p3_2xlarge(), named_resources["aws_p3.2xlarge"]
        )
        self.assertEqual(
            named_resources_aws.aws_p3_8xlarge(), named_resources["aws_p3.8xlarge"]
        )

    def test_named_resources_contains(self) -> None:
        self.assertTrue("aws_p3.8xlarge" in named_resources)
        self.assertFalse("nonexistant" in named_resources)

    def test_resource_util_fn(self) -> None:
        self.assertEqual(Resource(cpu=2, gpu=0, memMB=1024), resource())
        self.assertEqual(Resource(cpu=1, gpu=0, memMB=1024), resource(cpu=1))
        self.assertEqual(Resource(cpu=2, gpu=1, memMB=1024), resource(cpu=2, gpu=1))
        self.assertEqual(
            Resource(cpu=2, gpu=1, memMB=2048), resource(cpu=2, gpu=1, memMB=2048)
        )

        h = "aws_t3.medium"
        self.assertEqual(named_resources[h], resource(h=h))
        self.assertEqual(named_resources[h], resource(cpu=16, gpu=4, h="aws_t3.medium"))


class SentinelsTest(unittest.TestCase):
    def test_unknown(self) -> None:
        # the literal is the contract: schedulers that cannot read an attribute
        # back write it directly, without importing torchx
        self.assertEqual("<UNKNOWN>", UNKNOWN)
        self.assertEqual(UNKNOWN, specs.UNKNOWN)


class RoleBuilderTest(unittest.TestCase):
    def test_defaults(self) -> None:
        default = Role("foobar", "torch")
        self.assertEqual("foobar", default.name)
        self.assertEqual("torch", default.image)
        self.assertEqual(MISSING, default.entrypoint)
        self.assertEqual({}, default.env)
        self.assertEqual([], default.args)
        self.assertEqual(NULL_RESOURCE, default.resource)
        self.assertEqual(1, default.num_replicas)
        self.assertEqual(0, default.max_retries)
        self.assertEqual(RetryPolicy.APPLICATION, default.retry_policy)
        self.assertEqual({}, default.metadata)

    def test_build_role(self) -> None:
        # runs: ENV_VAR_1=FOOBAR /bin/echo hello world
        resource = Resource(cpu=1, gpu=2, memMB=128)
        trainer = Role(
            "trainer",
            image="torch",
            entrypoint="/bin/echo",
            args=["hello", "world"],
            env={"ENV_VAR_1": "FOOBAR"},
            num_replicas=2,
            retry_policy=RetryPolicy.REPLICA,
            max_retries=5,
            resource=resource,
            port_map={"foo": 8080},
            metadata={"foo": "bar"},
        )

        self.assertEqual("trainer", trainer.name)
        self.assertEqual("torch", trainer.image)
        self.assertEqual("/bin/echo", trainer.entrypoint)
        self.assertEqual({"ENV_VAR_1": "FOOBAR"}, trainer.env)
        self.assertEqual(["hello", "world"], trainer.args)
        self.assertDictEqual({"foo": "bar"}, trainer.metadata)
        self.assertDictEqual({"foo": 8080}, trainer.port_map)
        self.assertEqual(resource, trainer.resource)
        self.assertEqual(2, trainer.num_replicas)
        self.assertEqual(5, trainer.max_retries)
        self.assertEqual(RetryPolicy.REPLICA, trainer.retry_policy)

    def test_retry_policies(self) -> None:
        self.assertCountEqual(
            set(RetryPolicy),  # pyre-ignore[6]: Enum isn't iterable
            {
                RetryPolicy.APPLICATION,
                RetryPolicy.REPLICA,
                RetryPolicy.ROLE,
            },
        )

    def test_override_role(self) -> None:
        default = Role(
            "foobar",
            "torch",
            overrides={"image": lambda: "base", "entrypoint": lambda: "nentry"},
        )
        self.assertEqual("base", default.image)
        self.assertEqual("nentry", default.entrypoint)

    def test_async_override_role(self) -> None:
        async def update(value: str, time_seconds: int) -> str:
            await asyncio.sleep(time_seconds)
            return value

        default = Role(
            "foobar",
            "torch",
            overrides={"image": update("base", 1), "entrypoint": update("nentry", 2)},
        )
        self.assertEqual("base", default.image)
        self.assertEqual("nentry", default.entrypoint)

    def test_override_role_resolved_in_place(self) -> None:
        calls = 0

        def produce() -> str:
            nonlocal calls
            calls += 1
            return "base"

        default = Role(
            "foobar",
            "torch",
            overrides={"image": produce},
        )
        self.assertEqual("base", default.image)
        self.assertEqual("base", default.image, "the resolved value must persist")
        self.assertEqual(1, calls, "the producer must run exactly once")
        self.assertIn(
            "image",
            default.overrides,
            "resolution writes back in place — the key must stay in `overrides`",
        )

    def test_override_role_single_flight_under_concurrency(self) -> None:
        calls = 0

        def produce() -> str:
            nonlocal calls
            calls += 1
            time.sleep(0.05)  # widen the race window
            return "base"

        role = Role("foobar", "torch", overrides={"image": produce})
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: role.image, range(8)))
        self.assertEqual(["base"] * 8, results)
        self.assertEqual(
            1, calls, "concurrent readers must resolve the override exactly once"
        )

    def test_override_single_flight_across_roles_sharing_dict(self) -> None:
        calls = 0

        def produce() -> str:
            nonlocal calls
            calls += 1
            time.sleep(0.05)  # widen the race window
            return "base"

        overrides: dict[str, object] = {"image": produce}
        role_a = Role("a", "torch", overrides=overrides)
        role_b = Role("b", "torch", overrides=overrides)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda i: (role_a if i % 2 else role_b).image, range(8))
            )
        self.assertEqual(["base"] * 8, results)
        self.assertEqual(
            1,
            calls,
            "roles sharing an overrides dict must share its single-flight lock",
        )

    def test_override_resolution_not_serialized_across_dicts(self) -> None:
        producer_entered: threading.Event = threading.Event()
        release_producer: threading.Event = threading.Event()

        def slow_produce() -> str:
            producer_entered.set()
            assert release_producer.wait(timeout=30), "watchdog: test never released"
            return "slow"

        slow_role = Role("slow", "torch", overrides={"image": slow_produce})
        fast_role = Role("fast", "torch", overrides={"image": lambda: "fast"})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            slow_read = pool.submit(lambda: slow_role.image)
            self.assertTrue(
                producer_entered.wait(timeout=30),
                "watchdog: slow producer never started",
            )
            try:
                # bounded read: under a process-wide resolution lock this
                # blocks behind the (still-running) slow producer and times out
                fast_read = pool.submit(lambda: fast_role.image)
                self.assertEqual(
                    "fast",
                    fast_read.result(timeout=10),
                    "one dict's slow producer must not stall reads on another",
                )
            finally:
                release_producer.set()
            self.assertEqual("slow", slow_read.result(timeout=30))

    def test_override_producer_rereads_same_role(self) -> None:
        overrides: dict[str, object] = {"entrypoint": lambda: "nentry"}
        role = Role("foobar", "torch", overrides=overrides)
        # the producer reads another overridden attr on the same role,
        # re-entering the same dict's resolution lock on its own thread
        overrides["image"] = lambda: f"{role.entrypoint}-img"
        # daemon thread (not a pool): on deadlock the join times out and the
        # test fails instead of hanging the process on worker shutdown
        result: list[str] = []
        reader = threading.Thread(target=lambda: result.append(role.image))
        reader.daemon = True
        reader.start()
        reader.join(timeout=30)
        self.assertEqual(
            ["nentry-img"],
            result,
            "a producer reading an overridden attr on the same role must"
            " not self-deadlock (resolution lock is re-entrant)",
        )
        self.assertEqual("nentry", role.entrypoint)

    def test_resolved_override_dict_entry_remains_callable(self) -> None:
        role = Role("foobar", "torch", overrides={"image": lambda: "base"})
        self.assertEqual("base", role.image)  # resolves + writes back
        self.assertEqual(
            "base",
            role.overrides["image"](),
            "raw-dict consumers invoke the memoized entry post-resolution"
            " (legacy callable contract)",
        )

    def test_override_resolved_read_skips_resolution_lock(self) -> None:
        role = Role("foobar", "torch", overrides={"image": lambda: "base"})
        self.assertEqual("base", role.image)  # resolves + mints the dict's lock

        class FailingLock:
            def __enter__(self) -> None:
                raise AssertionError(
                    "a resolved override read must not take the resolution lock"
                )

            def __exit__(self, *exc: object) -> None:
                pass

        role.overrides[_OVERRIDES_LOCK_KEY].lock = FailingLock()
        self.assertEqual("base", role.image, "fast path must serve resolved values")
        self.assertEqual("foobar", role.name, "non-overridden reads must not lock")

    def test_role_with_resolved_overrides_deepcopies(self) -> None:
        role = Role("foobar", "torch", overrides={"image": lambda: "base"})
        self.assertEqual("base", role.image)  # resolves + mints the dict's lock
        clone = copy.deepcopy(role)
        self.assertEqual(
            "base",
            clone.image,
            "a role with a minted resolution lock must stay deep-copyable",
        )

    def test_async_override_role_inside_running_loop(self) -> None:
        async def update(value: str) -> str:
            await asyncio.sleep(0)
            return value

        async def resolve_image() -> str:
            role = Role(
                "foobar",
                "torch",
                overrides={"image": update("base")},
            )
            return role.image

        self.assertEqual(
            "base",
            asyncio.run(resolve_image()),
            "awaitable overrides must resolve when accessed inside a running event loop",
        )

    def test_override_role_failed_resolution_keeps_override(self) -> None:
        def boom() -> str:
            raise RuntimeError("boom")

        role = Role("foobar", "torch", overrides={"image": boom})
        with self.assertRaisesRegex(RuntimeError, "boom"):
            _ = role.image
        self.assertIs(
            boom,
            role.overrides.get("image"),
            "a failed resolution must leave the producer in `overrides` for retry",
        )

    def test_override_role_future_bound_to_other_running_loop(self) -> None:
        owner_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        thread.start()
        try:

            async def make_future() -> "asyncio.Future[str]":
                loop = asyncio.get_running_loop()
                fut: "asyncio.Future[str]" = loop.create_future()
                loop.call_later(0.05, fut.set_result, "base")
                return fut

            fut: "asyncio.Future[str]" = asyncio.run_coroutine_threadsafe(
                make_future(), owner_loop
            ).result()

            async def resolve_image() -> str:
                role = Role("foobar", "torch", overrides={"image": fut})
                return role.image

            self.assertEqual(
                "base",
                asyncio.run(resolve_image()),
                "a future bound to another thread's running loop must resolve"
                " when accessed inside this thread's running loop",
            )
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            thread.join(timeout=5)
            owner_loop.close()

    def test_override_role_future_bound_to_other_running_loop_no_local_loop(
        self,
    ) -> None:
        owner_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
        thread.start()
        try:

            async def make_future() -> "asyncio.Future[str]":
                loop = asyncio.get_running_loop()
                fut: "asyncio.Future[str]" = loop.create_future()
                loop.call_later(0.05, fut.set_result, "base")
                return fut

            fut: "asyncio.Future[str]" = asyncio.run_coroutine_threadsafe(
                make_future(), owner_loop
            ).result()
            role = Role("foobar", "torch", overrides={"image": fut})
            self.assertEqual(
                "base",
                role.image,
                "a future bound to another thread's running loop must resolve"
                " from a thread with no loop of its own",
            )
        finally:
            owner_loop.call_soon_threadsafe(owner_loop.stop)
            thread.join(timeout=5)
            owner_loop.close()

    def test_override_role_future_bound_to_current_loop_raises(self) -> None:
        async def resolve_image() -> None:
            fut: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
            role = Role("foobar", "torch", overrides={"image": fut})
            with self.assertRaisesRegex(
                RuntimeError, "currently running in this thread"
            ):
                _ = role.image
            self.assertIs(
                fut,
                role.overrides.get("image"),
                "an unresolvable loop-bound override must stay in `overrides`"
                " so the caller can still `await` it",
            )

        asyncio.run(resolve_image())

    def test_concurrent_override_role(self) -> None:

        def delay(value: tuple[str, str], time_seconds: int) -> tuple[str, str]:
            time.sleep(time_seconds)
            return value

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launcher_image_future: concurrent.futures.Future = executor.submit(
                delay, ("value1", "value2"), 2
            )

        def get_image() -> str:
            concurrent.futures.wait([launcher_image_future], 3)
            return launcher_image_future.result()[0]

        def get_entrypoint() -> str:
            concurrent.futures.wait([launcher_image_future], 3)
            return launcher_image_future.result()[1]

        default = Role(
            "foobar",
            "torch",
            overrides={"image": get_image, "entrypoint": get_entrypoint},
        )
        self.assertEqual("value1", default.image)
        self.assertEqual("value2", default.entrypoint)


class AppHandleTest(unittest.TestCase):
    def test_parse_malformed_app_handles(self) -> None:
        bad_app_handles = {
            "my_session/my_application_id": "missing scheduler backend",
            "local://my_session/": "missing app_id",
            "local://my_application_id": "missing session",
        }

        for handle, msg in bad_app_handles.items():
            with self.subTest(f"malformed app handle: {msg}", handle=handle):
                with self.assertRaises(MalformedAppHandleException):
                    parse_app_handle(handle)

    def test_parse_app_handle_empty_session_name(self) -> None:
        # missing session name is not OK but an empty one is
        app_handle = "local:///my_application_id"
        handle = parse_app_handle(app_handle)

        self.assertEqual(handle.app_id, "my_application_id")
        self.assertEqual("local", handle.scheduler_backend)
        self.assertEqual("", handle.session_name)

    def test_parse(self) -> None:
        scheduler_backend, session_name, app_id = parse_app_handle(
            "local://my_session/my_app_id_1234"
        )
        self.assertEqual("local", scheduler_backend)
        self.assertEqual("my_session", session_name)
        self.assertEqual("my_app_id_1234", app_id)


class AppDefTest(unittest.TestCase):
    def test_application(self) -> None:
        trainer = Role(
            "trainer",
            "test_image",
            entrypoint="/bin/sleep",
            args=["10"],
            num_replicas=2,
        )
        app = AppDef(name="test_app", roles=[trainer])
        self.assertEqual("test_app", app.name)
        self.assertEqual(1, len(app.roles))
        self.assertEqual(trainer, app.roles[0])

    def test_application_default(self) -> None:
        app = AppDef(name="test_app")
        self.assertEqual(0, len(app.roles))

    def test_getset_metadata(self) -> None:
        app = AppDef(name="test_app", metadata={"test_key": "test_value"})
        self.assertEqual("test_value", app.metadata["test_key"])
        self.assertEqual(None, app.metadata.get("non_existent"))


class RunConfigTest(unittest.TestCase):
    def get_cfg(self) -> Mapping[str, CfgVal]:
        return {
            "run_as": "root",
            "cluster_id": 123,
            "priority": 0.5,
            "preemptible": True,
        }

    def test_valid_values(self) -> None:
        cfg = self.get_cfg()

        self.assertEqual("root", cfg.get("run_as"))
        self.assertEqual(123, cfg.get("cluster_id"))
        self.assertEqual(0.5, cfg.get("priority"))
        self.assertTrue(cfg.get("preemptible"))
        self.assertIsNone(cfg.get("unknown"))

    def test_runopt_cast_to_type_bool_vocabulary(self) -> None:
        opt = runopt(default=False, opt_type=bool, is_required=False, help="help")
        for literal in ("true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"):
            self.assertTrue(opt.cast_to_type(literal), f"`{literal}` must cast to True")
        for literal in ("false", "False", "FALSE", "0", "no", "No", "off", "OFF"):
            self.assertFalse(
                opt.cast_to_type(literal), f"`{literal}` must cast to False"
            )

    def test_runopt_cast_to_type_bool_rejects_garbage(self) -> None:
        opt = runopt(default=False, opt_type=bool, is_required=False, help="help")
        with self.assertRaisesRegex(ValueError, "garbage"):
            opt.cast_to_type("garbage")

    def test_runopt_cast_to_type_typing_list(self) -> None:
        opt = runopt(default="", opt_type=List[str], is_required=False, help="help")
        self.assertEqual(["a", "b", "c"], opt.cast_to_type("a,b,c"))
        self.assertEqual(["abc", "def", "ghi"], opt.cast_to_type("abc;def;ghi"))

    def test_runopt_cast_to_type_builtin_list(self) -> None:
        opt = runopt(default="", opt_type=list[str], is_required=False, help="help")
        self.assertEqual(["a", "b", "c"], opt.cast_to_type("a,b,c"))
        self.assertEqual(["abc", "def", "ghi"], opt.cast_to_type("abc;def;ghi"))

    def test_runopts_add(self) -> None:
        """
        tests for various add option variations
        does not assert anything, a successful test
        should not raise any unexpected errors
        """
        opts = runopts()
        opts.add("run_as", type_=str, help="run as user")
        opts.add("run_as_default", type_=str, help="run as user", default="root")
        opts.add("run_as_required", type_=str, help="run as user", required=True)

        with self.assertRaises(ValueError):
            opts.add(
                "run_as", type_=str, help="run as user", default="root", required=True
            )

        opts.add("priority", type_=int, help="job priority", default=10)

        with self.assertRaises(TypeError):
            opts.add("priority", type_=int, help="job priority", default=0.5)

        # this print is intentional (demonstrates the intended usecase)
        print(opts)

    def get_runopts(self) -> runopts:
        opts = runopts()
        opts.add("run_as", type_=str, help="run as user", required=True)
        opts.add("priority", type_=int, help="job priority", default=10)
        opts.add("cluster_id", type_=str, help="cluster to submit job")
        return opts

    def test_runopts_resolve_minimal(self) -> None:
        opts = self.get_runopts()
        cfg = {"run_as": "foobar"}

        resolved = opts.resolve(cfg)
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(10, resolved.get("priority"))
        self.assertIsNone(resolved.get("cluster_id"))

        # make sure original config is untouched
        self.assertEqual("foobar", cfg.get("run_as"))
        self.assertIsNone(cfg.get("priority"))
        self.assertIsNone(cfg.get("cluster_id"))

    def test_runopts_resolve_override(self) -> None:
        opts = self.get_runopts()
        cfg = {
            "run_as": "foobar",
            "priority": 20,
            "cluster_id": "test_cluster",
        }

        resolved = opts.resolve(cfg)
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(20, resolved.get("priority"))
        self.assertEqual("test_cluster", resolved.get("cluster_id"))

    def test_runopts_resolve_missing_required(self) -> None:
        opts = self.get_runopts()

        cfg = {
            "priority": 20,
            "cluster_id": "test_cluster",
        }

        with self.assertRaises(InvalidRunConfigException):
            opts.resolve(cfg)

    def test_runopts_resolve_bad_type(self) -> None:
        opts = self.get_runopts()

        cfg = {
            "run_as": "foobar",
            "cluster_id": 123,
        }

        with self.assertRaises(InvalidRunConfigException):
            opts.resolve(cfg)

    def test_runopts_resolve_unioned(self) -> None:
        # runconfigs is a union of all run opts for all schedulers
        # make sure  opts resolves run configs that have more
        # configs than it knows about
        opts = self.get_runopts()
        cfg = {
            "run_as": "foobar",
            "some_other_opt": "baz",
        }

        resolved = opts.resolve(cfg)
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(10, resolved.get("priority"))
        self.assertIsNone(resolved.get("cluster_id"))
        self.assertEqual("baz", resolved.get("some_other_opt"))

    def test_runopts_get_camelcase_fallback(self) -> None:
        """get() with a camelCase name falls back to the snake_case key."""
        opts = self.get_runopts()
        self.assertIsNotNone(opts.get("cluster_id"))
        self.assertIsNotNone(
            opts.get("clusterId"),
            "camelCase lookup should find snake_case key",
        )

    def test_runopts_resolve_camelcase_cfg(self) -> None:
        """resolve() accepts camelCase cfg keys for snake_case registered opts."""
        opts = self.get_runopts()
        resolved = opts.resolve({"runAs": "foobar"})
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(10, resolved.get("priority"), "default should be filled")

    def test_runopts_resolve_camelcase_canonicalized(self) -> None:
        """resolve() returns only the registered spelling, never the alias."""
        opts = self.get_runopts()
        resolved = opts.resolve({"runAs": "foobar", "clusterId": "c1"})
        self.assertNotIn("runAs", resolved)
        self.assertNotIn("clusterId", resolved)
        self.assertEqual("foobar", resolved["run_as"])
        self.assertEqual("c1", resolved["cluster_id"])

    def test_runopts_resolve_conflicting_spellings_raise(self) -> None:
        """resolve() raises when an opt is passed under two spellings with
        different values instead of silently preferring one."""
        opts = self.get_runopts()
        with self.assertRaisesRegex(InvalidRunConfigException, "run_as.*two spellings"):
            opts.resolve({"run_as": "alice", "runAs": "bob"})

    def test_runopts_resolve_equal_spellings_collapse(self) -> None:
        """resolve() collapses two spellings of an opt with equal values into
        the registered spelling."""
        opts = self.get_runopts()
        resolved = opts.resolve({"run_as": "alice", "runAs": "alice"})
        self.assertEqual("alice", resolved["run_as"])
        self.assertNotIn("runAs", resolved)

    def test_cfg_from_str_canonicalizes_camelcase(self) -> None:
        """cfg_from_str() keys the parsed value by the registered spelling."""
        opts = self.get_runopts()
        self.assertDictEqual({"run_as": "alice"}, opts.cfg_from_str("runAs=alice"))

    def test_cfg_from_str_conflicting_spellings_raise(self) -> None:
        """cfg_from_str() raises when an opt is passed under two spellings
        with different values."""
        opts = self.get_runopts()
        with self.assertRaisesRegex(InvalidRunConfigException, "run_as.*two spellings"):
            opts.cfg_from_str("run_as=alice,runAs=bob")

    def test_cfg_from_json_repr_canonicalizes_camelcase(self) -> None:
        """cfg_from_json_repr() keys the parsed value by the registered spelling."""
        opts = self.get_runopts()
        self.assertDictEqual(
            {"run_as": "alice"}, opts.cfg_from_json_repr('{"runAs": "alice"}')
        )

    def test_cfg_from_json_repr_conflicting_spellings_raise(self) -> None:
        """cfg_from_json_repr() raises when an opt is passed under two
        spellings with different values."""
        opts = self.get_runopts()
        with self.assertRaisesRegex(InvalidRunConfigException, "run_as.*two spellings"):
            opts.cfg_from_json_repr('{"run_as": "alice", "runAs": "bob"}')

    def test_cfg_from_str(self) -> None:
        opts = runopts()
        opts.add("K", type_=List[str], help="a list opt", default=[])
        opts.add("J", type_=str, help="a str opt", required=True)
        opts.add("E", type_=Dict[str, str], help="a dict opt", default=[])

        self.assertDictEqual({}, opts.cfg_from_str(""))
        self.assertDictEqual({}, opts.cfg_from_str("UNKWN=b"))
        self.assertDictEqual({"K": ["a"], "J": "b"}, opts.cfg_from_str("K=a,J=b"))
        self.assertDictEqual({"K": ["a"]}, opts.cfg_from_str("K=a,UNKWN=b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a,b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a;b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a,b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a,b;"))
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a,b,J=d")
        )
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a,b;J=d")
        )
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a;b,J=d")
        )
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a;b;J=d")
        )
        self.assertDictEqual(
            {"K": ["a"], "J": "d"}, opts.cfg_from_str("J=d,K=a,UNKWN=e")
        )
        self.assertDictEqual(
            {"E": {"f": "b", "F": "B"}}, opts.cfg_from_str("E=f:b,F:B")
        )

    def test_cfg_from_str_builtin_generic_types(self) -> None:
        # basically a repeat of "test_cfg_from_str()" but with
        # list[str] and dict[str, str] instead of List[str] and Dict[str, str]
        opts = runopts()
        opts.add("K", type_=list[str], help="a list opt", default=[])
        opts.add("J", type_=str, help="a str opt", required=True)
        opts.add("E", type_=dict[str, str], help="a dict opt", default=[])

        self.assertDictEqual({}, opts.cfg_from_str(""))
        self.assertDictEqual({}, opts.cfg_from_str("UNKWN=b"))
        self.assertDictEqual({"K": ["a"], "J": "b"}, opts.cfg_from_str("K=a,J=b"))
        self.assertDictEqual({"K": ["a"]}, opts.cfg_from_str("K=a,UNKWN=b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a,b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a;b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a,b"))
        self.assertDictEqual({"K": ["a", "b"]}, opts.cfg_from_str("K=a,b;"))
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a,b,J=d")
        )
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a,b;J=d")
        )
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a;b,J=d")
        )
        self.assertDictEqual(
            {"K": ["a", "b"], "J": "d"}, opts.cfg_from_str("K=a;b;J=d")
        )
        self.assertDictEqual(
            {"K": ["a"], "J": "d"}, opts.cfg_from_str("J=d,K=a,UNKWN=e")
        )
        self.assertDictEqual(
            {"E": {"f": "b", "F": "B"}}, opts.cfg_from_str("E=f:b,F:B")
        )

    def test_resolve_from_str(self) -> None:
        opts = runopts()
        opts.add("foo", type_=str, default="", help="")
        opts.add("test_key", type_=str, default="", help="")
        opts.add("default_time", type_=int, default=0, help="")
        opts.add("enable", type_=bool, default=True, help="")
        opts.add("disable", type_=bool, default=True, help="")
        opts.add("complex_list", type_=List[str], default=[], help="")

        self.assertDictEqual(
            {
                "foo": "bar",
                "test_key": "test_value",
                "default_time": 42,
                "enable": True,
                "disable": False,
                "complex_list": ["v1", "v2", "v3"],
            },
            opts.resolve(
                opts.cfg_from_str(
                    "foo=bar,test_key=test_value,default_time=42,enable=True,disable=False,complex_list=v1;v2;v3"
                )
            ),
        )

    def test_config_from_json_repr(self) -> None:
        opts = runopts()
        opts.add("foo", type_=str, default="", help="")
        opts.add("test_key", type_=str, default="", help="")
        opts.add("default_time", type_=int, default=0, help="")
        opts.add("enable", type_=bool, default=True, help="")
        opts.add("disable", type_=bool, default=True, help="")
        opts.add("complex_list", type_=List[str], default=[], help="")
        opts.add("complex_dict", type_=Dict[str, str], default={}, help="")
        opts.add("default_none", type_=List[str], help="")

        self.assertDictEqual(
            {
                "foo": "bar",
                "test_key": "test_value",
                "default_time": 42,
                "enable": True,
                "disable": False,
                "complex_list": ["v1", "v2", "v3"],
                "complex_dict": {"k1": "v1", "k2": "v2"},
                "default_none": None,
            },
            opts.resolve(
                opts.cfg_from_json_repr(
                    """{
                        "foo": "bar",
                        "test_key": "test_value",
                        "default_time": 42,
                        "enable": true,
                        "disable": false,
                        "complex_list": ["v1", "v2", "v3"],
                        "complex_dict": {"k1": "v1", "k2": "v2"},
                        "default_none": null
                    }"""
                )
            ),
        )

    def test_runopts_is_type(self) -> None:
        # primitive types
        self.assertTrue(runopts.is_type(3, int))
        self.assertFalse(runopts.is_type("foo", int))
        # List[str]
        self.assertFalse(runopts.is_type(None, List[str]))
        self.assertTrue(runopts.is_type([], List[str]))
        self.assertTrue(runopts.is_type(["a", "b"], List[str]))
        # List[str]
        self.assertFalse(runopts.is_type(None, Dict[str, str]))
        self.assertTrue(runopts.is_type({}, Dict[str, str]))
        self.assertTrue(runopts.is_type({"foo": "bar", "fee": "bez"}, Dict[str, str]))

    def test_runopts_iter(self) -> None:
        runopts = self.get_runopts()
        for name, opt in runopts:
            self.assertEqual(opt, runopts.get(name))

    def test_runopts_or_merges_options(self) -> None:
        """Test that __or__ merges two runopts into a new runopts."""
        opts1 = runopts()
        opts1.add("foo", type_=str, default="a", help="foo option")
        opts1.add("bar", type_=int, default=1, help="bar option")

        opts2 = runopts()
        opts2.add("baz", type_=bool, default=True, help="baz option")

        merged = opts1 | opts2

        # Original runopts should be unchanged
        self.assertIsNone(opts1.get("baz"))
        self.assertIsNone(opts2.get("foo"))

        # Merged should have all options
        self.assertIsNotNone(merged.get("foo"))
        self.assertIsNotNone(merged.get("bar"))
        self.assertIsNotNone(merged.get("baz"))
        self.assertEqual(sorted([key for key, _ in merged]), ["bar", "baz", "foo"])


class CasesTest(unittest.TestCase):
    def test_snake_to_camel(self) -> None:
        self.assertEqual(cases.snake_to_camel("cluster_name"), "clusterName")
        self.assertEqual(cases.snake_to_camel("num_retries"), "numRetries")
        self.assertEqual(cases.snake_to_camel("hpc_cluster_uuid"), "hpcClusterUuid")
        self.assertEqual(cases.snake_to_camel("name"), "name")

    def test_camel_to_snake(self) -> None:
        self.assertEqual(cases.camel_to_snake("clusterName"), "cluster_name")
        self.assertEqual(cases.camel_to_snake("numRetries"), "num_retries")
        self.assertEqual(cases.camel_to_snake("hpcClusterUuid"), "hpc_cluster_uuid")
        self.assertEqual(cases.camel_to_snake("name"), "name")

    def test_roundtrip(self) -> None:
        """snake → camel → snake preserves the original."""
        for name in ["cluster_name", "num_retries", "hpc_cluster_uuid", "name"]:
            self.assertEqual(
                cases.camel_to_snake(cases.snake_to_camel(name)),
                name,
                f"roundtrip failed for `{name}`",
            )


class GetTypeNameTest(unittest.TestCase):
    def test_get_type_name(self) -> None:
        self.assertEqual("int", get_type_name(int))
        self.assertEqual("list", get_type_name(list))
        self.assertEqual("typing.Union[str, int]", get_type_name(Union[str, int]))
        # pyrefly: ignore [bad-argument-type]
        self.assertEqual("typing.List[int]", get_type_name(List[int]))
        # pyrefly: ignore [bad-argument-type]
        self.assertEqual("typing.Dict[str, int]", get_type_name(Dict[str, int]))
        self.assertEqual(
            # pyrefly: ignore [bad-argument-type]
            "typing.List[typing.List[int]]",
            # pyrefly: ignore [bad-argument-type]
            get_type_name(List[List[int]]),
        )


class MacrosTest(unittest.TestCase):
    def test_substitute(self) -> None:
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            rank0_env="rank0_env",
        )
        for key, val in asdict(v).items():
            template = f"tmpl-{getattr(macros, key)}"
            self.assertEqual(v.substitute(template), f"tmpl-{val}")

    def test_apply(self) -> None:
        role = Role(
            name="test",
            image="test_image",
            entrypoint="foo.py",
            args=[macros.img_root],
            env={"FOO": macros.app_id},
        )
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            rank0_env="rank0_env",
        )
        newrole = v.apply(role)
        self.assertNotEqual(newrole, role)
        self.assertEqual(newrole.args, ["img_root"])
        self.assertEqual(newrole.env, {"FOO": "app_id"})

    def test_apply_preserves_role_overrides(self) -> None:
        overrides = {"entrypoint": lambda: "lazy.py"}
        role = Role(
            name="test",
            image="test_image",
            args=[macros.img_root],
            overrides=overrides,
        )
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            rank0_env="rank0_env",
        )
        newrole = v.apply(role)
        self.assertIs(
            role.overrides,
            overrides,
            "apply() must restore the caller role's overrides",
        )
        self.assertIs(
            newrole.overrides,
            overrides,
            "the copy shares the dict — write-back resolution makes either"
            " owner's resolution visible to both",
        )
        self.assertEqual("lazy.py", newrole.entrypoint)
        self.assertEqual(
            "lazy.py",
            role.entrypoint,
            "resolving on the copy must resolve for the original too",
        )
        self.assertEqual(newrole.args, ["img_root"])

    def test_apply_without_overrides_gives_copy_its_own_dict(self) -> None:
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            rank0_env="rank0_env",
        )

        role = Role(name="test", image="test_image")
        with self.assertNoLogs("torchx.specs.api", level="DEBUG"):
            newrole = v.apply(role)
        self.assertIsNot(
            newrole.overrides,
            role.overrides,
            "with nothing to share the copy must get its own overrides dict",
        )

        # the reserved lock-key entry alone must not read as "has overrides"
        locked = Role(name="test", image="test_image")
        locked.overrides[_OVERRIDES_LOCK_KEY] = _OverridesLock()
        with self.assertNoLogs("torchx.specs.api", level="DEBUG"):
            newlocked = v.apply(locked)
        self.assertIsNot(
            newlocked.overrides,
            locked.overrides,
            "a lock-key-only dict counts as empty — no sharing",
        )

    def test_apply_nested_with_list_of_dicts(self) -> None:
        """Test that _apply_nested correctly handles dictionaries nested inside lists."""
        role = Role(
            name="test",
            image="test_image",
            entrypoint="foo.py",
            metadata={
                "nested_list": [
                    {"key1": macros.app_id, "key2": "static"},
                    {"key3": macros.img_root},
                ]
            },
        )
        v = macros.Values(
            img_root="img_root_value",
            app_id="app_id_value",
            replica_id="replica_id_value",
            rank0_env="rank0_env_value",
        )
        newrole = v.apply(role)
        self.assertEqual(newrole.metadata["nested_list"][0]["key1"], "app_id_value")
        self.assertEqual(newrole.metadata["nested_list"][0]["key2"], "static")
        self.assertEqual(newrole.metadata["nested_list"][1]["key3"], "img_root_value")

    def test_apply_nested_with_deeply_nested_structures(self) -> None:
        """Test that _apply_nested handles deeply nested structures with mixed types."""
        role = Role(
            name="test",
            image="test_image",
            entrypoint="foo.py",
            metadata={
                "level1": {
                    "level2": {
                        "list_with_dicts": [
                            {
                                "nested_key": macros.replica_id,
                                "nested_list": [macros.app_id, "static_value"],
                            },
                            {"another_key": macros.img_root},
                        ],
                        "simple_string": macros.rank0_env,
                    }
                }
            },
        )
        v = macros.Values(
            img_root="img_root_value",
            app_id="app_id_value",
            replica_id="replica_id_value",
            rank0_env="rank0_env_value",
        )
        newrole = v.apply(role)

        # Check deeply nested dict in list
        nested_dict = newrole.metadata["level1"]["level2"]["list_with_dicts"][0]
        self.assertEqual(nested_dict["nested_key"], "replica_id_value")
        self.assertEqual(nested_dict["nested_list"][0], "app_id_value")
        self.assertEqual(nested_dict["nested_list"][1], "static_value")

        # Check second dict in list
        second_dict = newrole.metadata["level1"]["level2"]["list_with_dicts"][1]
        self.assertEqual(second_dict["another_key"], "img_root_value")

        # Check simple string at nested level
        self.assertEqual(
            newrole.metadata["level1"]["level2"]["simple_string"], "rank0_env_value"
        )

    def test_apply_nested_with_list_of_strings(self) -> None:
        """Test that _apply_nested still works correctly with lists of strings."""
        role = Role(
            name="test",
            image="test_image",
            entrypoint="foo.py",
            metadata={
                "string_list": [macros.app_id, macros.img_root, "static"],
            },
        )
        v = macros.Values(
            img_root="img_root_value",
            app_id="app_id_value",
            replica_id="replica_id_value",
            rank0_env="rank0_env_value",
        )
        newrole = v.apply(role)
        self.assertEqual(newrole.metadata["string_list"][0], "app_id_value")
        self.assertEqual(newrole.metadata["string_list"][1], "img_root_value")
        self.assertEqual(newrole.metadata["string_list"][2], "static")

    def test_apply_nested_with_mixed_list_types(self) -> None:
        """Test that _apply_nested handles lists with mixed types (strings, dicts, other)."""
        role = Role(
            name="test",
            image="test_image",
            entrypoint="foo.py",
            metadata={
                "mixed_list": [
                    macros.app_id,
                    {"nested": macros.img_root},
                    42,  # non-string, non-dict value
                    "static_string",
                ],
            },
        )
        v = macros.Values(
            img_root="img_root_value",
            app_id="app_id_value",
            replica_id="replica_id_value",
            rank0_env="rank0_env_value",
        )
        newrole = v.apply(role)
        self.assertEqual(newrole.metadata["mixed_list"][0], "app_id_value")
        self.assertEqual(newrole.metadata["mixed_list"][1]["nested"], "img_root_value")
        self.assertEqual(newrole.metadata["mixed_list"][2], 42)
        self.assertEqual(newrole.metadata["mixed_list"][3], "static_string")
