#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import concurrent
import os
import tempfile
import time
import unittest
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union
from unittest import mock
from unittest.mock import MagicMock

from torchx.specs import named_resources, named_resources_aws, resource
from torchx.specs.api import (
    _TERMINAL_STATES,
    AppDef,
    AppDryRunInfo,
    AppState,
    AppStatus,
    AppStatusError,
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
                            },
                            {
                                "id": 1,
                                "state": 3,
                                "role": "worker",
                                "hostname": "localhost",
                                "structured_error_msg": "<NONE>",
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

    def test_concurrent_override_role(self) -> None:

        def delay(value: Tuple[str, str], time_seconds: int) -> Tuple[str, str]:
            time.sleep(time_seconds)
            return value

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            launcher_fbpkg_future: concurrent.futures.Future = executor.submit(
                delay, ("value1", "value2"), 2
            )

        def get_image() -> str:
            concurrent.futures.wait([launcher_fbpkg_future], 3)
            return launcher_fbpkg_future.result()[0]

        def get_entrypoint() -> str:
            concurrent.futures.wait([launcher_fbpkg_future], 3)
            return launcher_fbpkg_future.result()[1]

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
        (scheduler_backend, session_name, app_id) = parse_app_handle(
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

    def test_runopts_add_with_aliases(self) -> None:
        opts = runopts()
        opts.add(
            "job_priority",
            aliases=["jobPriority"],
            type_=str,
            help="priority for the job",
        )
        self.assertEqual(1, len(opts._opts))
        self.assertIsNotNone(opts.get("job_priority"))
        self.assertIsNotNone(opts.get("jobPriority"))

    def test_runopts_resolve_with_aliases(self) -> None:
        opts = runopts()
        opts.add(
            "job_priority",
            aliases=["jobPriority"],
            type_=str,
            help="priority for the job",
        )
        opts.resolve({"job_priority": "high"})
        opts.resolve({"jobPriority": "low"})
        with self.assertRaises(InvalidRunConfigException):
            opts.resolve({"job_priority": "high", "jobPriority": "low"})

    def test_runopts_resolve_with_none_valued_aliases(self) -> None:
        opts = runopts()
        opts.add(
            "job_priority",
            aliases=["jobPriority"],
            type_=str,
            help="priority for the job",
        )
        opts.add(
            "model_type_name",
            aliases=["modelTypeName"],
            type_=Union[str, None],
            help="ML Hub Model Type to attribute resource utilization for job",
        )
        resolved_opts = opts.resolve({"modelTypeName": None, "jobPriority": "low"})
        self.assertEqual(resolved_opts.get("modelTypeName"), None)
        self.assertEqual(resolved_opts.get("jobPriority"), "low")
        self.assertEqual(resolved_opts, {"modelTypeName": None, "jobPriority": "low"})

        with self.assertRaises(InvalidRunConfigException):
            opts.resolve({"modelTypeName": None, "model_type_name": "low"})

    def test_runopts_add_with_deprecated_aliases(self) -> None:
        opts = runopts()
        opts.add(
            "job_priority",
            deprecated_aliases=["priority"],
            type_=str,
            help="run as user",
        )

        opts.resolve({"job_priority": "high"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opts.resolve({"priority": "high"})
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0].category, UserWarning)
            self.assertEqual(
                str(w[0].message),
                "Run option `priority` is deprecated, use `job_priority` instead",
            )

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


class GetTypeNameTest(unittest.TestCase):
    def test_get_type_name(self) -> None:
        self.assertEqual("int", get_type_name(int))
        self.assertEqual("list", get_type_name(list))
        self.assertEqual("typing.Union[str, int]", get_type_name(Union[str, int]))
        self.assertEqual("typing.List[int]", get_type_name(List[int]))
        self.assertEqual("typing.Dict[str, int]", get_type_name(Dict[str, int]))
        self.assertEqual(
            "typing.List[typing.List[int]]", get_type_name(List[List[int]])
        )


class MacrosTest(unittest.TestCase):
    def test_substitute(self) -> None:
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            base_img_root="base_img_root",
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
            base_img_root="base_img_root",
            rank0_env="rank0_env",
        )
        newrole = v.apply(role)
        self.assertNotEqual(newrole, role)
        self.assertEqual(newrole.args, ["img_root"])
        self.assertEqual(newrole.env, {"FOO": "app_id"})
