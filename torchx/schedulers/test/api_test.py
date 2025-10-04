#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from datetime import datetime
from enum import Enum
from typing import Iterable, List, Mapping, Optional, TypeVar, Union
from unittest.mock import MagicMock, patch

from torchx.schedulers.api import (
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    split_lines,
    split_lines_iterator,
    Stream,
)
from torchx.specs.api import (
    AppDef,
    AppDryRunInfo,
    CfgVal,
    InvalidRunConfigException,
    macros,
    NULL_RESOURCE,
    Resource,
    Role,
    runopts,
)
from torchx.workspace.api import WorkspaceMixin

T = TypeVar("T")
A = TypeVar("A")
D = TypeVar("D")


class EnumConfig(str, Enum):
    option1 = "option1"
    option2 = "option2"


class IntEnumConfig(int, Enum):
    option1 = 1
    option2 = 2


class SchedulerTest(unittest.TestCase):
    class MockScheduler(Scheduler[T, A, D], WorkspaceMixin[None]):
        def __init__(self, session_name: str) -> None:
            super().__init__("mock", session_name)

        def schedule(self, dryrun_info: AppDryRunInfo[None]) -> str:
            app = dryrun_info._app
            assert app is not None
            return app.name

        def _submit_dryrun(
            self,
            app: AppDef,
            cfg: Mapping[str, CfgVal],
        ) -> AppDryRunInfo[None]:
            return AppDryRunInfo(None, lambda t: "None")

        def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
            return None

        def _cancel_existing(self, app_id: str) -> None:
            pass

        def log_iter(
            self,
            app_id: str,
            role_name: str,
            k: int = 0,
            regex: Optional[str] = None,
            since: Optional[datetime] = None,
            until: Optional[datetime] = None,
            should_tail: bool = False,
            streams: Optional[Stream] = None,
        ) -> Iterable[str]:
            return iter([])

        def list(self) -> List[ListAppResponse]:
            return []

        def _run_opts(self) -> runopts:
            opts = runopts()
            opts.add("foo", type_=str, required=True, help="required option")
            opts.add(
                "bar",
                type_=EnumConfig,
                required=True,
                help=f"Test Enum Config {[m.name for m in EnumConfig]}",
                creator=lambda x: EnumConfig(x),
            ),
            opts.add(
                "ienum",
                type_=IntEnumConfig,
                required=False,
                help=f"Test Enum Config {[m.name for m in IntEnumConfig]}",
                creator=lambda x: IntEnumConfig(x),
            ),

            return opts

        def resolve_resource(self, resource: Union[str, Resource]) -> Resource:
            return NULL_RESOURCE

        def build_workspace_and_update_role(
            self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
        ) -> None:
            role.image = workspace

    def test_invalid_run_cfg(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()

        empty_cfg = {}
        with self.assertRaises(InvalidRunConfigException):
            scheduler_mock.submit(app_mock, empty_cfg)

        bad_type_cfg = {"foo": 100}
        with self.assertRaises(InvalidRunConfigException):
            scheduler_mock.submit(app_mock, bad_type_cfg)

        bad_type_cfg = {"foo": "here", "bar": "temp"}
        with self.assertRaises(InvalidRunConfigException):
            scheduler_mock.submit(app_mock, bad_type_cfg)

    def test_submit_workspace(self) -> None:
        role = Role(
            name="sleep",
            image="",
            entrypoint="foo.sh",
        )
        app = AppDef(name="test_app", roles=[role])

        scheduler_mock = SchedulerTest.MockScheduler("test_session")

        cfg = {"foo": "asdf", "bar": EnumConfig["option1"], "ienum": 1}
        scheduler_mock.submit(app, cfg, workspace="some_workspace")
        self.assertEqual(app.roles[0].image, "some_workspace")

    def test_metadata_macro_substitute(self) -> None:
        role = Role(
            name="sleep",
            image="",
            entrypoint="foo.sh",
            metadata={
                "bridge": {
                    "tier": "${app_id}",
                },
                "packages": ["foo", "package_${app_id}"],
            },
        )
        values = macros.Values(
            img_root="",
            app_id="test_app",
            replica_id=str(1),
            rank0_env="TORCHX_RANK0_HOST",
        )
        replica_role = values.apply(role)
        self.assertEqual(replica_role.metadata["bridge"]["tier"], "test_app")
        self.assertEqual(replica_role.metadata["packages"], ["foo", "package_test_app"])

    def test_invalid_dryrun_cfg(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()

        with self.assertRaises(InvalidRunConfigException):
            empty_cfg = {}
            scheduler_mock.submit_dryrun(app_mock, empty_cfg)

        with self.assertRaises(InvalidRunConfigException):
            bad_type_cfg = {"foo": 100}
            scheduler_mock.submit_dryrun(app_mock, bad_type_cfg)

    def test_role_preproc_called(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = AppDef(name="test")
        app_mock.roles = [MagicMock()]

        cfg = {"foo": "bar", "bar": "option2"}
        scheduler_mock.submit_dryrun(app_mock, cfg)
        role_mock = app_mock.roles[0]
        role_mock.pre_proc.assert_called_once()

    def test_validate(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = AppDef(name="test")
        app_mock.roles = [MagicMock()]
        app_mock.roles[0].resource = NULL_RESOURCE

        with self.assertRaises(ValueError):
            scheduler_mock._validate(app_mock, "local", cfg={})

    def test_cancel_not_exists(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        with patch.object(scheduler_mock, "_cancel_existing") as cancel_mock:
            with patch.object(scheduler_mock, "exists") as exists_mock:
                exists_mock.return_value = True
                scheduler_mock.cancel("test_id")
                cancel_mock.assert_called_once()

    def test_cancel_exists(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        with patch.object(scheduler_mock, "_cancel_existing") as cancel_mock:
            with patch.object(scheduler_mock, "exists") as exists_mock:
                exists_mock.return_value = False
                scheduler_mock.cancel("test_id")
                cancel_mock.assert_not_called()

    def test_close_twice(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test")
        scheduler_mock.close()
        scheduler_mock.close()
        # nothing to validate explicitly, just that no errors are raised

    def test_split_lines(self) -> None:
        self.assertEqual(split_lines(""), [])
        self.assertEqual(split_lines("\n"), ["\n"])
        self.assertEqual(split_lines("foo\nbar"), ["foo\n", "bar"])
        self.assertEqual(split_lines("foo\nbar\n"), ["foo\n", "bar\n"])

    def test_split_lines_iterator(self) -> None:
        self.assertEqual(
            list(split_lines_iterator(["1\n2\n3\n4\n"])),
            [
                "1\n",
                "2\n",
                "3\n",
                "4\n",
            ],
        )
        self.assertEqual(
            list(split_lines_iterator(["foo\nbar", "foobar"])),
            [
                "foo\n",
                "bar",
                "foobar",
            ],
        )
