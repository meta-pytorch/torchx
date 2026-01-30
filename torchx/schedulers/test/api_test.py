#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, TypeVar
from unittest.mock import MagicMock, patch

from torchx.schedulers.api import (
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    split_lines,
    split_lines_iterator,
    Stream,
    StructuredOpts,
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


class SchedulerTest(unittest.TestCase):
    class MockScheduler(Scheduler[T], WorkspaceMixin[None]):
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

        def describe(self, app_id: str) -> DescribeAppResponse | None:
            return None

        def _cancel_existing(self, app_id: str) -> None:
            pass

        def log_iter(
            self,
            app_id: str,
            role_name: str,
            k: int = 0,
            regex: str | None = None,
            since: datetime | None = None,
            until: datetime | None = None,
            should_tail: bool = False,
            streams: Stream | None = None,
        ) -> Iterable[str]:
            return iter([])

        def list(
            self, cfg: Mapping[str, CfgVal] | None = None
        ) -> list[ListAppResponse]:
            return []

        def _run_opts(self) -> runopts:
            opts = runopts()
            opts.add("foo", type_=str, required=True, help="required option")
            return opts

        def resolve_resource(self, resource: str | Resource) -> Resource:
            return NULL_RESOURCE

        def build_workspace_and_update_role(
            self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
        ) -> None:
            role.image = workspace

    def test_invalid_run_cfg(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()

        with self.assertRaises(InvalidRunConfigException):
            empty_cfg = {}
            scheduler_mock.submit(app_mock, empty_cfg)

        with self.assertRaises(InvalidRunConfigException):
            bad_type_cfg = {"foo": 100}
            scheduler_mock.submit(app_mock, bad_type_cfg)

    def test_submit_workspace(self) -> None:
        role = Role(
            name="sleep",
            image="",
            entrypoint="foo.sh",
        )
        app = AppDef(name="test_app", roles=[role])

        scheduler_mock = SchedulerTest.MockScheduler("test_session")

        cfg = {"foo": "asdf"}
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

        cfg = {"foo": "bar"}
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


# =============================================================================
# Test dataclass for StructuredOpts tests
# =============================================================================


@dataclass
class SampleOpts(StructuredOpts):
    """Sample options for testing StructuredOpts base class."""

    cluster_name: str
    """Name of the cluster."""

    num_retries: int = 3
    """Number of retry attempts."""

    enable_debug: bool = False
    """Enable debug mode."""

    optional_tag: str | None = None
    """Optional tag for the job."""


class StructuredOptsTest(unittest.TestCase):
    """Tests for StructuredOpts base class functionality."""

    # -------------------------------------------------------------------------
    # snake_to_camel Tests
    # -------------------------------------------------------------------------

    def test_snake_to_camel(self) -> None:
        """Test snake_case to camelCase conversion."""
        self.assertEqual(StructuredOpts.snake_to_camel("cluster_name"), "clusterName")
        self.assertEqual(StructuredOpts.snake_to_camel("num_retries"), "numRetries")
        self.assertEqual(StructuredOpts.snake_to_camel("enable_debug"), "enableDebug")

    def test_snake_to_camel_single_word(self) -> None:
        """Test that single words are unchanged."""
        self.assertEqual(StructuredOpts.snake_to_camel("name"), "name")
        self.assertEqual(StructuredOpts.snake_to_camel("tags"), "tags")

    # -------------------------------------------------------------------------
    # from_cfg Tests
    # -------------------------------------------------------------------------

    def test_from_cfg_snake_case_keys(self) -> None:
        """Test from_cfg accepts snake_case keys."""
        cfg = {
            "cluster_name": "test_cluster",
            "num_retries": 5,
            "enable_debug": True,
        }
        opts = SampleOpts.from_cfg(cfg)

        self.assertEqual(opts.cluster_name, "test_cluster")
        self.assertEqual(opts.num_retries, 5)
        self.assertEqual(opts.enable_debug, True)

    def test_from_cfg_camel_case_keys(self) -> None:
        """Test from_cfg accepts camelCase keys as aliases."""
        cfg = {
            "clusterName": "test_cluster",
            "numRetries": 5,
            "enableDebug": True,
        }
        opts = SampleOpts.from_cfg(cfg)

        self.assertEqual(opts.cluster_name, "test_cluster")
        self.assertEqual(opts.num_retries, 5)
        self.assertEqual(opts.enable_debug, True)

    def test_from_cfg_snake_case_takes_precedence(self) -> None:
        """Test that snake_case keys take precedence over camelCase."""
        cfg = {
            "cluster_name": "snake_value",
            "clusterName": "camel_value",
        }
        opts = SampleOpts.from_cfg(cfg)

        self.assertEqual(opts.cluster_name, "snake_value")

    # -------------------------------------------------------------------------
    # Mapping Protocol Tests
    # -------------------------------------------------------------------------

    def test_get(self) -> None:
        """Test get() returns value or default."""
        opts = SampleOpts(cluster_name="test")

        self.assertEqual(opts.get("cluster_name"), "test")
        self.assertEqual(opts.get("num_retries"), 3)
        self.assertEqual(opts.get("nonexistent", "default"), "default")

    def test_get_returns_default_for_none(self) -> None:
        """Test get() returns default when value is None."""
        opts = SampleOpts(cluster_name="test", optional_tag=None)

        self.assertEqual(opts.get("optional_tag", "default"), "default")

    def test_getitem(self) -> None:
        """Test __getitem__ returns value or raises KeyError."""
        opts = SampleOpts(cluster_name="test")

        self.assertEqual(opts["cluster_name"], "test")
        with self.assertRaises(KeyError):
            _ = opts["nonexistent"]

    def test_len(self) -> None:
        """Test __len__ returns number of fields."""
        opts = SampleOpts(cluster_name="test")
        self.assertEqual(len(opts), 4)

    def test_iter(self) -> None:
        """Test __iter__ yields field names."""
        opts = SampleOpts(cluster_name="test")
        field_names = list(opts)

        self.assertEqual(
            field_names, ["cluster_name", "num_retries", "enable_debug", "optional_tag"]
        )

    def test_contains(self) -> None:
        """Test __contains__ checks for field existence."""
        opts = SampleOpts(cluster_name="test")

        self.assertIn("cluster_name", opts)
        self.assertIn("num_retries", opts)
        self.assertNotIn("nonexistent", opts)

    # -------------------------------------------------------------------------
    # as_runopts Tests
    # -------------------------------------------------------------------------

    def test_as_runopts_returns_runopts(self) -> None:
        """Test as_runopts returns a runopts instance."""
        opts = SampleOpts.as_runopts()
        self.assertIsInstance(opts, runopts)

    def test_as_runopts_includes_required_fields(self) -> None:
        """Test as_runopts includes required fields marked as required."""
        opts = SampleOpts.as_runopts()

        cluster_name_opt = opts.get("cluster_name")
        self.assertIsNotNone(cluster_name_opt)
        self.assertTrue(cluster_name_opt.is_required)

    def test_as_runopts_includes_optional_fields_with_defaults(self) -> None:
        """Test as_runopts includes optional fields with their defaults."""
        opts = SampleOpts.as_runopts()

        num_retries_opt = opts.get("num_retries")
        self.assertIsNotNone(num_retries_opt)
        self.assertFalse(num_retries_opt.is_required)
        self.assertEqual(num_retries_opt.default, 3)

        enable_debug_opt = opts.get("enable_debug")
        self.assertIsNotNone(enable_debug_opt)
        self.assertFalse(enable_debug_opt.is_required)
        self.assertEqual(enable_debug_opt.default, False)

    def test_as_runopts_generates_camelcase_aliases(self) -> None:
        """Test as_runopts generates camelCase aliases for snake_case fields."""
        opts = SampleOpts.as_runopts()

        # Check that camelCase alias resolves to the option
        cluster_name_opt = opts.get("clusterName")
        self.assertIsNotNone(cluster_name_opt)

        num_retries_opt = opts.get("numRetries")
        self.assertIsNotNone(num_retries_opt)

    def test_as_runopts_resolve_accepts_aliases_and_adds_defaults(self) -> None:
        """Test runopts.resolve() accepts alias keys and adds defaults."""
        opts = SampleOpts.as_runopts()

        # Pass cfg with camelCase keys (aliases)
        cfg_with_aliases = {
            "clusterName": "foo",
        }

        # resolve() should not raise - aliases are valid
        resolved_cfg = opts.resolve(cfg_with_aliases)

        # Default values are added with canonical (snake_case) keys
        self.assertIn("num_retries", resolved_cfg)
        self.assertEqual(resolved_cfg["num_retries"], 3)

        # SampleOpts.from_cfg() works with resolved cfg containing alias keys
        sample_opts = SampleOpts.from_cfg(resolved_cfg)
        self.assertEqual(sample_opts.cluster_name, "foo")
        self.assertEqual(sample_opts.num_retries, 3)

    # -------------------------------------------------------------------------
    # get_docstrings Tests
    # -------------------------------------------------------------------------

    def test_get_docstrings(self) -> None:
        """Test get_docstrings extracts field docstrings."""
        docstrings = SampleOpts.get_docstrings()

        self.assertIn("cluster_name", docstrings)
        self.assertEqual(docstrings["cluster_name"], "Name of the cluster.")

        self.assertIn("num_retries", docstrings)
        self.assertEqual(docstrings["num_retries"], "Number of retry attempts.")

    # -------------------------------------------------------------------------
    # __or__ Operator Tests
    # -------------------------------------------------------------------------

    def test_or_merges_two_structured_opts(self) -> None:
        """Test __or__ merges two StructuredOpts into a cfg dict."""
        opts1 = SampleOpts(cluster_name="test1", num_retries=5)
        opts2 = SampleOpts(cluster_name="test2", enable_debug=True)

        cfg = opts1 | opts2

        # Result is a dict, not StructuredOpts
        self.assertIsInstance(cfg, dict)
        # Should have all fields from both opts
        self.assertIn("cluster_name", cfg)
        self.assertIn("num_retries", cfg)
        self.assertIn("enable_debug", cfg)

    def test_or_second_opts_overwrites_first(self) -> None:
        """Test that values from second opts overwrite the first in merge."""

        @dataclass
        class OptsA(StructuredOpts):
            foo: str = "from_a"
            bar: int = 1

        @dataclass
        class OptsB(StructuredOpts):
            foo: str = "from_b"
            baz: bool = True

        opts_a = OptsA()
        opts_b = OptsB()

        cfg = opts_a | opts_b

        # foo from opts_b overwrites foo from opts_a
        self.assertEqual(cfg["foo"], "from_b")
        # bar from opts_a is preserved
        self.assertEqual(cfg["bar"], 1)
        # baz from opts_b is added
        self.assertEqual(cfg["baz"], True)

    def test_or_none_values_are_preserved(self) -> None:
        """Test that None values are preserved in the merge."""
        opts1 = SampleOpts(cluster_name="test1", optional_tag=None)
        opts2 = SampleOpts(cluster_name="test2", num_retries=10)

        cfg = opts1 | opts2

        # None values should be preserved
        self.assertIn("optional_tag", cfg)
        self.assertIsNone(cfg["optional_tag"])
