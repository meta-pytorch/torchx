# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from argparse import ArgumentParser
from unittest import mock

from torchx.cli import argparse_util
from torchx.cli.argparse_util import ArgOnceAction, torchxconfig_list, torchxconfig_run
from torchx.test.fixtures import TestWithTmpDir

DEFAULT_CONFIG_DIRS = "torchx.runner.config.DEFAULT_CONFIG_DIRS"


class ArgparseUtilTest(TestWithTmpDir):
    def setUp(self) -> None:
        super().setUp()
        argparse_util.torchxconfig._subcmd_configs.clear()

    def test_torchxconfig_action(self) -> None:
        with mock.patch(DEFAULT_CONFIG_DIRS, [str(self.tmpdir)]):
            self.write(
                ".torchxconfig",
                """
[cli:run]
workspace = baz
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                default="foo",
                type=str,
                action=torchxconfig_run,
            )

            # arguments specified in CLI should take outmost precedence
            args = parser.parse_args(["run", "--workspace", "bar"])
            self.assertEqual("bar", args.workspace)

            # if not specified in CLI, then grab it from .torchxconfig
            args = parser.parse_args(["run"])
            self.assertEqual("baz", args.workspace)

    def test_arg_once_state_is_parse_scoped(self) -> None:
        # the duplicate-arg guard must not leak state across parses
        # in the same process
        parser = ArgumentParser()
        parser.add_argument("--workspace", type=str, action=ArgOnceAction)

        args_1 = parser.parse_args(["--workspace", "foo"])
        args_2 = parser.parse_args(["--workspace", "bar"])
        self.assertEqual("foo", args_1.workspace)
        self.assertEqual("bar", args_2.workspace)

        # a genuine duplicate within ONE parse still errors
        with self.assertRaises(SystemExit):
            parser.parse_args(["--workspace", "foo", "--workspace", "bar"])

    def test_torchxconfig_action_argparse_default(self) -> None:
        with mock.patch(DEFAULT_CONFIG_DIRS, [str(self.tmpdir)]):
            self.write(
                ".torchxconfig",
                """
[cli:run]
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                default="foo",
                type=str,
                action=torchxconfig_run,
            )

            # if not found in .torchxconfig should use argparse default
            args = parser.parse_args(["run"])
            self.assertEqual("foo", args.workspace)

    def test_torchxconfig_action_required(self) -> None:
        with mock.patch(DEFAULT_CONFIG_DIRS, [str(self.tmpdir)]):
            self.write(
                ".torchxconfig",
                """
[cli:run]
workspace = bazz
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                required=True,
                type=str,
                action=torchxconfig_run,
            )

            # arguments specified in CLI should take outmost precedence
            args = parser.parse_args(["run", "--workspace", "bar"])
            self.assertEqual("bar", args.workspace)

            # if not specified in CLI, then grab it from .torchxconfig
            args = parser.parse_args(["run"])
            self.assertEqual("bazz", args.workspace)

    def test_torchxconfig_action_aliases(self) -> None:
        # for aliases, the config file needs to declare the original arg
        with mock.patch(DEFAULT_CONFIG_DIRS, [str(self.tmpdir)]):
            self.write(
                ".torchxconfig",
                """
[cli:run]
workspace = baz
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                "--buck-target",
                type=str,
                required=True,
                action=torchxconfig_run,
            )

            args = parser.parse_args(["run"])
            self.assertEqual("baz", args.workspace)

    def test_torchxconfig_list_action(self) -> None:
        with mock.patch(DEFAULT_CONFIG_DIRS, [str(self.tmpdir)]):
            self.write(
                ".torchxconfig",
                """
[cli:list]
scheduler = kubernetes
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            list_parser = subparsers.add_parser("list")

            list_parser.add_argument(
                "--scheduler",
                default="local_cwd",
                type=str,
                action=torchxconfig_list,
            )

            # arguments specified in CLI should take outmost precedence
            args = parser.parse_args(["list", "--scheduler", "slurm"])
            self.assertEqual("slurm", args.scheduler)

            # if not specified in CLI, then grab it from .torchxconfig
            args = parser.parse_args(["list"])
            self.assertEqual("kubernetes", args.scheduler)
