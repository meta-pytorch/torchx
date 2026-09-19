#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import unittest
from unittest.mock import MagicMock, patch

from torchx.cli.argparse_util import torchxconfig
from torchx.cli.cmd_list import CmdList
from torchx.schedulers import DEFAULT_SCHEDULER_MODULES
from torchx.schedulers.local_scheduler import create_scheduler as create_local_scheduler


class CmdListTest(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the class variables to prevent state leaking between tests
        torchxconfig._subcmd_configs = {}

    def tearDown(self) -> None:
        # Reset the class variables after each test
        torchxconfig._subcmd_configs = {}

    @patch("torchx.runner.config.apply")
    @patch("torchx.runner.api.Runner.list")
    def test_run(self, list_mock: MagicMock, config_apply_mock: MagicMock) -> None:
        parser = argparse.ArgumentParser()
        cmd_list = CmdList()
        cmd_list.add_arguments(parser)

        args = parser.parse_args(
            [
                "--scheduler",
                "kubernetes",
            ]
        )
        cmd_list.run(args)

        config_apply_mock.assert_called_with(scheduler="kubernetes", cfg={})
        self.assertEqual(list_mock.call_count, 1)
        list_mock.assert_called_with("kubernetes", None)

    @patch("torchx.runner.config.apply")
    @patch("torchx.runner.api.Runner.scheduler_run_opts")
    @patch("torchx.runner.api.Runner.list")
    def test_run_with_cfg(
        self,
        list_mock: MagicMock,
        run_opts_mock: MagicMock,
        config_apply_mock: MagicMock,
    ) -> None:
        # Mock the scheduler_run_opts to return a runopts that can parse the args
        mock_runopts = MagicMock()
        mock_runopts.cfg_from_str.return_value = {"cluster": "foo"}
        run_opts_mock.return_value = mock_runopts

        parser = argparse.ArgumentParser()
        cmd_list = CmdList()
        cmd_list.add_arguments(parser)

        args = parser.parse_args(
            ["--scheduler", "kubernetes", "--scheduler_args", "cluster=foo"]
        )
        cmd_list.run(args)

        run_opts_mock.assert_called_with("kubernetes")
        mock_runopts.cfg_from_str.assert_called_with("cluster=foo")
        config_apply_mock.assert_called_with(
            scheduler="kubernetes", cfg={"cluster": "foo"}
        )
        list_mock.assert_called_with("kubernetes", {"cluster": "foo"})

    @patch("torchx.schedulers.plugins")
    def test_scheduler_choices_include_plugins_and_builtins(
        self, plugins_mock: MagicMock
    ) -> None:
        plugins_mock.registry.return_value.get.return_value = {
            "custom_sched": create_local_scheduler
        }

        parser = argparse.ArgumentParser()
        CmdList().add_arguments(parser)
        action = next(a for a in parser._actions if a.dest == "scheduler")

        self.assertEqual(
            {"custom_sched", *DEFAULT_SCHEDULER_MODULES},
            set(action.choices or []),
            "`torchx list -s` must offer the plugin and every built-in",
        )
        self.assertEqual(
            "custom_sched", action.default, "the registered plugin is the default"
        )
