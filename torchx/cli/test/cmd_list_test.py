#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import unittest
from unittest.mock import MagicMock, patch

from torchx.cli.argparse_util import torchxconfig
from torchx.cli.cmd_list import CmdList


class CmdListTest(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the class variables to prevent state leaking between tests
        torchxconfig.called_args = set()
        torchxconfig._subcmd_configs = {}

    def tearDown(self) -> None:
        # Reset the class variables after each test
        torchxconfig.called_args = set()
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
