#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from torchx.cli.argparse_util import ArgOnceAction, torchxconfig
from torchx.cli.cmd_base import SubCommand
from torchx.cli.main import get_sub_cmds, main


_root: Path = Path(__file__).parent

_SIMPLE_CONF: str = str(Path(__file__).parent / "components.py:simple")


class _TestCmd(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        pass

    def run(self, args: argparse.Namespace) -> None:
        pass


class CLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_cwd = os.getcwd()
        os.chdir(_root / "container")

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        ArgOnceAction.called_args = set()
        torchxconfig.called_args = set()

    def test_run_abs_config_path(self) -> None:
        main(
            [
                "run",
                "--scheduler",
                "local_cwd",
                str(_root / "components.py:simple"),
                "--num_trainers",
                "2",
            ]
        )

    def test_run_builtin_config(self) -> None:
        main(
            [
                "run",
                "--scheduler",
                "local_cwd",
                _SIMPLE_CONF,
                "--num_trainers",
                "2",
            ]
        )

    def test_version(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "--version",
                ]
            )

    @patch("torchx.cli.main.load_group")
    def test_get_sub_cmds(self, load_group_mock: MagicMock) -> None:
        load_group_mock.return_value = {"run": _TestCmd}
        cmds = get_sub_cmds()
        self.assertIsInstance(cmds["run"], _TestCmd)
