#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import configparser
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_configure import CmdConfigure
from torchx.schedulers import DEFAULT_SCHEDULER_MODULES
from torchx.schedulers.local_scheduler import create_scheduler as create_local_scheduler


class CmdConfigureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.cmd_configure = CmdConfigure()
        self.cmd_configure.add_arguments(self.parser)

        self.test_dir = tempfile.mkdtemp(prefix="torchx_cmd_configure_test")
        self._old_cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)
        shutil.rmtree(self.test_dir)

    def _args(self, sys_args: list[str]) -> argparse.Namespace:
        return self.parser.parse_args(sys_args)

    def test_configure_print(self) -> None:
        # nothing to assert, just make sure the cmd runs
        self.cmd_configure.run(self._args(["--print"]))
        self.cmd_configure.run(self._args(["--print", "--all"]))

    def test_configure(self) -> None:
        os.chdir(self.test_dir)
        self.cmd_configure.run(self._args([]))

        self.assertTrue((Path(self.test_dir) / ".torchxconfig").exists())

    def test_configure_all(self) -> None:
        self.cmd_configure.run(self._args(["--all"]))
        self.assertTrue((Path(self.test_dir) / ".torchxconfig").exists())

    def test_configure_local_cwd(self) -> None:
        self.cmd_configure.run(self._args(["--schedulers", "local_cwd"]))
        self.assertTrue((Path(self.test_dir) / ".torchxconfig").exists())

    @patch("torchx.schedulers.plugins")
    def test_configure_dumps_plugins_and_builtins(
        self, plugins_mock: MagicMock
    ) -> None:
        plugins_mock.registry.return_value.get.return_value = {
            "custom_sched": create_local_scheduler
        }

        self.cmd_configure.run(self._args([]))

        config = configparser.ConfigParser()
        config.read(Path(self.test_dir) / ".torchxconfig")

        self.assertIn(
            "custom_sched", config.sections(), "the plugin scheduler must be dumped"
        )
        for name in DEFAULT_SCHEDULER_MODULES:
            self.assertIn(
                name, config.sections(), f"built-in `{name}` must still be dumped"
            )
