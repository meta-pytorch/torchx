#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from torchx.cli.cmd_configure import CmdConfigure


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
