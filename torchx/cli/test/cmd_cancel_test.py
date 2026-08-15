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

from torchx.cli.cmd_cancel import CmdCancel


class CmdCancelTest(unittest.TestCase):
    @patch("torchx.runner.api.Runner.cancel")
    def test_run(self, cancel: MagicMock) -> None:
        parser = argparse.ArgumentParser()
        cmd_runopts = CmdCancel()
        cmd_runopts.add_arguments(parser)

        args = parser.parse_args(["foo://session/id"])
        cmd_runopts.run(args)

        self.assertEqual(cancel.call_count, 1)
        cancel.assert_called_with("foo://session/id")

    @patch("torchx.runner.api.Runner.close")
    @patch("torchx.runner.api.Runner.cancel")
    def test_run_closes_runner(self, cancel: MagicMock, close: MagicMock) -> None:
        """Pins AppHandleSubCommand's runner lifecycle: `with get_runner()`
        closes scheduler clients when the command body returns."""
        parser = argparse.ArgumentParser()
        cmd_cancel = CmdCancel()
        cmd_cancel.add_arguments(parser)

        cmd_cancel.run(parser.parse_args(["foo://session/id"]))

        close.assert_called_once()
