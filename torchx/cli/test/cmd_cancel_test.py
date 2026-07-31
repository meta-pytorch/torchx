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
