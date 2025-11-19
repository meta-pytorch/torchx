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

from torchx.cli.cmd_delete import CmdDelete


class CmdDeleteTest(unittest.TestCase):
    @patch("torchx.runner.api.Runner.delete")
    def test_run(self, delete: MagicMock) -> None:
        parser = argparse.ArgumentParser()
        cmd_delete = CmdDelete()
        cmd_delete.add_arguments(parser)

        args = parser.parse_args(["foo://session/id"])
        cmd_delete.run(args)

        self.assertEqual(delete.call_count, 1)
        delete.assert_called_with("foo://session/id")
