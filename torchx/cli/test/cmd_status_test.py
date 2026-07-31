#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import unittest
from unittest.mock import patch

from torchx.cli.cmd_status import CmdStatus
from torchx.specs.api import AppState, AppStatus


class CmdStatusTest(unittest.TestCase):
    def test_run(self) -> None:
        parser = argparse.ArgumentParser()
        cmd_status = CmdStatus()
        cmd_status.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app"])

        for app_status in [None, AppStatus(state=AppState.RUNNING)]:
            with self.subTest(app_status=app_status):
                with patch("torchx.runner.api.Runner.status") as status_mock:
                    status_mock.return_value = app_status

                    try:
                        cmd_status.run(args)
                        exit_code = None
                    except SystemExit as e:
                        exit_code = e.code

                    status_mock.assert_called_once_with(args.app_handle)

                    if app_status is None:
                        self.assertEqual(exit_code, 1)
                    else:
                        self.assertIsNone(exit_code)
