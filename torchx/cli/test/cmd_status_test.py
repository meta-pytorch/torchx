#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import io
import json
import unittest
from unittest.mock import patch

from torchx.cli.cmd_status import CmdStatus
from torchx.specs.api import AppState, AppStatus, ReplicaStatus, RoleStatus


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

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_human_output(self, stdout: io.StringIO) -> None:
        parser = argparse.ArgumentParser()
        cmd_status = CmdStatus()
        cmd_status.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app"])

        app_status = AppStatus(state=AppState.RUNNING)
        with patch("torchx.runner.api.Runner.status", return_value=app_status):
            cmd_status.run(args)

        self.assertEqual(stdout.getvalue(), app_status.format() + "\n")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_run_json_output(self, stdout: io.StringIO) -> None:
        parser = argparse.ArgumentParser()
        cmd_status = CmdStatus()
        cmd_status.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app", "--json"])

        app_status = AppStatus(
            state=AppState.RUNNING,
            ui_url="https://scheduler.example.com/jobs/test_app",
            roles=[
                RoleStatus(
                    role="worker",
                    replicas=[
                        ReplicaStatus(
                            id=0,
                            state=AppState.RUNNING,
                            role="worker",
                            hostname="localhost",
                        )
                    ],
                )
            ],
        )
        with patch("torchx.runner.api.Runner.status", return_value=app_status):
            cmd_status.run(args)

        status_json = json.loads(stdout.getvalue())
        self.assertEqual(status_json["handle"], "local://test_session/test_app")
        self.assertEqual(status_json["app_id"], "test_app")
        self.assertEqual(status_json["scheduler"], "local")
        self.assertEqual(status_json["state"], "RUNNING")
        self.assertEqual(
            status_json["ui_url"], "https://scheduler.example.com/jobs/test_app"
        )
        self.assertEqual(status_json["url"], status_json["ui_url"])
        self.assertEqual(len(status_json["roles"]), 1)
        self.assertEqual(status_json["roles"][0]["role"], "worker")
