#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import unittest
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_tracker import CmdTracker
from torchx.tracker.api import TrackerArtifact

CMD_BUILD_TRACKERS = "torchx.cli.cmd_tracker.build_trackers"
CMD_TRACKER = "torchx.cli.cmd_tracker.CmdTracker.tracker"


class CmdTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker_name = "my_tracker"
        self.tracker_config = "myconfig.txt"
        self.test_run_id = "scheduler://session/app_id"

    @patch(CMD_BUILD_TRACKERS, return_value=[])
    def test_initalize_on_missing_tracker(self, _: MagicMock) -> None:
        # should be able to create the object even if no tracker configured
        cmd = CmdTracker()

        with self.assertRaises(RuntimeError):
            # the command actions won't work without a configured
            # tracker so when we ask for a tracker it should raise
            cmd.tracker

    @patch(CMD_TRACKER)
    def test_list_jobs_cmd(self, mock_tracker: MagicMock) -> None:
        mock_tracker.run_ids.return_value = ["id1", "id2"]

        parser = argparse.ArgumentParser()
        CmdTracker().add_arguments(parser)
        args = parser.parse_args(["list", "jobs"])
        args.func(args)
        mock_tracker.run_ids.assert_called_once()

    @patch(CMD_BUILD_TRACKERS, return_value=[])
    def test_list_jobs_cmd_with_missing_tracker(self, _: MagicMock) -> None:
        cmd = CmdTracker()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args(["list", "jobs"])
        with self.assertRaises(RuntimeError):
            args.func(args)

    @patch(CMD_TRACKER)
    def test_list_jobs_cmd_with_parent_id(self, mock_tracker: MagicMock) -> None:
        expected_parent_run_id = "my_experiment"
        mock_tracker.run_ids.return_value = ["id1", "id2"]

        parser = argparse.ArgumentParser()
        CmdTracker().add_arguments(parser)
        args = parser.parse_args(
            ["list", "jobs", "--parent-run-id", expected_parent_run_id]
        )
        args.func(args)

        mock_tracker.run_ids.assert_called_once_with(
            parent_run_id=expected_parent_run_id
        )

    @patch(CMD_TRACKER)
    def test_list_metadata_cmd(self, mock_tracker: MagicMock) -> None:
        mock_tracker.metadata.return_value = {"v1": 1, "v2": "2"}

        parser = argparse.ArgumentParser()
        CmdTracker().add_arguments(parser)

        args = parser.parse_args(["list", "metadata", self.test_run_id])
        args.func(args)
        mock_tracker.metadata.assert_called_once()

    @patch(CMD_TRACKER)
    def test_list_artifacts_cmd(self, mock_tracker: MagicMock) -> None:
        mock_tracker.artifacts.return_value = {
            "test_artifact": TrackerArtifact("test_artifact", "/path", None)
        }

        parser = argparse.ArgumentParser()
        CmdTracker().add_arguments(parser)

        args = parser.parse_args(["list", "artifacts", self.test_run_id])
        args.func(args)
        mock_tracker.artifacts.assert_called_once_with(self.test_run_id)
