#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.schedulers.docker_scheduler import DockerScheduler
from torchx.schedulers.local_scheduler import LocalScheduler


class spy_load_group:
    def __call__(
        self,
        group: str,
        default: Dict[str, Any],
        ignore_missing: Optional[bool] = False,
        skip_defaults: bool = False,
    ) -> Dict[str, Any]:
        return default


class SchedulersTest(unittest.TestCase):
    @patch("torchx.schedulers.load_group", new_callable=spy_load_group)
    def test_get_local_schedulers(self, mock_load_group: MagicMock) -> None:
        schedulers = {}
        for k, v in get_scheduler_factories().items():
            try:
                schedulers[k] = v("test_session")
            except ModuleNotFoundError:
                pass
        self.assertTrue(isinstance(schedulers["local_cwd"], LocalScheduler))
        self.assertTrue(isinstance(schedulers["local_docker"], DockerScheduler))

        self.assertEqual(get_default_scheduler_name(), "local_docker")

        for scheduler in schedulers.values():
            self.assertEqual("test_session", scheduler.session_name)

    @patch("torchx.schedulers.load_group")
    def test_custom_schedulers_merged(self, mock_load_group: MagicMock) -> None:
        mock_scheduler = MagicMock()
        mock_load_group.return_value = {"custom": mock_scheduler}

        factories = get_scheduler_factories()

        self.assertIn("custom", factories)
        self.assertEqual(factories["custom"], mock_scheduler)
        self.assertIn("local_docker", factories)

    @patch("torchx.schedulers.load_group")
    def test_custom_scheduler_overrides_default(
        self, mock_load_group: MagicMock
    ) -> None:
        mock_scheduler = MagicMock()
        mock_load_group.return_value = {"local_docker": mock_scheduler}

        factories = get_scheduler_factories()

        self.assertEqual(factories["local_docker"], mock_scheduler)

    @patch("torchx.schedulers.load_group")
    def test_skip_defaults_with_custom_schedulers(
        self, mock_load_group: MagicMock
    ) -> None:
        mock_scheduler = MagicMock()
        mock_load_group.return_value = {"custom": mock_scheduler}

        factories = get_scheduler_factories(skip_defaults=True)

        self.assertEqual(factories, {"custom": mock_scheduler})
        self.assertNotIn("local_docker", factories)

    @patch("torchx.schedulers.load_group")
    def test_with_custom_schedulers_skip_defaults_false(
        self, mock_load_group: MagicMock
    ) -> None:
        """with custom schedulers, skip_defaults=False returns both"""
        mock_aws = MagicMock()
        mock_custom = MagicMock()
        mock_load_group.return_value = {"aws_batch": mock_aws, "custom_1": mock_custom}

        factories = get_scheduler_factories(skip_defaults=False)

        self.assertIn("aws_batch", factories)
        self.assertIn("custom_1", factories)
        self.assertIn("local_docker", factories)
        self.assertIn("slurm", factories)

    @patch("torchx.schedulers.load_group")
    def test_with_custom_schedulers_skip_defaults_true(
        self, mock_load_group: MagicMock
    ) -> None:
        """with custom schedulers, skip_defaults=True returns only custom"""
        mock_aws = MagicMock()
        mock_custom = MagicMock()
        mock_load_group.return_value = {"aws_batch": mock_aws, "custom_1": mock_custom}

        factories = get_scheduler_factories(skip_defaults=True)

        self.assertEqual(set(factories.keys()), {"aws_batch", "custom_1"})

    @patch("torchx.schedulers.load_group")
    def test_no_custom_schedulers_skip_defaults_false(
        self, mock_load_group: MagicMock
    ) -> None:
        """no custom schedulers, skip_defaults=False returns defaults"""
        mock_load_group.return_value = {}

        factories = get_scheduler_factories(skip_defaults=False)

        self.assertIn("local_docker", factories)
        self.assertIn("slurm", factories)

    @patch("torchx.schedulers.load_group")
    def test_no_custom_schedulers_skip_defaults_true(
        self, mock_load_group: MagicMock
    ) -> None:
        """no custom schedulers, skip_defaults=True returns empty"""
        mock_load_group.return_value = {}

        factories = get_scheduler_factories(skip_defaults=True)

        self.assertEqual(factories, {})

    @patch("torchx.schedulers.load_group")
    def test_custom_scheduler_is_default(self, mock_load_group: MagicMock) -> None:
        """first custom scheduler becomes the default"""
        mock_aws = MagicMock()
        mock_custom = MagicMock()
        mock_load_group.return_value = {"aws_batch": mock_aws, "custom_1": mock_custom}

        default_name = get_default_scheduler_name()

        self.assertIn(default_name, ["aws_batch", "custom_1"])
