#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.schedulers.docker_scheduler import DockerScheduler
from torchx.schedulers.local_scheduler import LocalScheduler


class spy_load_group:
    def __call__(
        self,
        group: str,
        default: dict[str, Any],
        ignore_missing: bool | None = False,
        skip_defaults: bool = False,
    ) -> dict[str, Any]:
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
    def test_torchx_schedulers_overrides_all(self, mock_load_group: MagicMock) -> None:
        """torchx.schedulers completely overrides defaults and ignores extras"""
        mock_custom: MagicMock = MagicMock()
        mock_extra: MagicMock = MagicMock()

        mock_load_group.side_effect = lambda group, default: (
            {"custom": mock_custom}
            if group == "torchx.schedulers"
            else {"extra": mock_extra} if group == "torchx.schedulers.extra" else {}
        )

        factories = get_scheduler_factories()

        self.assertEqual(factories, {"custom": mock_custom})
        self.assertNotIn("local_docker", factories)
        self.assertNotIn("extra", factories)

    @patch("torchx.schedulers.load_group")
    def test_no_custom_returns_defaults_and_extras(
        self, mock_load_group: MagicMock
    ) -> None:
        """no custom schedulers returns built-in + extras"""
        mock_extra: MagicMock = MagicMock()

        mock_load_group.side_effect = lambda group, default: (
            {"extra": mock_extra} if group == "torchx.schedulers.extra" else default
        )

        factories = get_scheduler_factories()

        self.assertIn("local_docker", factories)
        self.assertIn("slurm", factories)
        self.assertIn("extra", factories)

    @patch("torchx.schedulers.load_group")
    def test_no_custom_no_extras_returns_builtins(
        self, mock_load_group: MagicMock
    ) -> None:
        """no custom, no extras returns only built-in schedulers"""
        mock_load_group.side_effect = lambda group, default: default

        factories = get_scheduler_factories()

        self.assertIn("local_docker", factories)
        self.assertIn("slurm", factories)

    @patch("torchx.schedulers.load_group")
    def test_skip_defaults_returns_empty(self, mock_load_group: MagicMock) -> None:
        """skip_defaults=True with no custom schedulers returns empty"""
        mock_load_group.side_effect = lambda group, default: default

        factories = get_scheduler_factories(skip_defaults=True)

        self.assertEqual(factories, {})

    @patch("torchx.schedulers.load_group")
    def test_custom_scheduler_is_default(self, mock_load_group: MagicMock) -> None:
        """first custom scheduler becomes the default"""
        mock_aws: MagicMock = MagicMock()
        mock_custom: MagicMock = MagicMock()

        mock_load_group.side_effect = lambda group, default: (
            {"aws_batch": mock_aws, "custom_1": mock_custom}
            if group == "torchx.schedulers"
            else {}
        )

        default_name = get_default_scheduler_name()

        self.assertIn(default_name, ["aws_batch", "custom_1"])
