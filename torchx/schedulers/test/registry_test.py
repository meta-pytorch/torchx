#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.schedulers.docker_scheduler import DockerScheduler
from torchx.schedulers.local_scheduler import LocalScheduler


class SchedulersTest(unittest.TestCase):
    @patch("torchx.schedulers.plugins")
    def test_plugins_override_defaults(self, plugins_mock: MagicMock) -> None:
        """When plugins return non-empty dict, defaults are not used."""
        sentinel = MagicMock()
        plugins_mock.registry.return_value.get.return_value = {"custom_sched": sentinel}
        result = get_scheduler_factories()
        self.assertEqual(
            result,
            {"custom_sched": sentinel},
            "should return plugin result when non-empty",
        )
        self.assertNotIn(
            "local_cwd",
            result,
            "defaults should be skipped when plugins return non-empty",
        )

    @patch("torchx.schedulers.plugins")
    def test_get_local_schedulers(self, plugins_mock: MagicMock) -> None:
        plugins_mock.registry.return_value.get.return_value = {}
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
