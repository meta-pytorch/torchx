#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest.mock import MagicMock, patch

from torchx.schedulers import (
    DEFAULT_SCHEDULER_MODULES,
    get_default_scheduler_name,
    get_scheduler_factories,
)
from torchx.schedulers.docker_scheduler import DockerScheduler
from torchx.schedulers.local_scheduler import LocalScheduler


class SchedulersTest(unittest.TestCase):
    @patch("torchx.schedulers.plugins")
    def test_plugins_add_to_defaults(self, plugins_mock: MagicMock) -> None:
        sentinel = MagicMock()
        plugins_mock.registry.return_value.get.return_value = {"custom_sched": sentinel}

        result = get_scheduler_factories()

        self.assertEqual(
            sentinel, result["custom_sched"], "the registered plugin must be present"
        )
        self.assertEqual(
            set(DEFAULT_SCHEDULER_MODULES),
            set(result) - {"custom_sched"},
            "registering a plugin must not drop any built-in scheduler",
        )

    @patch("torchx.schedulers.plugins")
    def test_plugin_wins_a_name_clash_with_a_builtin(
        self, plugins_mock: MagicMock
    ) -> None:
        sentinel = MagicMock()
        plugins_mock.registry.return_value.get.return_value = {"local_docker": sentinel}

        result = get_scheduler_factories()

        self.assertEqual(
            sentinel,
            result["local_docker"],
            "a plugin registered under a built-in name must replace it",
        )
        self.assertEqual(
            set(DEFAULT_SCHEDULER_MODULES),
            set(result),
            "overriding a built-in must not add or drop a name",
        )

    @patch("torchx.schedulers.plugins")
    def test_registered_plugin_becomes_the_default(
        self, plugins_mock: MagicMock
    ) -> None:
        plugins_mock.registry.return_value.get.return_value = {
            "custom_sched": MagicMock()
        }

        self.assertEqual("custom_sched", get_default_scheduler_name())

    @patch("torchx.schedulers.plugins")
    def test_skip_defaults_returns_plugins_only(self, plugins_mock: MagicMock) -> None:
        sentinel = MagicMock()
        plugins_mock.registry.return_value.get.return_value = {"custom_sched": sentinel}

        self.assertEqual(
            {"custom_sched": sentinel}, get_scheduler_factories(skip_defaults=True)
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
