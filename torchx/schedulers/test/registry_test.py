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
