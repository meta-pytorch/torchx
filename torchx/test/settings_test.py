# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx import settings


class SettingsValueTest(unittest.TestCase):
    """Verify that each constant holds the expected string value."""

    def test_tracker_env_vars(self) -> None:
        self.assertEqual(settings.ENV_TORCHX_TRACKERS, "TORCHX_TRACKERS")
        self.assertEqual(settings.ENV_TORCHX_PARENT_RUN_ID, "TORCHX_PARENT_RUN_ID")
        self.assertEqual(settings.ENV_TORCHX_JOB_ID, "TORCHX_JOB_ID")

    def test_config_env_var(self) -> None:
        self.assertEqual(settings.ENV_TORCHXCONFIG, "TORCHXCONFIG")

    def test_session_env_var(self) -> None:
        self.assertEqual(
            settings.TORCHX_INTERNAL_SESSION_ID, "TORCHX_INTERNAL_SESSION_ID"
        )

    def test_scheduler_env_vars(self) -> None:
        self.assertEqual(settings.ENV_TORCHX_ROLE_IDX, "TORCHX_ROLE_IDX")
        self.assertEqual(settings.ENV_TORCHX_ROLE_NAME, "TORCHX_ROLE_NAME")
        self.assertEqual(settings.ENV_TORCHX_IMAGE, "TORCHX_IMAGE")


class SettingsBCReexportTest(unittest.TestCase):
    """Verify that BC re-exports are the *same object* as the canonical constant.

    Using ``assertIs`` ensures the re-exports are true aliases (not copies),
    so ``is`` checks and identity-based caching remain correct.
    """

    def test_bc_reexport_session(self) -> None:
        from torchx.util import session

        self.assertIs(
            session.TORCHX_INTERNAL_SESSION_ID,
            settings.TORCHX_INTERNAL_SESSION_ID,
            "session re-export should be the same object",
        )
