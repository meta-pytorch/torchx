#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx.specs import app_metadata, NA, TORCHX_CONTEXT_NAME


class MetadataKeysGoldenSpellingTest(unittest.TestCase):
    """These literals are downstream telemetry join keys — a changed spelling
    silently breaks downstream telemetry, so each is asserted against the
    literal."""

    def test_env_var_spelling(self) -> None:
        self.assertEqual(
            "TORCHX_CONTEXT_NAME",
            TORCHX_CONTEXT_NAME,
            "context env var is a wire contract",
        )

    def test_na_sentinel_spelling(self) -> None:
        self.assertEqual("<<NA>>", NA, "NA sentinel is a wire contract")

    def test_app_metadata_key_spellings(self) -> None:
        self.assertEqual(
            "torchx/context",
            app_metadata.CONTEXT,
            "context metadata key is a downstream telemetry join key",
        )
        self.assertEqual(
            "torchx/version",
            app_metadata.VERSION,
            "version metadata key is a downstream telemetry join key",
        )
        self.assertEqual(
            "torchx/scheduler",
            app_metadata.SCHEDULER,
            "scheduler metadata key is a downstream telemetry join key",
        )
        self.assertEqual(
            "torchx/roles.",
            app_metadata.ROLES_PREFIX,
            "per-role metadata key prefix is a wire contract",
        )
