#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from unittest.mock import MagicMock, patch

from torchx.specs import (
    _NamedResourcesLibrary,
    named_resources,
    NULL_RESOURCE,
    Resource,
)


def mock_resource() -> Resource:
    return Resource(cpu=0, gpu=0, memMB=0)


class NamedResourcesTest(unittest.TestCase):
    @patch("torchx.specs._named_resource_factories")
    def test_named_resources_library(self, mock_named_resources: MagicMock) -> None:
        mock_named_resources.keys.return_value = [
            "p3.2xlarge",
            "p3.16xlarge",
            "p4d.24xlarge",
        ]

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `foo`. Registered named resources:.*",
        ):
            _ = _NamedResourcesLibrary()["foo"]

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `p316xl`. Did you mean `p3.16xlarge`?",
        ):
            _ = _NamedResourcesLibrary()["p316xl"]

    def test_null_and_missing_named_resources(self) -> None:
        self.assertEqual(named_resources["NULL"], NULL_RESOURCE)
        self.assertEqual(named_resources["MISSING"], NULL_RESOURCE)

    def test_custom_named_resources_env_var(self) -> None:
        import sys

        mock_module = type(sys)("test_module")
        mock_module.NAMED_RESOURCES = {"test_resource": mock_resource}

        with patch.dict(sys.modules, {"test_module": mock_module}):
            with patch(
                "torchx.specs.CUSTOM_NAMED_RESOURCES", mock_module.NAMED_RESOURCES
            ):
                import torchx.specs

                factories = torchx.specs._load_named_resources()

                self.assertIn("test_resource", factories)
