# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import unittest
from configparser import ConfigParser
from unittest.mock import patch

from torchx.specs.named_resources_config import (
    _create_resource_factory,
    _load_config_file,
    _load_named_resources_from_config,
    _parse_resource_config,
    NAMED_RESOURCES,
)


class ConfigNamedResourcesTest(unittest.TestCase):
    def test_parse_resource_config_basic(self) -> None:
        """Test parsing basic resource configuration."""
        config_str = '{"cpu": 32, "gpu": 4, "memMB": 131072}'
        resource = _parse_resource_config(config_str)

        self.assertEqual(resource.cpu, 32)
        self.assertEqual(resource.gpu, 4)
        self.assertEqual(resource.memMB, 131072)
        self.assertEqual(resource.capabilities, {})
        self.assertEqual(resource.devices, {})

    def test_parse_resource_config_with_devices(self) -> None:
        """Test parsing resource configuration with devices."""
        config_str = '{"cpu": 100, "gpu": 8, "memMB": 819200, "devices": {"vpc.amazonaws.com/efa": 1}}'
        resource = _parse_resource_config(config_str)

        self.assertEqual(resource.cpu, 100)
        self.assertEqual(resource.gpu, 8)
        self.assertEqual(resource.memMB, 819200)
        self.assertEqual(resource.devices, {"vpc.amazonaws.com/efa": 1})

    def test_parse_resource_config_with_capabilities(self) -> None:
        """Test parsing resource configuration with capabilities."""
        config_str = '{"cpu": 64, "gpu": 0, "memMB": 262144, "capabilities": {"node.kubernetes.io/instance-type": "m5.16xlarge"}}'
        resource = _parse_resource_config(config_str)

        self.assertEqual(resource.cpu, 64)
        self.assertEqual(resource.gpu, 0)
        self.assertEqual(resource.memMB, 262144)
        self.assertEqual(
            resource.capabilities, {"node.kubernetes.io/instance-type": "m5.16xlarge"}
        )

    def test_parse_resource_config_defaults(self) -> None:
        """Test parsing resource configuration with default values."""
        config_str = '{"cpu": 16, "memMB": 65536}'
        resource = _parse_resource_config(config_str)

        self.assertEqual(resource.cpu, 16)
        self.assertEqual(resource.gpu, 0)  # default
        self.assertEqual(resource.memMB, 65536)

    def test_parse_resource_config_invalid_json(self) -> None:
        """Test parsing invalid JSON configuration."""
        config_str = '{"cpu": 32, "gpu": 4, "memMB": 131072'  # missing closing brace

        with self.assertRaises(ValueError) as cm:
            _parse_resource_config(config_str)

        self.assertIn("Invalid JSON", str(cm.exception))

    def test_create_resource_factory(self) -> None:
        """Test creating resource factory function."""
        config_str = '{"cpu": 8, "gpu": 1, "memMB": 32768}'
        factory = _create_resource_factory(config_str)

        resource = factory()
        self.assertEqual(resource.cpu, 8)
        self.assertEqual(resource.gpu, 1)
        self.assertEqual(resource.memMB, 32768)

    def test_load_config_file_not_found(self) -> None:
        """Test loading config file when none exists."""
        with patch("os.path.exists", return_value=False):
            config = _load_config_file()
            self.assertFalse(config.sections())

    def test_load_config_file_current_directory(self) -> None:
        """Test loading config file from current directory."""
        with patch.dict("os.environ", {}, clear=True):  # Clear TORCHXCONFIG
            with patch(
                "torchx.specs.named_resources_config.os.path.exists", return_value=True
            ) as mock_exists:
                with patch("configparser.ConfigParser.read") as mock_read:
                    _load_config_file()

                    # Verify the method was called with current directory path
                    mock_exists.assert_called_with(".torchxconfig")
                    mock_read.assert_called_with(".torchxconfig")

    def test_load_config_file_with_torchxconfig_env(self) -> None:
        """Test loading config file from TORCHXCONFIG environment variable."""
        temp_config_path = "/tmp/custom_torchx_config"

        with patch.dict("os.environ", {"TORCHXCONFIG": temp_config_path}):
            with patch(
                "torchx.specs.named_resources_config.os.path.exists", return_value=True
            ):
                with patch("configparser.ConfigParser.read") as mock_read:
                    _load_config_file()

                    # Verify the method was called with the env var path
                    mock_read.assert_called_with(temp_config_path)

    def test_load_named_resources_from_config_empty(self) -> None:
        """Test loading named resources when no config section exists."""
        with patch(
            "torchx.specs.named_resources_config._load_config_file"
        ) as mock_load:
            mock_config = ConfigParser()
            mock_load.return_value = mock_config

            resources = _load_named_resources_from_config()
            self.assertEqual(resources, {})

    def test_load_named_resources_from_config_with_resources(self) -> None:
        """Test loading named resources from config with valid resources."""
        with patch(
            "torchx.specs.named_resources_config._load_config_file"
        ) as mock_load:
            mock_config = ConfigParser()
            mock_config.add_section("named_resources")
            mock_config.set(
                "named_resources",
                "test_resource",
                json.dumps({"cpu": 32, "gpu": 4, "memMB": 131072}),
            )
            mock_config.set(
                "named_resources",
                "gpu_resource",
                json.dumps(
                    {
                        "cpu": 64,
                        "gpu": 8,
                        "memMB": 262144,
                        "devices": {"vpc.amazonaws.com/efa": 2},
                    }
                ),
            )
            mock_load.return_value = mock_config

            resources = _load_named_resources_from_config()

            self.assertIn("test_resource", resources)
            self.assertIn("gpu_resource", resources)

            # Test the factory functions
            test_res = resources["test_resource"]()
            self.assertEqual(test_res.cpu, 32)
            self.assertEqual(test_res.gpu, 4)
            self.assertEqual(test_res.memMB, 131072)

            gpu_res = resources["gpu_resource"]()
            self.assertEqual(gpu_res.cpu, 64)
            self.assertEqual(gpu_res.gpu, 8)
            self.assertEqual(gpu_res.memMB, 262144)
            self.assertEqual(gpu_res.devices, {"vpc.amazonaws.com/efa": 2})

    def test_load_named_resources_from_config_invalid_json(self) -> None:
        """Test loading named resources with invalid JSON (should fail when factory is called)."""
        with patch(
            "torchx.specs.named_resources_config._load_config_file"
        ) as mock_load:
            mock_config = ConfigParser()
            mock_config.add_section("named_resources")
            mock_config.set(
                "named_resources",
                "valid_resource",
                json.dumps({"cpu": 32, "gpu": 4, "memMB": 131072}),
            )
            mock_config.set(
                "named_resources",
                "invalid_resource",
                '{"cpu": 32, "gpu": 4, "memMB": 131072',
            )  # invalid JSON
            mock_load.return_value = mock_config

            resources = _load_named_resources_from_config()

            # Should have both resources (validation happens when factory is called)
            self.assertIn("valid_resource", resources)
            self.assertIn("invalid_resource", resources)

            # Valid resource should work
            valid_res = resources["valid_resource"]()
            self.assertEqual(valid_res.cpu, 32)

            # Invalid resource should raise exception when called
            with self.assertRaises(ValueError):
                resources["invalid_resource"]()

    def test_named_resources_module_level(self) -> None:
        """Test that NAMED_RESOURCES is properly loaded at module level."""
        # This tests the actual module-level NAMED_RESOURCES
        # The exact content depends on the actual .torchxconfig file present
        self.assertIsInstance(NAMED_RESOURCES, dict)

        # Test that all values are callable factory functions
        for name, factory in NAMED_RESOURCES.items():
            self.assertTrue(callable(factory))
            # Test that calling the factory returns a Resource
            resource = factory()
            self.assertTrue(hasattr(resource, "cpu"))
            self.assertTrue(hasattr(resource, "gpu"))
            self.assertTrue(hasattr(resource, "memMB"))
