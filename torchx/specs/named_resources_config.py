# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Configuration-based named resources that can be defined via .torchxconfig file.
This allows users to define custom named resources with specific CPU, GPU, memory,
and device requirements without hardcoding them.

Example .torchxconfig:
[named_resources]
dynamic = {"cpu": 100, "gpu": 8, "memMB": 819200, "devices": {"vpc.amazonaws.com/efa": 1}}
my_custom = {"cpu": 32, "gpu": 4, "memMB": 131072}
"""

import json
import os
from configparser import ConfigParser
from typing import Callable, Dict, Mapping

from torchx.specs.api import Resource


def _load_config_file() -> ConfigParser:
    """Load the .torchxconfig file from TORCHXCONFIG env var or current directory."""
    config = ConfigParser()

    # Check TORCHXCONFIG environment variable first, then current directory
    config_path = os.environ.get("TORCHXCONFIG", ".torchxconfig")

    if os.path.exists(config_path):
        config.read(config_path)

    return config


def _parse_resource_config(config_str: str) -> Resource:
    """Parse a resource configuration string into a Resource object."""
    try:
        config_dict = json.loads(config_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in resource configuration: {e}")

    # Extract standard resource parameters
    cpu = config_dict.get("cpu", 1)
    gpu = config_dict.get("gpu", 0)
    memMB = config_dict.get("memMB", 1024)

    # Extract optional parameters
    capabilities = config_dict.get("capabilities", {})
    devices = config_dict.get("devices", {})

    return Resource(
        cpu=cpu,
        gpu=gpu,
        memMB=memMB,
        capabilities=capabilities,
        devices=devices,
    )


def _create_resource_factory(config_str: str) -> Callable[[], Resource]:
    """Create a factory function for a resource configuration."""

    def factory() -> Resource:
        return _parse_resource_config(config_str)

    return factory


def _load_named_resources_from_config() -> Dict[str, Callable[[], Resource]]:
    """Load named resources from the configuration file."""
    config = _load_config_file()
    named_resources = {}

    if config.has_section("named_resources"):
        for name, config_str in config.items("named_resources"):
            named_resources[name] = _create_resource_factory(config_str)

    return named_resources


# Load named resources from configuration
NAMED_RESOURCES: Mapping[str, Callable[[], Resource]] = (
    _load_named_resources_from_config()
)
