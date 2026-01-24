#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import warnings
from functools import partial
from typing import Callable, Mapping

from torchx.specs.api import DeviceMount
from torchx.specs.named_resources_aws import EFA_DEVICE, NEURON_DEVICE


def to_devicemounts(num_devices: int, device_type: str) -> list[DeviceMount]:
    device_mounts = []
    for device_index in range(0, num_devices):
        device_mounts.append(
            DeviceMount(
                src_path=device_type + str(device_index),
                dst_path=device_type + str(device_index),
            )
        )
    return device_mounts


neuron_to_devicemounts: Callable[[int], list[DeviceMount]] = partial(
    to_devicemounts, device_type="/dev/neuron"
)
efa_to_devicemounts: Callable[[int], list[DeviceMount]] = partial(
    to_devicemounts, device_type="/dev/infiniband/uverbs"
)


DEVICES: Mapping[str, Callable[[int], list[DeviceMount]]] = {
    EFA_DEVICE: efa_to_devicemounts,
    NEURON_DEVICE: neuron_to_devicemounts,
}


def get_device_mounts(devices: dict[str, int]) -> list[DeviceMount]:
    """
    Takes in a list of named devices/quantities, and returns a list of DeviceMount objects
    based on the mappings defined in DEVICES
    """
    device_mounts = []
    for device_name, num_devices in devices.items():
        if device_name not in DEVICES:
            warnings.warn(f"Could not find named device: {device_name}")
            continue
        device_mounts += DEVICES[device_name](num_devices)
    return device_mounts
