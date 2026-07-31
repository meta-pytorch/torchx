#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

from torchx.schedulers.devices import get_device_mounts
from torchx.specs.api import DeviceMount


class DevicesTest(unittest.TestCase):
    def test_get_efa(self) -> None:
        devices = {"vpc.amazonaws.com/efa": 2, "aws.amazon.com/neurondevice": 1}
        self.assertEqual(
            get_device_mounts(devices),
            [
                DeviceMount(
                    src_path="/dev/infiniband/uverbs0",
                    dst_path="/dev/infiniband/uverbs0",
                ),
                DeviceMount(
                    src_path="/dev/infiniband/uverbs1",
                    dst_path="/dev/infiniband/uverbs1",
                ),
                DeviceMount(src_path="/dev/neuron0", dst_path="/dev/neuron0"),
            ],
        )

    def test_not_found_device_name(self) -> None:
        devices = {"shouldWarn": 1}
        self.assertEqual(get_device_mounts(devices), [])
        self.assertWarns(UserWarning, get_device_mounts, devices)
