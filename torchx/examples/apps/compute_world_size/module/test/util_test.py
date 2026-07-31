# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import random
import unittest
from unittest import mock

from omegaconf import DictConfig
from torchx.examples.apps.compute_world_size.module.util import compute_world_size


class UtilTest(unittest.TestCase):
    # clears env vars like MASTER_PORT, MASTER_ADDR that may be lingering from other tests
    # when running as a part of a test suite
    mock.patch.dict(os.environ, clear=True)

    def test_compute_world_size(self) -> None:
        cfg = DictConfig(
            content={
                "main": {
                    "rank": 0,
                    "world_size": 1,
                    "master_addr": "localhost",
                    # specifying 0 as master_port makes TCPStore chose a free random port
                    "master_port": 0,
                    "backend": "gloo",
                }
            }
        )

        self.assertEqual(1, compute_world_size(cfg))
