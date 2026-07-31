#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import shutil
import tempfile
import unittest

import torchx.apps.utils.booth_main as booth
from torchx.runtime.tracking import FsspecResultTracker


class BoothTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp("torchx_apps_utils_booth_test")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_booth(self) -> None:
        # evaluate booth function at (1,3) - which is its global minimum (0)
        booth.main(["--x1", "1", "--x2", "3", "--tracker_base", self.test_dir])

        tracker = FsspecResultTracker(self.test_dir)
        self.assertEqual(0.0, tracker[0]["booth_eval"])
