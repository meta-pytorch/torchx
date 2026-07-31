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

from torchx.runtime.tracking.api import FsspecResultTracker


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp("torchx_runtime_tracking_test")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_put_get(self) -> None:
        tracker = FsspecResultTracker(self.test_dir)
        tracker["a/b"] = {
            "hartmann6": 1,
            "model_name": "foobar",
            "l2norm/mean": 3.4,
            "l2norm/sem": 5.6,
        }

        self.assertEqual(1, tracker["a/b"]["hartmann6"])
        self.assertEqual("foobar", tracker["a/b"]["model_name"])
        self.assertEqual(3.4, tracker["a/b"]["l2norm/mean"])
        self.assertEqual(5.6, tracker["a/b"]["l2norm/sem"])

    def test_get_missing_key(self) -> None:
        tracker = FsspecResultTracker(self.test_dir)
        res = tracker[1]
        self.assertFalse(res)

    def test_put_get_x2(self) -> None:
        tracker = FsspecResultTracker(self.test_dir)
        tracker[1] = {"l2norm": 1}
        tracker[1] = {"l2norm": 2}

        self.assertEqual(2, tracker["1"]["l2norm"])
        self.assertEqual(2, tracker["1"]["l2norm"])
