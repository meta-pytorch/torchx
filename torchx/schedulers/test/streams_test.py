#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import os.path
import shutil
import tempfile
import unittest

from torchx.schedulers.streams import Tee


class TeeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="torchx_runner_config_test")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_combined(self) -> None:
        a_path = os.path.join(self.test_dir, "a")
        b_path = os.path.join(self.test_dir, "b")
        ab_path = os.path.join(self.test_dir, "ab")

        ab = io.open(ab_path, "wb", buffering=0)

        with open(a_path, "wb") as a, open(b_path, "wb") as b:
            a.write(b"1")
            b.write(b"2")
            tee = Tee(ab, a_path, b_path)
            a.write(b"3")
            b.write(b"4")

        tee.close()

        with open(ab_path, "rb") as f:
            self.assertCountEqual(f.read(), b"1234")
