#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import os.path
import tempfile
import unittest

import fsspec
from torchx.apps.utils.copy_main import main


class CopyTest(unittest.TestCase):
    def _copy(self, src: str, dst: str) -> None:
        data = os.urandom(200 * 1024)  # more than the batch size
        with fsspec.open(src, "wb") as f:
            f.write(data)

        main(
            [
                "--src",
                src,
                "--dst",
                dst,
            ]
        )

        with fsspec.open(dst, "rb") as f:
            out = f.read()
            self.assertEqual(out, data)

    def test_same_fs(self) -> None:
        src = "memory://foo"
        dst = "memory://bar"
        self._copy(src, dst)

    def test_different_fs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = "memory://foo"
            dst = "file://" + os.path.join(tmpdir, "foo", "bar")

            self._copy(src, dst)
