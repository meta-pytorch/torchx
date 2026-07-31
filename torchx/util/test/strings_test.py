# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx.util.strings import normalize_str


class StringsTest(unittest.TestCase):
    def test_normalize_str(self) -> None:
        self.assertEqual("abcd123", normalize_str("abcd123"))
        self.assertEqual("abcd123", normalize_str("-/_a/b/CD!123!"))
        self.assertEqual("a-bcd123", normalize_str("-a-bcd123"))
        self.assertEqual("", normalize_str("!!!"))
