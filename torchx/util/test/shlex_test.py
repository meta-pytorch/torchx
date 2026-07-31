# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx.util import shlex


class ShlexTest(unittest.TestCase):
    def test_join(self) -> None:
        self.assertEqual(shlex.join(["foo", "foo bar", "$yes"]), "foo 'foo bar' '$yes'")
