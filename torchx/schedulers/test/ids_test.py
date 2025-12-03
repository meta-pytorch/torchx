#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import typing
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from torchx.schedulers.ids import make_unique, random_id, random_uint64


@contextmanager
def scoped_random_seed(seed: int) -> typing.Generator[None, None, None]:
    """
    Temporarily set the random module's seed and restore its state afterward.
    """
    import random

    state = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(state)


class IdsTest(unittest.TestCase):
    def test_make_unique(self) -> None:
        name = "test"
        self.assertNotEqual(make_unique(name), make_unique(name))
        size = 6
        self.assertNotEqual(make_unique(name, size), make_unique(name, size))

    def test_make_unique_min_len(self) -> None:
        unique_name = make_unique("test")
        # 16 chars in hex is 64 bits
        self.assertTrue(len(unique_name) >= len("test") + 5)
        self.assertTrue(unique_name.startswith("test-"))

    def test_random_uint64(self) -> None:
        self.assertGreater(random_uint64(), 0)
        self.assertNotEqual(random_uint64(), random_uint64())

    def test_random_id(self) -> None:
        ALPHAS = "abcdefghijklmnopqrstuvwxyz"
        v = random_id()
        self.assertIn(v[0], ALPHAS)
        self.assertGreater(len(v), 5)

    def test_random_id_max_length(self) -> None:
        for max_length in range(6, 10):
            with self.subTest(max_length=max_length):
                self.assertLessEqual(len(random_id(max_length)), max_length)
                self.assertNotEqual(random_id(max_length), random_id(max_length))

    def test_random_id_zero_max_length(self) -> None:
        self.assertEqual("", random_id(max_length=0))

    @patch("os.urandom", return_value=bytes(range(8)))
    def test_random_id_seed(self, urandom: MagicMock) -> None:
        self.assertEqual(random_id(), "fzfjxlmln9")
        self.assertEqual(random_id(max_length=6), "fzfjxl")

    @patch("os.urandom", return_value=bytes(range(8)))
    def test_make_unique_seed(self, urandom: MagicMock) -> None:
        self.assertEqual(make_unique("test"), "test-fzfjxlmln9")

    def test_make_unique_not_affected_by_random_seed(self) -> None:
        # Seeding the Python random module should not affect make_unique(),
        # which relies on os.urandom for entropy.
        with scoped_random_seed(0):
            v1 = make_unique("test")

        with scoped_random_seed(0):
            v2 = make_unique("test")

        # Even with the same random seed, make_unique should produce different values.
        self.assertNotEqual(v1, v2)
        self.assertTrue(v1.startswith("test-"))
        self.assertTrue(v2.startswith("test-"))
