# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx.util.modules import import_attr, load_module


class ModulesTest(unittest.TestCase):
    def test_load_module(self) -> None:
        result = load_module("os.path")
        import os

        self.assertEqual(result, os.path)

    def test_load_module_method(self) -> None:
        result = load_module("os.path:join")
        import os

        self.assertEqual(result, os.path.join)

    def test_try_import(self) -> None:
        def _join(_0: str, *_1: str) -> str:
            return ""  # should never be called

        os_path_join = import_attr("os.path", "join", default=_join)
        import os

        self.assertEqual(os.path.join, os_path_join)

    def test_try_import_non_existent_module(self) -> None:
        should_default = import_attr("non.existent", "foo", default="bar")
        self.assertEqual("bar", should_default)

    def test_try_import_non_existent_attr(self) -> None:
        def _join(_0: str, *_1: str) -> str:
            return ""  # should never be called

        with self.assertRaises(AttributeError):
            import_attr("os.path", "joyin", default=_join)
