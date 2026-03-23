# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for :py:mod:`torchx.deprecations`."""

import unittest
import warnings

from torchx.deprecations import deprecated, deprecated_module


class DeprecatedModuleTest(unittest.TestCase):
    """Tests for :func:`deprecated_module`."""

    def test_emits_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_module(
                old_import="torchx.old.module",
                new_import="torchx.new.module",
                stacklevel=1,
            )

        dep = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(dep), 1, "should emit exactly one UserWarning")

    def test_warning_message_contains_imports(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_module(
                old_import="torchx.old.path",
                new_import="torchx.new.path",
                stacklevel=1,
            )

        msg = str(caught[0].message)
        self.assertIn(
            "torchx.old.path",
            msg,
            "warning should mention old import path",
        )
        self.assertIn(
            "torchx.new.path",
            msg,
            "warning should mention new import path",
        )
        self.assertIn(
            "will be removed",
            msg,
            "warning should indicate future removal",
        )
        self.assertIn(
            "Deprecated",
            msg,
            "warning should clearly state this is a deprecation",
        )

    def test_no_warning_when_filtered(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("ignore", UserWarning)
            deprecated_module(
                old_import="torchx.a",
                new_import="torchx.b",
                stacklevel=1,
            )

        dep = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(dep), 0, "UserWarning should be suppressed when filtered")


class DeprecatedDecoratorTest(unittest.TestCase):
    """Tests for :func:`deprecated` decorator."""

    def test_emits_warning_on_call(self) -> None:
        @deprecated()
        def old_func() -> str:
            return "ok"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = old_func()

        self.assertEqual(result, "ok", "decorated function should still work")
        dep = [w for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(dep), 1, "should emit exactly one UserWarning")
        self.assertIn(
            "old_func",
            str(dep[0].message),
            "warning should mention the function name",
        )

    def test_replacement_in_message(self) -> None:
        @deprecated(replacement="new_func")
        def old_func() -> None:
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_func()

        msg = str(caught[0].message)
        self.assertIn(
            "new_func",
            msg,
            "warning should mention the replacement",
        )

    def test_preserves_function_metadata(self) -> None:
        @deprecated()
        def documented_func() -> None:
            """This is the docstring."""
            pass

        self.assertEqual(
            documented_func.__name__,
            "documented_func",
            "decorator should preserve __name__",
        )
        self.assertEqual(
            documented_func.__doc__,
            "This is the docstring.",
            "decorator should preserve __doc__",
        )

    def test_passes_args_and_kwargs(self) -> None:
        @deprecated()
        def add(a: int, b: int, extra: int = 0) -> int:
            return a + b + extra

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = add(1, 2, extra=3)

        self.assertEqual(result, 6, "decorated function should pass args correctly")
