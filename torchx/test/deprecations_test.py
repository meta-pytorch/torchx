# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for :py:mod:`torchx.deprecations`."""

import unittest
import warnings

from torchx.deprecations import (
    _PLUGIN_GROUPS,
    deprecated,
    deprecated_entrypoint,
    deprecated_module,
)


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


class DeprecatedEntrypointTest(unittest.TestCase):
    """Tests for ``deprecated_entrypoint()``."""

    def test_warns_for_scheduler_group(self) -> None:
        """Emits DeprecationWarning for the 'torchx.schedulers' group."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.schedulers", ["local_cwd", "slurm"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            1,
            "should emit exactly one DeprecationWarning for torchx.schedulers",
        )
        msg = str(dep_warnings[0].message)
        self.assertIn(
            "torchx.schedulers",
            msg,
            "warning should mention the deprecated group",
        )
        self.assertIn(
            "torchx_plugins.schedulers",
            msg,
            "warning should mention the namespace-package alternative",
        )
        self.assertIn(
            "TORCHX_NO_ENTRYPOINTS=1",
            msg,
            "warning should mention the opt-out env var",
        )

    def test_warns_for_named_resource_group(self) -> None:
        """Emits DeprecationWarning for the 'torchx.named_resources' group."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.named_resources", ["aws_p5"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            1,
            "should emit exactly one DeprecationWarning for torchx.named_resources",
        )

    def test_warns_for_tracker_group(self) -> None:
        """Emits DeprecationWarning for the 'torchx.tracker' group."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.tracker", ["mlflow"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            1,
            "should emit exactly one DeprecationWarning for torchx.tracker",
        )

    def test_no_warning_for_orchestrator_group(self) -> None:
        """No warning for 'torchx.schedulers.orchestrator' — no namespace alternative."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.schedulers.orchestrator", ["fblearner"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            0,
            "should not warn for groups without namespace-plugin alternatives",
        )

    def test_no_warning_for_components_group(self) -> None:
        """No warning for 'torchx.components' — no namespace alternative."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.components", ["my_component"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            0,
            "should not warn for torchx.components group",
        )

    def test_no_warning_for_file_group(self) -> None:
        """No warning for 'torchx.file' — no namespace alternative."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.file", ["get_file_contents"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            0,
            "should not warn for torchx.file group",
        )

    def test_warning_includes_sorted_plugin_names(self) -> None:
        """Warning message lists plugin names in sorted order."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.schedulers", ["zz_scheduler", "aa_scheduler"])

        msg = str(caught[0].message)
        self.assertIn(
            "aa_scheduler, zz_scheduler",
            msg,
            "plugin names should be sorted alphabetically in the warning",
        )

    def test_plugin_groups_matches_plugin_type(self) -> None:
        """_PLUGIN_GROUPS covers all PluginType values."""
        from torchx.plugins._registry import PluginType

        expected = {pt.value for pt in PluginType}
        self.assertEqual(
            _PLUGIN_GROUPS,
            expected,
            "_PLUGIN_GROUPS should match PluginType values exactly. "
            "If you added a new PluginType, add it to _PLUGIN_GROUPS too.",
        )
