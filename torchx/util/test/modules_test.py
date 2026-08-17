# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

from torchx.util.modules import import_attr, load_module


class ModulesTest(unittest.TestCase):
    def _write_module_on_sys_path(self, name: str, body: str) -> None:
        """Writes ``{name}.py`` with ``body`` into a tmp dir prepended to ``sys.path``."""
        tmpdir: "tempfile.TemporaryDirectory[str]" = tempfile.TemporaryDirectory(
            prefix="torchx_modules_test"
        )
        self.addCleanup(tmpdir.cleanup)
        (Path(tmpdir.name) / f"{name}.py").write_text(body)
        sys.path.insert(0, tmpdir.name)
        importlib.invalidate_caches()

        # LIFO: runs before tmpdir.cleanup
        def cleanup() -> None:
            sys.path.remove(tmpdir.name)
            sys.modules.pop(name, None)

        self.addCleanup(cleanup)

    def test_load_module(self) -> None:
        result = load_module("os.path")
        import os

        self.assertEqual(result, os.path)

    def test_load_module_method(self) -> None:
        result = load_module("os.path:join")
        import os

        self.assertEqual(result, os.path.join)

    def test_load_module_only_splits_on_first_colon(self) -> None:
        self.assertIsNone(
            load_module("os.path:join:extra"),
            "everything after the first `:` is the attr name;"
            " `join:extra` must not silently resolve to `join`",
        )

    def test_load_module_absent_module_is_silent(self) -> None:
        with self.assertNoLogs("torchx.util.modules", level="WARNING"):
            self.assertIsNone(
                load_module("non.existent.module"),
                "an absent module must load as None",
            )

    def test_load_module_broken_module_logs_warning(self) -> None:
        self._write_module_on_sys_path(
            "torchx_test_broken_mod", "raise RuntimeError('boom')\n"
        )
        with self.assertLogs("torchx.util.modules", level="WARNING") as logs:
            self.assertIsNone(
                load_module("torchx_test_broken_mod"),
                "a broken module must load as None",
            )
        self.assertIn(
            "torchx_test_broken_mod",
            "\n".join(logs.output),
            "the warning must name the module that failed to load",
        )

    def test_load_module_broken_dep_module_logs_warning(self) -> None:
        self._write_module_on_sys_path(
            "torchx_test_broken_dep_mod2", "import torchx_nonexistent_dependency\n"
        )
        with self.assertLogs("torchx.util.modules", level="WARNING") as logs:
            self.assertIsNone(
                load_module("torchx_test_broken_dep_mod2"),
                "a module that fails importing its own deps must load as None",
            )
        self.assertIn(
            "torchx_nonexistent_dependency",
            "\n".join(logs.output),
            "the warning must name the missing dependency, not read as absence",
        )

    def test_try_import(self) -> None:
        def _join(_0: str, *_1: str) -> str:
            return ""  # should never be called

        os_path_join = import_attr("os.path", "join", default=_join)
        import os

        self.assertEqual(os.path.join, os_path_join)

    def test_try_import_non_existent_module(self) -> None:
        should_default = import_attr("non.existent", "foo", default="bar")
        self.assertEqual("bar", should_default)

    def test_try_import_broken_module_raises(self) -> None:
        self._write_module_on_sys_path(
            "torchx_test_broken_dep_mod", "import torchx_nonexistent_dependency\n"
        )
        with self.assertRaises(
            ModuleNotFoundError, msg="a broken module is not an absent module"
        ):
            import_attr("torchx_test_broken_dep_mod", "foo", default="bar")

    def test_try_import_non_existent_attr(self) -> None:
        def _join(_0: str, *_1: str) -> str:
            return ""  # should never be called

        with self.assertRaises(AttributeError):
            import_attr("os.path", "joyin", default=_join)
