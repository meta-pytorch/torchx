#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

from torchx import plugins
from torchx.specs import (
    _NamedResourcesLibrary,
    named_resources,
    NULL_RESOURCE,
    Resource,
)

# A namespace plugin package whose module imports torchx.specs at top-level.
_LAZY_FIXTURE_DIR: str = str(Path(__file__).resolve().parent / "lazy_fixture")

# A namespace plugin package whose module LOOKS UP a named resource at
# top-level, re-entering discovery mid-scan.
_REENTRANT_LOOKUP_FIXTURE_DIR: str = str(
    Path(__file__).resolve().parent / "reentrant_lookup_fixture"
)


def mock_resource() -> Resource:
    return Resource(cpu=0, gpu=0, memMB=0)


class NamedResourcesTest(unittest.TestCase):
    def test_named_resources_library(self) -> None:
        lib = _NamedResourcesLibrary()
        factories: dict[str, Callable[[], Resource]] = {}
        for name in ["p3.2xlarge", "p3.16xlarge", "p4d.24xlarge"]:
            factories[name] = mock_resource
        lib._factories = factories

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `foo`. Registered named resources:.*",
        ):
            _ = lib["foo"]

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `p316xl`. Did you mean `p3.16xlarge`?",
        ):
            _ = lib["p316xl"]

    def test_null_and_missing_named_resources(self) -> None:
        self.assertEqual(named_resources["NULL"], NULL_RESOURCE)
        self.assertEqual(named_resources["MISSING"], NULL_RESOURCE)

    def test_keys_and_items(self) -> None:
        lib = _NamedResourcesLibrary()
        lib._factories = {"p3.2xlarge": mock_resource}

        self.assertEqual({"p3.2xlarge"}, set(lib.keys()), "keys() must list names")
        self.assertEqual(
            [("p3.2xlarge", mock_resource())],
            list(lib.items()),
            "items() must materialize each resource",
        )

    def test_lazy_load_and_reset(self) -> None:
        lib = _NamedResourcesLibrary()
        self.assertIsNone(lib._factories, "no discovery may run before first access")

        self.assertIn("NULL", lib)

        self.assertIsNotNone(lib._factories, "first access must populate the cache")
        lib.reset()
        self.assertIsNone(lib._factories, "reset() must drop the cache")

    def test_custom_named_resources_env_var(self) -> None:
        mock_module = type(sys)("test_module")
        mock_module.NAMED_RESOURCES = {"test_resource": mock_resource}

        with patch.dict(sys.modules, {"test_module": mock_module}):
            with patch.dict(
                os.environ, {"TORCHX_CUSTOM_NAMED_RESOURCES": "test_module"}
            ):
                lib = _NamedResourcesLibrary()
                self.assertIn("test_resource", lib)


class LazyDiscoveryTest(unittest.TestCase):
    def tearDown(self) -> None:
        for k in [k for k in sys.modules if k.startswith("torchx_plugins")]:
            del sys.modules[k]

    def test_import_torchx_specs_performs_no_discovery(self) -> None:
        """`import torchx.specs` must not import plugins or resource modules."""
        code = "; ".join(
            [
                "import sys",
                "import torchx.specs",
                "mods = [m for m in sys.modules"
                " if m.startswith('torchx_plugins')"
                " or m == 'torchx.specs.named_resources_aws'"
                " or m == 'torchx.specs.named_resources_generic']",
                "assert not mods, f'import torchx.specs triggered discovery: {mods}'",
            ]
        )
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )

    def test_first_lookup_triggers_discovery(self) -> None:
        lib = _NamedResourcesLibrary()
        with patch.object(plugins, "registry") as registry_mock:
            registry_mock.return_value.get.return_value = {}
            self.assertIsNone(
                lib._factories, "instantiation must not trigger discovery"
            )
            registry_mock.assert_not_called()

            self.assertIn("NULL", lib)

            registry_mock.return_value.get.assert_called_once_with(
                plugins.PluginType.NAMED_RESOURCE
            )

    def test_reentrant_plugin_import_scans_clean(self) -> None:
        """A namespace plugin importing torchx.specs at module top-level is
        discovered cleanly when discovery is triggered from torchx.specs."""
        lib = _NamedResourcesLibrary()
        with patch("sys.path", [_LAZY_FIXTURE_DIR, *sys.path]):
            plugins.registry().clear()
            try:
                self.assertIn(
                    "reentrant_gpu",
                    lib,
                    "plugin with a top-level torchx.specs import must be discovered",
                )
                self.assertEqual(
                    [],
                    plugins.registry().load_errors(plugins.PluginType.NAMED_RESOURCE),
                    "re-entrant torchx.specs import must scan clean",
                )
            finally:
                plugins.registry().clear()

    def test_reentrant_lookup_at_import_is_a_load_error(self) -> None:
        """A plugin looking up a named resource at import time re-enters
        ``_load`` mid-scan — pinned behavior: the lookup raises
        ``RuntimeError``, the scanner records the module as a load error,
        and the outer scan still completes and caches."""
        with patch("sys.path", [_REENTRANT_LOOKUP_FIXTURE_DIR, *sys.path]):
            plugins.registry().clear()
            named_resources.reset()
            try:
                self.assertNotIn(
                    "lookup_at_import_gpu",
                    named_resources,
                    "a plugin whose import fails must not be registered",
                )
                errors = plugins.registry().load_errors(
                    plugins.PluginType.NAMED_RESOURCE
                )
                self.assertEqual(
                    1,
                    len(errors),
                    f"expected exactly the fixture's load error, got: {errors}",
                )
                self.assertIn(
                    "re-entrant named-resource lookup",
                    errors[0].error,
                    "the load error must carry the re-entrancy diagnostic",
                )
                self.assertEqual(
                    NULL_RESOURCE,
                    named_resources["NULL"],
                    "the outer scan must complete despite the broken plugin",
                )
            finally:
                plugins.registry().clear()
                named_resources.reset()

    def test_concurrent_first_lookups_load_once(self) -> None:
        lib: _NamedResourcesLibrary = _NamedResourcesLibrary()
        registry_calls: list[int] = []

        def slow_registry() -> MagicMock:
            registry_calls.append(1)
            time.sleep(0.1)  # widen the check-then-set window
            mock = MagicMock()
            mock.get.return_value = {}
            return mock

        n = 8
        barrier: threading.Barrier = threading.Barrier(n)
        results: list[bool] = []

        def lookup() -> None:
            barrier.wait()
            results.append("NULL" in lib)

        with patch.object(plugins, "registry", side_effect=slow_registry):
            threads = [threading.Thread(target=lookup) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual([True] * n, results, "every lookup must see the loaded set")
        self.assertEqual(
            1,
            len(registry_calls),
            "concurrent first lookups must run discovery exactly once",
        )
