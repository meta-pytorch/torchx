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
import warnings
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

from torchx import plugins
from torchx.specs import (
    NULL_RESOURCE,
    Resource,
    _NamedResourcesLibrary,
    get_named_resources,
    named_resources,
    resource,
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
            lib._lookup("foo")

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `p316xl`. Did you mean `p3.16xlarge`?",
        ):
            lib._lookup("p316xl")

    def test_null_and_missing_named_resources(self) -> None:
        self.assertEqual(resource(h="NULL"), NULL_RESOURCE)
        self.assertEqual(resource(h="MISSING"), NULL_RESOURCE)

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


class OneLookupPathTest(unittest.TestCase):
    def test_lookup_is_case_insensitive(self) -> None:
        expected = resource(h="aws_t3.medium")
        for name in ["aws_t3.medium", "AWS_T3.MEDIUM", "Aws_T3.Medium"]:
            with self.subTest(name=name):
                self.assertEqual(expected, resource(h=name))

    def test_exact_name_wins_over_a_case_insensitive_match(self) -> None:
        shouty = Resource(cpu=9, gpu=9, memMB=9)
        lib = _NamedResourcesLibrary()
        lib._factories = {"gpu_x2": mock_resource, "GPU_X2": lambda: shouty}

        self.assertEqual(mock_resource(), lib._lookup("gpu_x2"))
        self.assertEqual(shouty, lib._lookup("GPU_X2"))

    def test_ambiguous_case_insensitive_match_is_an_error(self) -> None:
        lib = _NamedResourcesLibrary()
        lib._factories = {"gpu_x2": mock_resource, "GPU_X2": mock_resource}

        with self.assertRaisesRegex(
            KeyError,
            r"`Gpu_X2` matches more than one registered named resource,"
            r" ignoring case: \['GPU_X2', 'gpu_x2'\]",
        ):
            lib._lookup("Gpu_X2")

    def test_contains_is_case_insensitive(self) -> None:
        lib = _NamedResourcesLibrary()
        lib._factories = {"gpu_x2": mock_resource}

        self.assertIn("GPU_X2", lib)
        self.assertNotIn("gpu_x4", lib)

    def test_subscript_warns_and_forwards(self) -> None:
        with self.assertWarnsRegex(
            FutureWarning, r"`named_resources\[name\]` is deprecated"
        ):
            got = named_resources["AWS_T3.MEDIUM"]

        self.assertEqual(resource(h="aws_t3.medium"), got)

    def test_get_named_resources_warns_and_forwards(self) -> None:
        with self.assertWarnsRegex(
            FutureWarning, r"`get_named_resources\(\)` is deprecated"
        ):
            got = get_named_resources("AWS_T3.MEDIUM")

        self.assertEqual(resource(h="aws_t3.medium"), got)

    def test_resource_is_not_deprecated(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            resource(h="aws_t3.medium")


class PluginRoundTripTest(unittest.TestCase):
    def tearDown(self) -> None:
        for k in [k for k in sys.modules if k.startswith("torchx_plugins")]:
            del sys.modules[k]

    def test_registered_names_round_trip_through_resource(self) -> None:
        with patch("sys.path", [_LAZY_FIXTURE_DIR, *sys.path]):
            plugins.registry().clear()
            named_resources.reset()
            try:
                names = list(named_resources.keys())
                self.assertIn(
                    "reentrant_gpu",
                    names,
                    "the plugin-registered resource must be listed",
                )
                for name in names:
                    if name in ("NULL", "MISSING"):
                        continue
                    with self.subTest(name=name):
                        self.assertEqual(
                            name,
                            resource(h=name).get_resource_name(),
                            "every registered name must resolve to its own resource",
                        )
                self.assertEqual(
                    resource(h="reentrant_gpu"),
                    resource(h="REENTRANT_GPU"),
                    "a plugin name must resolve ignoring case too",
                )
            finally:
                plugins.registry().clear()
                named_resources.reset()


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
                    resource(h="NULL"),
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
