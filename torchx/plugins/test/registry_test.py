# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[6, 13, 16]

"""Tests for :py:mod:`torchx.plugins._registry`.

Covers namespace mapping, ``registry()`` / ``PluginRegistry`` caching and merge
priority, named resource discovery, and fault-tolerance of discovery against
broken plugin modules.
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from torchx.plugins._registration import register
from torchx.plugins._registry import (
    PluginRegistry,
    PluginType,
    RegistrationError,
    registry,
)
from torchx.util import entrypoints


# ── Test namespace package setup ─────────────────────────────────────────────

# Four directories simulate four independently pip-installed packages
# that all contribute to the ``torchx_plugins`` PEP 420 namespace:
#
#   test/default/      → torchx-plugins-default (local/docker schedulers & generic resources)
#   test/aws/          → torchx-plugins-aws     (AWS-specific plugins)
#   test/gcp/          → torchx-plugins-gcp     (GCP-specific plugins)
#   test/bad/          → torchx-plugins-bad     (broken/mis-registered plugins for error testing)
_TEST_DIR: Path = Path(__file__).resolve().parent
_TEST_PLUGINS_DIRS: list[str] = [
    str(_TEST_DIR / "default"),
    str(_TEST_DIR / "aws"),
    str(_TEST_DIR / "gcp"),
    str(_TEST_DIR / "bad"),
]


def mock_install_torchx_plugins() -> Any:  # pyre-ignore[3]: returns mock._patch
    """Mock-install the test ``torchx_plugins`` namespace package.

    Temporarily prepends four test directories to ``sys.path`` so
    ``importlib`` and ``pkgutil`` discover ``torchx_plugins/`` as a
    PEP 420 namespace package spanning multiple distributions.  Works as
    a context manager or a test-method decorator.
    """
    return patch("sys.path", [*_TEST_PLUGINS_DIRS, *sys.path])


# ── Base class ───────────────────────────────────────────────────────────────


class _RegistryTestBase(unittest.TestCase):
    """Common setUp/tearDown for tests that exercise plugin discovery."""

    def setUp(self) -> None:
        registry.cache_clear()

    def tearDown(self) -> None:
        to_remove = [k for k in sys.modules if k.startswith("torchx_plugins")]
        for k in to_remove:
            del sys.modules[k]
        registry.cache_clear()


# ── registry() / PluginRegistry tests ────────────────────────────────────────


class RegistryTest(_RegistryTestBase):
    """Tests ``registry()`` and ``PluginRegistry`` caching, discovery, and merge priority."""

    def test_returns_empty_when_not_installed(self) -> None:
        reg = PluginRegistry(load_entrypoints=False)
        result = reg.get(PluginType.SCHEDULER)
        self.assertEqual(
            result,
            {},
            "should return empty dict when nothing is installed",
        )

    def test_namespace_for_type(self) -> None:
        """PluginType → torchx_plugins.* namespace mapping."""
        self.assertEqual(
            PluginRegistry._namespace_for_type(PluginType.SCHEDULER),
            "torchx_plugins.schedulers",
            "should map SCHEDULER to torchx_plugins.schedulers",
        )
        self.assertEqual(
            PluginRegistry._namespace_for_type(PluginType.NAMED_RESOURCE),
            "torchx_plugins.named_resources",
            "should map NAMED_RESOURCE to torchx_plugins.named_resources",
        )
        self.assertEqual(
            PluginRegistry._namespace_for_type(PluginType.TRACKER),
            "torchx_plugins.tracker",
            "should map TRACKER to torchx_plugins.tracker",
        )

    @mock_install_torchx_plugins()
    def test_discovers_decorated_plugins(self) -> None:
        reg = PluginRegistry(load_entrypoints=False)
        result = reg.get(PluginType.SCHEDULER)

        self.assertIn(
            "local_cwd",
            result,
            "should discover local_cwd from @register.scheduler()",
        )
        self.assertIn(
            "local_docker",
            result,
            "should discover local_docker from @register.scheduler()",
        )
        self.assertTrue(
            callable(result["local_cwd"]),
            "get() should return callables",
        )
        sched = result["local_cwd"]("sess")
        self.assertEqual(
            sched.session_name,
            "sess",
            "discovered scheduler factory should pass session_name through",
        )

    @mock_install_torchx_plugins()
    def test_skips_private_modules(self) -> None:
        reg = PluginRegistry(load_entrypoints=False)
        result = reg.get(PluginType.SCHEDULER)

        # _private.py should have been skipped.
        self.assertNotIn(
            "_private",
            result,
            "should skip modules starting with underscore",
        )
        self.assertNotIn(
            "SECRET",
            result,
            "contents of private modules should not leak into results",
        )

    def test_returns_cached_registry(self) -> None:
        """Repeated registry() calls return the same PluginRegistry instance."""
        first = registry()
        second = registry()
        self.assertIs(
            first,
            second,
            "registry() should return the cached PluginRegistry",
        )

    @mock_install_torchx_plugins()
    def test_get_returns_cached_dict(self) -> None:
        """Second .get() call returns the same dict object (lazy cache)."""
        reg = PluginRegistry(load_entrypoints=False)
        first = reg.get(PluginType.SCHEDULER)
        second = reg.get(PluginType.SCHEDULER)

        self.assertIs(
            first,
            second,
            "get() should return the cached result on second call",
        )

    @mock_install_torchx_plugins()
    def test_clear_resets_cache(self) -> None:
        """registry().clear() discards cached discovery results."""
        with patch.object(entrypoints, "load_group", return_value=None):
            reg = registry()
            first = reg.get(PluginType.SCHEDULER)
            reg.clear()
            second = reg.get(PluginType.SCHEDULER)

        self.assertIsNot(
            first,
            second,
            "after clear(), get() should return a freshly discovered dict",
        )
        self.assertEqual(
            set(first),
            set(second),
            "both calls should discover the same set of plugins",
        )

    @mock_install_torchx_plugins()
    def test_entry_points_take_priority(self) -> None:
        ep_result = {
            "local_cwd": "ep_version",
            "ep_only": "ep_only_value",
        }

        with patch.object(entrypoints, "load_group", return_value=ep_result):
            result = registry().get(PluginType.SCHEDULER)

        self.assertEqual(
            result["local_cwd"],
            "ep_version",
            "entry point should override namespace plugin with same name",
        )
        self.assertIn("ep_only", result, "entry point-only key should be present")
        # Namespace plugins fill gaps.
        self.assertIn(
            "local_docker",
            result,
            "namespace plugin should fill gaps not covered by entry points",
        )

    @mock_install_torchx_plugins()
    def test_load_entrypoints_false(self) -> None:
        """load_entrypoints=False skips entry-point loading."""
        ep_result = {"ep_only": "ep_value"}

        with patch.object(entrypoints, "load_group", return_value=ep_result):
            reg = PluginRegistry(load_entrypoints=False)
            result = reg.get(PluginType.SCHEDULER)

        self.assertNotIn(
            "ep_only",
            result,
            "should not include entry-point-only plugins when load_entrypoints=False",
        )
        self.assertIn(
            "local_cwd",
            result,
            "namespace plugins should still be discovered",
        )

    @mock_install_torchx_plugins()
    def test_info_returns_all_groups(self) -> None:
        """info() returns dict[PluginType, dict[str, Callable]]."""
        reg = PluginRegistry(load_entrypoints=False)
        all_plugins = reg.info()

        self.assertIsInstance(all_plugins, dict, "info() should return a dict")
        # All PluginType keys should be present.
        for pt in PluginType:
            self.assertIn(pt, all_plugins, f"info() should include {pt}")
            self.assertIsInstance(
                all_plugins[pt], dict, f"info()[{pt}] should be a dict"
            )

        # Check specific plugins.
        scheds = all_plugins[PluginType.SCHEDULER]
        self.assertIn("local_cwd", scheds, "should include local_cwd scheduler")

        named_res = all_plugins[PluginType.NAMED_RESOURCE]
        self.assertIn("gpu", named_res, "should include gpu named resource")

        # Verify module is accessible from the callable.
        local = scheds["local_cwd"]
        self.assertEqual(
            local.__module__,
            "torchx_plugins.schedulers.local",
            "should come from local module",
        )

    @mock_install_torchx_plugins()
    def test_info_single_type(self) -> None:
        """info(PluginType) returns dict[str, Callable] for that group."""
        reg = PluginRegistry(load_entrypoints=False)
        scheds = reg.info(PluginType.SCHEDULER)

        self.assertIsInstance(scheds, dict, "info(type) should return a dict")
        self.assertIn("local_cwd", scheds, "should include local_cwd")

        # Defensive copy — modifying the returned dict shouldn't affect cache.
        scheds["injected"] = scheds["local_cwd"]
        self.assertNotIn(
            "injected",
            reg.get(PluginType.SCHEDULER),
            "info() should return a defensive copy",
        )

    @mock_install_torchx_plugins()
    def test_str(self) -> None:
        """Snapshot of str(registry) YAML output with test fixtures."""
        reg = PluginRegistry(load_entrypoints=False)
        output = str(reg)

        expected = """\
scheduler:
  - name: local_cwd
    module: torchx_plugins.schedulers.local
  - name: local_docker
    module: torchx_plugins.schedulers.local
  - name: aws_k8s
    module: torchx_plugins.schedulers.aws.k8s
  - name: gcp_k8s
    module: torchx_plugins.schedulers.gcp.k8s
named_resource:
  - name: cpu
    module: torchx_plugins.named_resources.generic
    fractionals: [cpu_16, cpu_32, cpu_64, cpu_8]
  - name: gpu
    module: torchx_plugins.named_resources.generic
    aliases: [t4g]
    fractionals: [gpu_1, gpu_2, gpu_4, gpu_8]
  - name: aws_p5_48xlarge
    module: torchx_plugins.named_resources.aws
  - name: gcp_a3_highgpu_8g
    module: torchx_plugins.named_resources.gcp
  - name: implicit_gpu
    module: torchx_plugins.named_resources.implicit_sub.hardware
tracker:
  - name: mlflow
    module: torchx_plugins.schedulers.mlflow
    error: "is a tracker but is under the scheduler namespace \u2014 use @register.scheduler() or move to `torchx_plugins.tracker`"
errors:
  - module: torchx_plugins.schedulers.airflow
    error: "RuntimeError: cannot reach Airflow REST API at http://localhost:8080"
  - module: torchx_plugins.schedulers.ray
    error: "ModuleNotFoundError: No module named 'ray'"
""".strip()
        self.assertEqual(
            output,
            expected,
            f"str(registry) does not match expected snapshot.\n\nActual:\n{output}",
        )

    @mock_install_torchx_plugins()
    def test_str_is_loadable_yaml(self) -> None:
        """str(registry) is valid YAML that round-trips through to_dict()."""
        import yaml

        reg = PluginRegistry(load_entrypoints=False)
        output = str(reg)
        expected = reg.to_dict()

        loaded = yaml.safe_load(output)
        self.assertEqual(
            loaded,
            expected,
            f"str(registry) should round-trip through to_dict().\n\nOutput:\n{output}",
        )

    @mock_install_torchx_plugins()
    def test_to_dict(self) -> None:
        """to_dict() returns the expected structure."""
        reg = PluginRegistry(load_entrypoints=False)
        data = reg.to_dict()

        # Schedulers.
        sched_names = [p["name"] for p in data["scheduler"]]
        self.assertIn("local_cwd", sched_names, "should include local_cwd scheduler")
        self.assertIn("aws_k8s", sched_names, "should include aws_k8s scheduler")

        # Named resources — base plugins only (children are collapsed).
        nr_names = [p["name"] for p in data["named_resource"]]
        self.assertIn("gpu", nr_names, "should include gpu named resource")
        self.assertIn("cpu", nr_names, "should include cpu named resource")
        self.assertNotIn(
            "gpu_8", nr_names, "fractionals should not be top-level entries"
        )

        # gpu should have aliases and fractionals.
        gpu = next(p for p in data["named_resource"] if p["name"] == "gpu")
        self.assertEqual(gpu["aliases"], ["t4g"], "gpu should have t4g alias")
        self.assertEqual(
            gpu["fractionals"],
            ["gpu_1", "gpu_2", "gpu_4", "gpu_8"],
            "gpu should have 4 fractionals",
        )

        # Plugin-level errors merge into their actual type's section.
        # Import-level errors (airflow.py, ray.py) stay in errors:.
        self.assertEqual(
            len(data["errors"]),
            2,
            "should have two import-level errors (airflow.py RuntimeError, ray.py ImportError)",
        )
        error_modules = [e["module"] for e in data["errors"]]
        self.assertIn(
            "torchx_plugins.schedulers.airflow",
            error_modules,
            "should capture RuntimeError from airflow.py",
        )
        self.assertIn(
            "torchx_plugins.schedulers.ray",
            error_modules,
            "should capture ImportError from ray.py",
        )
        # mlflow tracker was in schedulers/ — should appear in tracker section.
        tracker_names = [p["name"] for p in data["tracker"]]
        self.assertIn(
            "mlflow", tracker_names, "mlflow error should be in tracker section"
        )
        mlflow = next(p for p in data["tracker"] if p["name"] == "mlflow")
        self.assertIn("error", mlflow, "mlflow entry should have an error field")


# ── Named resource discovery tests ───────────────────────────────────────────


class NamedResourceDiscoveryTest(_RegistryTestBase):
    """Tests named resource discovery via ``registry().get(PluginType.NAMED_RESOURCE)``."""

    def test_returns_empty_when_not_installed(self) -> None:
        reg = PluginRegistry(load_entrypoints=False)
        result = reg.get(PluginType.NAMED_RESOURCE)
        self.assertEqual(
            result,
            {},
            "should return empty dict when torchx_plugins.named_resources is not installed",
        )

    @mock_install_torchx_plugins()
    def test_discovers_named_resources_from_multiple_packages(self) -> None:
        """End-to-end named resource discovery across multiple packages.

        Covers: base resources, fractionals, cross-package namespace merging,
        and fractional resource value correctness.
        """
        reg = PluginRegistry(load_entrypoints=False)
        result = reg.get(PluginType.NAMED_RESOURCE)

        # --- generic.py: gpu resource (powers_of_two_gpus fractionals) ---
        self.assertIn("gpu", result, "should discover gpu from generic.py")
        self.assertIn("gpu_8", result, "should discover fractional gpu_8 (8 GPUs)")
        self.assertIn("gpu_4", result, "should discover fractional gpu_4 (4 GPUs)")
        self.assertIn("gpu_2", result, "should discover fractional gpu_2 (2 GPUs)")
        self.assertIn("gpu_1", result, "should discover fractional gpu_1 (1 GPU)")

        res_gpu = result["gpu"]()
        self.assertEqual(res_gpu.gpu, 8, "gpu (whole) should have 8 GPUs")
        self.assertEqual(res_gpu.cpu, 64, "gpu (whole) should have 64 CPUs")

        res_gpu_4 = result["gpu_4"]()
        self.assertEqual(res_gpu_4.gpu, 4, "gpu_4 (half) should produce 4 GPUs")
        self.assertEqual(res_gpu_4.cpu, 32, "gpu_4 should have 32 CPUs (half of 64)")

        # --- generic.py: cpu resource (halve_mem_down_to fractionals) ---
        self.assertIn("cpu", result, "should discover cpu from generic.py")
        self.assertIn("cpu_64", result, "should discover fractional cpu_64 (64 GiB)")
        self.assertIn("cpu_32", result, "should discover fractional cpu_32 (32 GiB)")
        self.assertIn("cpu_16", result, "should discover fractional cpu_16 (16 GiB)")
        self.assertIn("cpu_8", result, "should discover fractional cpu_8 (8 GiB)")
        self.assertNotIn("cpu_4", result, "minGiB=8 should stop at 8 GiB")

        res_cpu = result["cpu"]()
        self.assertEqual(res_cpu.cpu, 16, "cpu should have 16 CPUs")
        self.assertEqual(res_cpu.gpu, 2, "cpu should have 2 GPUs")

        res_cpu_32 = result["cpu_32"]()
        self.assertEqual(res_cpu_32.cpu, 8, "cpu_32 (half) should have 8 CPUs")
        self.assertEqual(res_cpu_32.gpu, 1, "cpu_32 (half) should have 1 GPU")

        # --- Cross-package: aws and gcp via namespace package merging ---
        res_aws = result["aws_p5_48xlarge"]()
        self.assertEqual(res_aws.cpu, 192, "aws_p5_48xlarge should have 192 CPUs")
        self.assertEqual(res_aws.gpu, 8, "aws_p5_48xlarge should have 8 GPUs")

        res_gcp = result["gcp_a3_highgpu_8g"]()
        self.assertEqual(res_gcp.cpu, 252, "gcp_a3_highgpu_8g should have 252 CPUs")

        # --- implicit_sub/hardware.py: no __init__.py in implicit_sub/ ---
        self.assertIn(
            "implicit_gpu",
            result,
            "should discover implicit_gpu from a subdirectory without __init__.py",
        )
        res_implicit = result["implicit_gpu"]()
        self.assertEqual(res_implicit.gpu, 4, "implicit_gpu should have 4 GPUs")


# ── Fault-tolerance tests ────────────────────────────────────────────────────


class DiscoveryFaultToleranceTest(_RegistryTestBase):
    """Verify that broken plugin modules are skipped rather than crashing discovery.

    The ``torchx-plugins-bad`` test package (``test/bad/``) contains real
    fixtures that trigger every class of import-time error:

    - ``airflow.py``  — ``RuntimeError`` at module init
    - ``ray.py``      — ``ImportError`` (missing dependency)
    - ``mlflow.py``   — type mismatch (tracker registered under schedulers/)

    A single bad plugin package should not prevent TorchX from loading.
    The discovery system should capture errors and continue scanning.
    """

    @mock_install_torchx_plugins()
    def test_bad_plugins_do_not_prevent_good_plugins(self) -> None:
        """Broken plugins in torchx-plugins-bad don't crash healthy discovery."""
        reg = PluginRegistry(load_entrypoints=False)
        scheds = reg.get(PluginType.SCHEDULER)

        # Good plugins from torchx-plugins-default still work.
        self.assertIn(
            "local_cwd",
            scheds,
            "healthy plugins should still be discovered alongside broken ones",
        )
        self.assertIn(
            "local_docker",
            scheds,
            "all healthy schedulers should be present",
        )

    @mock_install_torchx_plugins()
    def test_errors_captured_from_bad_package(self) -> None:
        """All error types from torchx-plugins-bad are captured in errors list."""
        reg = PluginRegistry(load_entrypoints=False)
        reg.info()  # trigger full discovery

        error_modules = {e.module for e in reg.errors}

        # RuntimeError at import (airflow.py).
        self.assertIn(
            "torchx_plugins.schedulers.airflow",
            error_modules,
            "should capture RuntimeError from airflow.py",
        )

        # ImportError — missing dep (ray.py).
        self.assertIn(
            "torchx_plugins.schedulers.ray",
            error_modules,
            "should capture ImportError from ray.py",
        )

        # Type mismatch — tracker under schedulers/ (mlflow.py).
        mlflow_errors = [
            e for e in reg.errors if e.module == "torchx_plugins.schedulers.mlflow"
        ]
        self.assertEqual(
            len(mlflow_errors),
            1,
            "should capture type-mismatch error from mlflow.py",
        )
        self.assertEqual(
            mlflow_errors[0].plugin_type,
            "tracker",
            "mlflow error should indicate it's a tracker",
        )

        # Every error is a RegistrationError with required fields.
        for err in reg.errors:
            self.assertIsInstance(
                err,
                RegistrationError,
                f"error for {err.module} should be a RegistrationError",
            )

    @mock_install_torchx_plugins()
    def test_cross_module_name_conflict_is_skipped(self) -> None:
        """A cross-module plugin name conflict keeps the first occurrence.

        This requires mocking because you can't have two files with the
        same plugin name in a single namespace directory.
        """
        dummy_mod = types.ModuleType("torchx_plugins.schedulers.conflict")
        dummy_mod.__name__ = "torchx_plugins.schedulers.conflict"
        sys.modules["torchx_plugins.schedulers.conflict"] = dummy_mod

        def fake_local_cwd(session_name: str) -> str:
            return "fake"

        fake_local_cwd.__module__ = "torchx_plugins.schedulers.conflict"
        register.scheduler(name="local_cwd")(fake_local_cwd)
        dummy_mod.fake_local_cwd = fake_local_cwd  # type: ignore[attr-defined]

        try:
            reg = PluginRegistry(load_entrypoints=False)
            result = reg._find_namespace_plugins(
                "torchx_plugins.schedulers",
                expected_type=PluginType.SCHEDULER,
            )

            self.assertIn(
                "local_cwd",
                result,
                "first-discovered local_cwd should remain",
            )
        finally:
            sys.modules.pop("torchx_plugins.schedulers.conflict", None)

    def test_broken_path_in_iter_modules_is_caught(self) -> None:
        """A broken __path__ that causes pkgutil.iter_modules to fail is caught.

        This requires mocking because you can't create a namespace package
        with a broken __path__ as a file fixture.
        """
        broken_pkg = types.ModuleType("torchx_plugins.broken_ns")
        broken_pkg.__path__ = ["/nonexistent/path/that/will/fail"]  # type: ignore[attr-defined]
        sys.modules["torchx_plugins.broken_ns"] = broken_pkg

        try:
            with patch(
                "torchx.plugins._registry.pkgutil.iter_modules",
                side_effect=OSError("broken __path__"),
            ):
                reg = PluginRegistry(load_entrypoints=False)
                result = reg._find_namespace_plugins("torchx_plugins.broken_ns")

            self.assertEqual(
                result,
                {},
                "should return empty dict when iter_modules fails on broken __path__",
            )
        finally:
            sys.modules.pop("torchx_plugins.broken_ns", None)
