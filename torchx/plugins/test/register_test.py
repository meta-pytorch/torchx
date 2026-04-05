# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[6, 13, 16]

"""Tests for :py:mod:`torchx.plugins._registration`.

Covers the ``@register`` decorator, ``_register_named_resource``,
fractional constants, and the ``powers_of_two_gpus`` /
``halve_mem_down_to`` generators.
"""

import sys
import types
import unittest
from typing import Any

from torchx.plugins._registration import _register_named_resource, register
from torchx.plugins._registry import NAMED_RESOURCES_ATTR, PluginType, registry
from torchx.specs.api import Resource


# ── Helpers ──────────────────────────────────────────────────────────────────


class _ModuleContextManager:
    """Context manager that creates a temporary module in ``sys.modules``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.mod = types.ModuleType(name)
        self.mod.__name__ = name

    def __enter__(self) -> types.ModuleType:
        sys.modules[self.name] = self.mod
        return self.mod

    def __exit__(self, *_: Any) -> None:
        sys.modules.pop(self.name, None)
        registry.cache_clear()


# ── Decorator & registration tests ──────────────────────────────────────────


class PluginDecoratorTest(unittest.TestCase):
    def tearDown(self) -> None:
        registry.cache_clear()

    def test_tags_function_attributes(self) -> None:
        """``@register`` sets ``_plugin_type`` and ``_plugin_name`` on the function."""
        with _ModuleContextManager("test_tags"):

            def my_sched(session_name: str) -> str:
                return session_name

            my_sched.__module__ = "test_tags"
            register.scheduler(name="custom")(my_sched)

            self.assertEqual(
                getattr(my_sched, "_plugin_type", None),
                PluginType.SCHEDULER,
                "should tag _plugin_type on the function",
            )
            self.assertEqual(
                getattr(my_sched, "_plugin_name", None),
                "custom",
                "should tag _plugin_name on the function",
            )

    def test_default_name_is_function_name(self) -> None:
        """``@register.scheduler()`` defaults ``_plugin_name`` to ``fn.__name__``."""
        with _ModuleContextManager("test_default_name"):

            def my_sched(session_name: str) -> str:
                return session_name

            my_sched.__module__ = "test_default_name"
            register.scheduler()(my_sched)

            self.assertEqual(
                getattr(my_sched, "_plugin_name", None),
                "my_sched",
                "should default _plugin_name to function __name__",
            )

    def test_custom_name_overrides(self) -> None:
        """``@register.scheduler(name=...)`` uses the provided name."""
        with _ModuleContextManager("test_custom_name"):

            def factory(session_name: str) -> str:
                return session_name

            factory.__module__ = "test_custom_name"
            register.scheduler(name="custom")(factory)

            self.assertEqual(
                getattr(factory, "_plugin_name", None),
                "custom",
                "should use custom name, not function name",
            )

    def test_multiple_per_module(self) -> None:
        """Multiple ``@register`` calls in the same module tag each function independently."""
        with _ModuleContextManager("test_multi"):

            def sched_a(session_name: str) -> str:
                return "a"

            def sched_b(session_name: str) -> str:
                return "b"

            sched_a.__module__ = "test_multi"
            sched_b.__module__ = "test_multi"
            register.scheduler()(sched_a)
            register.scheduler()(sched_b)

            self.assertEqual(
                getattr(sched_a, "_plugin_name", None),
                "sched_a",
                "sched_a should be tagged with its name",
            )
            self.assertEqual(
                getattr(sched_b, "_plugin_name", None),
                "sched_b",
                "sched_b should be tagged with its name",
            )

    def test_returns_original_function(self) -> None:
        """``@register`` returns the original function unchanged."""
        with _ModuleContextManager("test_returns"):

            def my_sched(session_name: str) -> str:
                return session_name

            my_sched.__module__ = "test_returns"
            result = register.scheduler()(my_sched)

            self.assertIs(
                result,
                my_sched,
                "@register should return the original function",
            )

    def test_all_plugin_types(self) -> None:
        """Each convenience classmethod sets the correct ``_plugin_type``."""
        with _ModuleContextManager("test_all_types"):
            cases = [
                (register.scheduler(), PluginType.SCHEDULER),
                (register.tracker(), PluginType.TRACKER),
            ]
            for i, (dec, expected_type) in enumerate(cases):

                def fn() -> str:
                    return "x"

                fn.__module__ = "test_all_types"
                fn.__name__ = f"fn_{i}"
                dec(fn)
                self.assertEqual(
                    getattr(fn, "_plugin_type", None),
                    expected_type,
                    f"convenience method for {expected_type} should set correct _plugin_type",
                )


# ── Named resource decorator tests ──────────────────────────────────────────


class NamedResourceRegisterTest(unittest.TestCase):
    """Tests for :py:class:`_register_named_resource` (generic named-resource mechanics)."""

    def tearDown(self) -> None:
        registry.cache_clear()

    def test_registers_in_named_resources_dict(self) -> None:
        """Base factory goes into NAMED_RESOURCES and is tagged for discovery."""
        with _ModuleContextManager("test_nr_basic") as mod:

            def my_resource() -> Resource:
                return Resource(cpu=1, gpu=0, memMB=512)

            my_resource.__module__ = "test_nr_basic"
            result = register.named_resource()(my_resource)

            nr = getattr(mod, NAMED_RESOURCES_ATTR, None)
            self.assertIsNotNone(
                nr, "NAMED_RESOURCES dict should be auto-created on module"
            )
            assert nr is not None, "narrowing for Pyre"
            self.assertIn(
                "my_resource", nr, "function should be registered by __name__"
            )
            self.assertIs(
                nr["my_resource"],
                result,
                "registered value should be the factory returned by __call__",
            )
            # Should be tagged for discovery.
            self.assertEqual(
                getattr(result, "_plugin_type", None),
                PluginType.NAMED_RESOURCE,
                "factory should have _plugin_type = NAMED_RESOURCE",
            )
            self.assertEqual(
                getattr(result, "_plugin_name", None),
                "my_resource",
                "factory should have _plugin_name = 'my_resource'",
            )

    def test_custom_name(self) -> None:
        """``name=`` overrides fn.__name__ and sets a module attribute."""
        with _ModuleContextManager("test_nr_custom") as mod:

            def my_fn() -> Resource:
                return Resource(cpu=2, gpu=1, memMB=1024)

            my_fn.__module__ = "test_nr_custom"
            register.named_resource(name="custom_gpu")(my_fn)

            nr = getattr(mod, NAMED_RESOURCES_ATTR, {})
            self.assertIn("custom_gpu", nr, "should register under the custom name")
            self.assertNotIn("my_fn", nr, "function name should not appear in registry")
            self.assertTrue(
                hasattr(mod, "custom_gpu"),
                "custom name should be injected as module attribute",
            )

    def test_aliases(self) -> None:
        """Aliases are registered in the dict and as module attributes."""
        with _ModuleContextManager("test_nr_aliases") as mod:

            def base_res() -> Resource:
                return Resource(cpu=4, gpu=2, memMB=2048)

            base_res.__module__ = "test_nr_aliases"
            factory = register.named_resource(aliases=["alias_a", "alias_b"])(base_res)

            nr = getattr(mod, NAMED_RESOURCES_ATTR, {})
            self.assertIn("base_res", nr, "base name should be registered")
            self.assertIn("alias_a", nr, "alias_a should be registered")
            self.assertIn("alias_b", nr, "alias_b should be registered")

            # Aliases produce the same resource as the base factory.
            self.assertEqual(
                nr["alias_a"](),
                factory(),
                "alias_a should produce the same resource as the base",
            )
            self.assertEqual(
                nr["alias_b"](),
                factory(),
                "alias_b should produce the same resource as the base",
            )

            # Aliases are tagged with _plugin_base_name.
            self.assertEqual(
                nr["alias_a"]._plugin_base_name,
                "base_res",
                "alias_a should be tagged with the base name",
            )
            self.assertEqual(
                nr["alias_b"]._plugin_base_name,
                "base_res",
                "alias_b should be tagged with the base name",
            )

            # Module attributes.
            self.assertTrue(
                hasattr(mod, "alias_a"),
                "alias_a should be a module attribute",
            )
            self.assertTrue(
                hasattr(mod, "alias_b"),
                "alias_b should be a module attribute",
            )

            # All tagged for discovery.
            for name in ["base_res", "alias_a", "alias_b"]:
                self.assertEqual(
                    getattr(nr[name], "_plugin_type", None),
                    PluginType.NAMED_RESOURCE,
                    f"{name} should have _plugin_type = NAMED_RESOURCE",
                )

    def test_fractionals_callable(self) -> None:
        """Fractional generation with a callable ``(Resource) -> dict``."""
        with _ModuleContextManager("test_nr_frac") as mod:

            def gpu_host(fractional: float = 1.0) -> Resource:
                gpu = int(8 * fractional)
                return Resource(cpu=96, gpu=gpu, memMB=1024)

            gpu_host.__module__ = "test_nr_frac"

            def gpu_fracs(res: Resource) -> dict[float, str]:
                return {1.0: "8", 0.5: "4", 0.25: "2"}

            register.named_resource(fractionals=gpu_fracs)(gpu_host)

            nr = getattr(mod, NAMED_RESOURCES_ATTR, {})
            self.assertIn("gpu_host", nr, "base name should be registered")
            self.assertIn(
                "gpu_host_8", nr, "fractional gpu_host_8 should be registered"
            )
            self.assertIn(
                "gpu_host_4", nr, "fractional gpu_host_4 should be registered"
            )
            self.assertIn(
                "gpu_host_2", nr, "fractional gpu_host_2 should be registered"
            )

            # Verify fractional factories produce correct resources.
            res_4 = nr["gpu_host_4"]()
            self.assertEqual(res_4.gpu, 4, "gpu_host_4 should produce a 4-GPU resource")
            res_2 = nr["gpu_host_2"]()
            self.assertEqual(res_2.gpu, 2, "gpu_host_2 should produce a 2-GPU resource")

            # Module attributes.
            self.assertTrue(
                hasattr(mod, "gpu_host_8"),
                "fractional should be injected as module attribute",
            )

            # All tagged for discovery.
            for name in ["gpu_host", "gpu_host_8", "gpu_host_4", "gpu_host_2"]:
                self.assertEqual(
                    getattr(nr[name], "_plugin_type", None),
                    PluginType.NAMED_RESOURCE,
                    f"{name} should have _plugin_type = NAMED_RESOURCE",
                )

    def test_fractionals_dict(self) -> None:
        """Fractional generation with a literal dict."""
        with _ModuleContextManager("test_nr_frac_dict") as mod:

            def small_host(fractional: float = 1.0) -> Resource:
                return Resource(cpu=int(4 * fractional), gpu=0, memMB=512)

            small_host.__module__ = "test_nr_frac_dict"

            register.named_resource(fractionals={1.0: "full", 0.5: "half"})(small_host)

            nr = getattr(mod, NAMED_RESOURCES_ATTR, {})
            self.assertIn(
                "small_host_full", nr, "fractional small_host_full should be registered"
            )
            self.assertIn(
                "small_host_half", nr, "fractional small_host_half should be registered"
            )

    def test_duplicate_detection(self) -> None:
        """Raises ``ValueError`` on duplicate name within a module (named resources)."""
        with _ModuleContextManager("test_nr_dup"):

            def res_a() -> Resource:
                return Resource(cpu=1, gpu=0, memMB=256)

            def res_b() -> Resource:
                return Resource(cpu=2, gpu=0, memMB=512)

            res_a.__module__ = "test_nr_dup"
            res_b.__module__ = "test_nr_dup"

            register.named_resource(name="same")(res_a)

            with self.assertRaises(
                ValueError, msg="should raise on duplicate named resource"
            ):
                register.named_resource(name="same")(res_b)

    def test_make_factory_hook(self) -> None:
        """Subclass override of ``_make_factory`` is called."""
        with _ModuleContextManager("test_nr_hook_factory"):
            make_factory_calls: list[str] = []

            class custom_nr(_register_named_resource):
                def _make_factory(self, fn: Any, name: str) -> Any:
                    make_factory_calls.append(name)
                    return fn

            def my_res() -> Resource:
                return Resource(cpu=1, gpu=0, memMB=128)

            my_res.__module__ = "test_nr_hook_factory"
            custom_nr()(my_res)

            self.assertEqual(
                make_factory_calls,
                ["my_res"],
                "_make_factory hook should be called with the resource name",
            )

    def test_make_fractional_hook(self) -> None:
        """Subclass override of ``_make_fractional`` is called for each fraction."""
        with _ModuleContextManager("test_nr_hook_frac"):
            frac_calls: list[tuple[float, str]] = []

            class custom_nr(_register_named_resource):
                def _make_fractional(
                    self, fn: Any, fraction: float, frac_name: str
                ) -> Any:
                    frac_calls.append((fraction, frac_name))
                    return super()._make_fractional(fn, fraction, frac_name)

            def host(fractional: float = 1.0) -> Resource:
                return Resource(cpu=int(8 * fractional), gpu=0, memMB=512)

            host.__module__ = "test_nr_hook_frac"
            custom_nr(fractionals={1.0: "8", 0.5: "4"})(host)

            self.assertEqual(
                len(frac_calls),
                2,
                "_make_fractional should be called once per fractional entry",
            )
            frac_names = {name for _, name in frac_calls}
            self.assertEqual(
                frac_names,
                {"host_8", "host_4"},
                "_make_fractional should receive correct fractional names",
            )


# ── Fractional generators ────────────────────────────────────────────────────


class PowersOfTwoGpusTest(unittest.TestCase):
    """Unit tests for :meth:`register.powers_of_two_gpus`."""

    def test_8_gpus(self) -> None:
        r = Resource(cpu=64, gpu=8, memMB=1024)
        result = register.powers_of_two_gpus(r)
        self.assertEqual(
            result,
            {1.0: "8", 0.5: "4", 0.25: "2", 0.125: "1"},
            "8-GPU resource should produce 4 fractional entries",
        )

    def test_2_gpus(self) -> None:
        r = Resource(cpu=16, gpu=2, memMB=512)
        result = register.powers_of_two_gpus(r)
        self.assertEqual(
            result,
            {1.0: "2", 0.5: "1"},
            "2-GPU resource should produce 2 fractional entries",
        )

    def test_1_gpu(self) -> None:
        r = Resource(cpu=8, gpu=1, memMB=256)
        result = register.powers_of_two_gpus(r)
        self.assertEqual(
            result,
            {1.0: "1"},
            "1-GPU resource should produce only the whole entry",
        )

    def test_zero_gpus_raises(self) -> None:
        r = Resource(cpu=8, gpu=0, memMB=256)
        with self.assertRaises(ValueError, msg="gpu=0 should raise"):
            register.powers_of_two_gpus(r)

    def test_non_power_of_two_gpus_raises(self) -> None:
        r = Resource(cpu=8, gpu=6, memMB=256)
        with self.assertRaises(ValueError, msg="gpu=6 should raise"):
            register.powers_of_two_gpus(r)

    def test_top_level_importable(self) -> None:
        """``powers_of_two_gpus`` and ``halve_mem_down_to`` are importable from ``torchx.plugins``."""
        from torchx.plugins import halve_mem_down_to, powers_of_two_gpus

        self.assertIs(
            powers_of_two_gpus,
            register.powers_of_two_gpus,
            "top-level import should be the same object as register.powers_of_two_gpus",
        )
        self.assertIs(
            halve_mem_down_to,
            register.halve_mem_down_to,
            "top-level import should be the same object as register.halve_mem_down_to",
        )


class HalvingMemGiBTest(unittest.TestCase):
    """Unit tests for :meth:`register.halve_mem_down_to`."""

    GiB: int = 1024

    def test_64_gib_min_1(self) -> None:
        r = Resource(cpu=16, gpu=0, memMB=64 * self.GiB)
        result = register.halve_mem_down_to(minGiB=1)(r)
        self.assertEqual(
            result,
            {
                1.0: "64",
                0.5: "32",
                0.25: "16",
                0.125: "8",
                0.0625: "4",
                0.03125: "2",
                0.015625: "1",
            },
            "64 GiB with min=1 should halve down to 1",
        )

    def test_64_gib_min_8(self) -> None:
        r = Resource(cpu=16, gpu=0, memMB=64 * self.GiB)
        factory = register.halve_mem_down_to(minGiB=8)
        result = factory(r)
        self.assertEqual(
            result,
            {1.0: "64", 0.5: "32", 0.25: "16", 0.125: "8"},
            "64 GiB with min=8 should produce 4 fractional entries",
        )

    def test_256_gib_min_32(self) -> None:
        r = Resource(cpu=24, gpu=0, memMB=256 * self.GiB)
        factory = register.halve_mem_down_to(minGiB=32)
        result = factory(r)
        self.assertEqual(
            result,
            {1.0: "256", 0.5: "128", 0.25: "64", 0.125: "32"},
            "256 GiB with min=32 should produce 4 fractional entries",
        )

    def test_non_power_of_two_mem(self) -> None:
        """96 GiB (= 3 * 2^5) is valid, halves to 3."""
        r = Resource(cpu=16, gpu=0, memMB=96 * self.GiB)
        factory = register.halve_mem_down_to(minGiB=3)
        result = factory(r)
        self.assertEqual(
            result,
            {1.0: "96", 0.5: "48", 0.25: "24", 0.125: "12", 0.0625: "6", 0.03125: "3"},
            "96 GiB with minGiB=3 should halve down to odd part (3)",
        )

    def test_min_below_odd_part_raises(self) -> None:
        """96 GiB has odd part 3; min=2 should raise."""
        r = Resource(cpu=16, gpu=0, memMB=96 * self.GiB)
        factory = register.halve_mem_down_to(minGiB=2)
        with self.assertRaises(ValueError, msg="min < odd part should raise"):
            factory(r)

    def test_zero_mem_raises(self) -> None:
        r = Resource(cpu=8, gpu=0, memMB=0)
        with self.assertRaises(ValueError, msg="memMB=0 should raise"):
            register.halve_mem_down_to(minGiB=1)(r)

    def test_non_gib_aligned_raises(self) -> None:
        r = Resource(cpu=8, gpu=0, memMB=500)
        with self.assertRaises(ValueError, msg="non-GiB-aligned memMB should raise"):
            register.halve_mem_down_to(minGiB=1)(r)
