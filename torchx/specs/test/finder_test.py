#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import shutil
import sys
import tempfile
import unittest
from importlib.metadata import EntryPoints
from pathlib import Path
from unittest.mock import MagicMock, patch

import torchx.specs.finder as finder
from torchx.plugins._registry import registry
from torchx.runner import get_runner
from torchx.runtime.tracking import FsspecResultTracker
from torchx.specs.api import AppDef, AppState, Role
from torchx.specs.finder import (
    _load_components,
    ComponentNotFoundException,
    ComponentValidationException,
    CustomComponentsFinder,
    get_component,
    get_components,
    ModuleComponentsFinder,
)
from torchx.util.test.entrypoints_test import EntryPoint_from_text
from torchx.util.types import none_throws

_METADATA_EPS: str = "torchx.util.entrypoints.metadata.entry_points"


def _test_component(name: str, role_name: str = "worker") -> AppDef:
    """
    Test component

    Args:
        name: AppDef name
        role_name: Role name

    Returns:
        AppDef
    """
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


def _test_component_without_docstring(name: str, role_name: str = "worker") -> AppDef:
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


# pyre-ignore[2]
def invalid_component(name, role_name: str = "worker") -> AppDef:
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


class FinderTest(unittest.TestCase):
    _ENTRY_POINTS: EntryPoints = EntryPoints(
        EntryPoint_from_text(
            """
[torchx.components]
_ = torchx.specs.test.finder_test
        """
        )
    )

    def setUp(self) -> None:
        # clear caches since find_component() has side-effects
        # and we load a bunch of mocks for components in the tests below
        finder._components = None
        registry.cache_clear()

    def tearDown(self) -> None:
        finder._components = None
        registry.cache_clear()

    def test_module_relname(self) -> None:
        import torchx.specs.test.components as c
        import torchx.specs.test.components.a as ca

        self.assertEqual("", finder.module_relname(c, relative_to=c))
        self.assertEqual("a", finder.module_relname(ca, relative_to=c))
        with self.assertRaises(ValueError):
            finder.module_relname(c, relative_to=ca)

    def test_get_component_by_name(self) -> None:
        component = none_throws(get_component("utils.echo"))
        self.assertEqual("utils.echo", component.name)
        self.assertEqual("echo", component.fn_name)
        self.assertIsNotNone(component.fn)

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_get_invalid_component_by_name(self, _: MagicMock) -> None:
        with self.assertRaises(ComponentValidationException):
            get_component("invalid_component")

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_get_unknown_component_by_name(self, _: MagicMock) -> None:
        with self.assertRaises(ComponentNotFoundException):
            get_component("unknown_component")

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_get_invalid_component(self, _: MagicMock) -> None:
        components = _load_components(None)
        foobar_component = components["invalid_component"]
        self.assertEqual(1, len(foobar_component.validation_errors))

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_get_entrypoints_components(self, _: MagicMock) -> None:
        components = _load_components(None)
        foobar_component = components["_test_component"]
        self.assertEqual(_test_component, foobar_component.fn)
        self.assertEqual("_test_component", foobar_component.fn_name)
        self.assertEqual("_test_component", foobar_component.name)
        self.assertEqual("Test component", foobar_component.description)

    @patch(
        _METADATA_EPS,
        return_value=EntryPoints(
            EntryPoint_from_text(
                """
[torchx.components]
foo = torchx.specs.test.components.a
bar = torchx.specs.test.components.c.d
"""
            )
        ),
    )
    def test_load_custom_components(self, _: MagicMock) -> None:
        components = _load_components(None)

        # the name of the appdefs returned by each component
        # is the expected component name
        for actual_name, comp in components.items():
            expected_name = comp.fn().name
            self.assertEqual(expected_name, actual_name)

        self.assertEqual(3, len(components))

    @patch(
        _METADATA_EPS,
        return_value=EntryPoints(
            EntryPoint_from_text(
                """
[torchx.components]
_0 = torchx.specs.test.components.a
_1 = torchx.specs.test.components.c.d
"""
            )
        ),
    )
    def test_load_custom_components_nogroup(self, _: MagicMock) -> None:
        components = _load_components(None)

        # test component names are hardcoded expecting
        # test.components.* to be grouped under foo.*
        # and components.a_namepace.* to be grouped under bar.*
        # since we are testing _* (no group prefix) remove the first prefix
        for actual_name, comp in components.items():
            expected_name = comp.fn().name.split(".", maxsplit=1)[1]
            self.assertEqual(expected_name, actual_name)

    def test_load_builtins(self) -> None:
        components = _load_components(None)

        # if nothing registered in entrypoints, then builtins should be loaded
        expected = {
            c.name
            for c in ModuleComponentsFinder("torchx.components", group="").find(None)
        }
        self.assertEqual(components.keys(), expected)

    def test_load_builtin_echo(self) -> None:
        components = _load_components(None)
        self.assertTrue(len(components) > 1)
        component = components["utils.echo"]
        self.assertEqual("utils.echo", component.name)
        self.assertEqual(
            "Echos a message to stdout (calls echo)", component.description
        )
        self.assertEqual("echo", component.fn_name)
        self.assertIsNotNone(component.fn)


def current_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), __file__)


class CustomComponentsFinderTest(unittest.TestCase):
    def test_find_components(self) -> None:
        components = CustomComponentsFinder(
            current_file_path(), "_test_component"
        ).find(None)
        self.assertEqual(1, len(components))
        component = components[0]
        self.assertEqual(f"{current_file_path()}:_test_component", component.name)
        self.assertEqual("Test component", component.description)
        self.assertEqual("_test_component", component.fn_name)
        self.assertListEqual([], component.validation_errors)

    def test_find_components_without_docstring(self) -> None:
        components = CustomComponentsFinder(
            current_file_path(), "_test_component_without_docstring"
        ).find(None)
        self.assertEqual(1, len(components))
        component = components[0]
        self.assertEqual(
            f"{current_file_path()}:_test_component_without_docstring", component.name
        )
        exprected_desc = """_test_component_without_docstring TIP: improve this help string by adding a docstring
to your component (see: https://meta-pytorch.org/torchx/latest/component_best_practices.html)"""
        self.assertEqual(exprected_desc, component.description)
        self.assertEqual("_test_component_without_docstring", component.fn_name)
        self.assertListEqual([], component.validation_errors)

    def test_get_component(self) -> None:
        component = get_component(f"{current_file_path()}:_test_component")
        self.assertEqual(f"{current_file_path()}:_test_component", component.name)
        self.assertEqual("Test component", component.description)
        self.assertEqual("_test_component", component.fn_name)
        self.assertListEqual([], component.validation_errors)

    def test_get_components(self) -> None:
        components = get_components()
        for component in components.values():
            self.assertListEqual([], component.validation_errors)

    def test_get_component_unknown(self) -> None:
        with self.assertRaises(ComponentNotFoundException):
            get_component(f"{current_file_path()}:unknown_component")

    def test_get_component_invalid(self) -> None:
        with self.assertRaises(ComponentValidationException):
            get_component(f"{current_file_path()}:invalid_component")


class ModuleIdentityLoadingTest(unittest.TestCase):
    """Component files load as regular modules (not exec'd into the finder's globals)."""

    _PKG_COMPONENT = """
from torchx.specs import AppDef, Role

from idpkg.helper import ROLE_NAME


def comp(msg: str = "hello") -> AppDef:
    \"\"\"Test component

    Args:
        msg: message
    \"\"\"
    return AppDef(msg, roles=[Role(name=ROLE_NAME, image="img", entrypoint="echo")])
"""

    _STANDALONE_COMPONENT = """
from torchx.specs import AppDef, Role

from id_neighbor import ROLE_NAME


def comp(msg: str = "hello") -> AppDef:
    \"\"\"Test component

    Args:
        msg: message
    \"\"\"
    return AppDef(msg, roles=[Role(name=ROLE_NAME, image="img", entrypoint="echo")])
"""

    _HELPER = 'ROLE_NAME = "sibling"\n'

    def setUp(self) -> None:
        finder._components = None
        registry.cache_clear()

        self.test_dir = Path(tempfile.mkdtemp("torchx_finder_module_identity_test"))

        pkg_dir = self.test_dir / "idpkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "helper.py").write_text(self._HELPER)
        (pkg_dir / "idcomp.py").write_text(self._PKG_COMPONENT)

        standalone_dir = self.test_dir / "standalone"
        standalone_dir.mkdir()
        (standalone_dir / "id_neighbor.py").write_text(self._HELPER)
        (standalone_dir / "id_standalone_comp.py").write_text(
            self._STANDALONE_COMPONENT
        )

        self._orig_sys_path = list(sys.path)

    def tearDown(self) -> None:
        finder._components = None
        registry.cache_clear()
        sys.path[:] = self._orig_sys_path
        for mod in [
            m
            for m, loaded in sys.modules.items()
            if m.startswith(("idpkg", "id_neighbor", "id_standalone_comp"))
            or str(self.test_dir) in (getattr(loaded, "__file__", None) or "")
        ]:
            del sys.modules[mod]
        shutil.rmtree(self.test_dir)

    def test_package_file_component_imports_siblings(self) -> None:
        component = get_component(f"{self.test_dir / 'idpkg' / 'idcomp.py'}:comp")
        app = component.fn()
        self.assertEqual("sibling", app.roles[0].name)

    def test_package_file_component_has_module_identity(self) -> None:
        filepath = str(self.test_dir / "idpkg" / "idcomp.py")
        component = get_component(f"{filepath}:comp")
        self.assertEqual("idpkg.idcomp", component.fn.__module__)
        self.assertIn("idpkg.idcomp", sys.modules)
        self.assertEqual(filepath, sys.modules["idpkg.idcomp"].__file__)

    def test_package_file_component_supports_relative_imports(self) -> None:
        (self.test_dir / "idpkg" / "relcomp.py").write_text(
            self._PKG_COMPONENT.replace(
                "from idpkg.helper import ROLE_NAME",
                "from .helper import ROLE_NAME",
            )
        )
        component = get_component(f"{self.test_dir / 'idpkg' / 'relcomp.py'}:comp")
        self.assertEqual(
            "sibling",
            component.fn().roles[0].name,
            "the parent package is imported before the leaf, so relative"
            " imports inside a package component file must resolve",
        )
        self.assertIs(
            sys.modules["idpkg.relcomp"],
            sys.modules["idpkg"].relcomp,
            "the loaded leaf must be set as an attribute on its parent"
            " package, as a real import would",
        )

    def test_shadowed_package_root_warns(self) -> None:
        shadow_root = self.test_dir / "shadow_root"
        shadow_pkg = shadow_root / "idpkg"
        shadow_pkg.mkdir(parents=True)
        (shadow_pkg / "__init__.py").write_text("")
        (shadow_pkg / "helper.py").write_text('ROLE_NAME = "shadowed"\n')
        (shadow_pkg / "shadowcomp.py").write_text(self._PKG_COMPONENT)
        sys.path.insert(0, str(self.test_dir))
        with self.assertLogs("torchx.specs.finder", level="WARNING") as logs:
            get_component(f"{shadow_pkg / 'shadowcomp.py'}:comp")
        self.assertTrue(
            any("shadows" in line for line in logs.output),
            "loading a component whose package name resolves to a different"
            " sys.path root must warn about the shadowing",
        )

    def test_non_identifier_package_dir_loads_standalone(self) -> None:
        pkg_dir = self.test_dir / "my-pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "helper.py").write_text(self._HELPER)
        (pkg_dir / "dashcomp.py").write_text(
            self._STANDALONE_COMPONENT.replace("id_neighbor", "helper")
        )
        component = get_component(f"{pkg_dir / 'dashcomp.py'}:comp")
        self.assertEqual(
            "dashcomp",
            component.fn.__module__,
            "a package dir that is not a valid identifier cannot appear in a"
            " dotted name; the file must load standalone under its stem",
        )
        self.assertEqual("sibling", component.fn().roles[0].name)

    def test_standalone_file_component_imports_neighbors(self) -> None:
        filepath = str(self.test_dir / "standalone" / "id_standalone_comp.py")
        component = get_component(f"{filepath}:comp")
        self.assertEqual("id_standalone_comp", component.fn.__module__)
        app = component.fn()
        self.assertEqual("sibling", app.roles[0].name)

    def test_reload_returns_same_module(self) -> None:
        filepath = str(self.test_dir / "idpkg" / "idcomp.py")
        fn1 = get_component(f"{filepath}:comp").fn
        fn2 = get_component(f"{filepath}:comp").fn
        self.assertIs(fn1, fn2)

    def test_module_name_collision_gets_unique_name(self) -> None:
        (self.test_dir / "standalone" / "logging.py").write_text(
            self._STANDALONE_COMPONENT
        )
        import logging as stdlib_logging

        filepath = str(self.test_dir / "standalone" / "logging.py")
        component = get_component(f"{filepath}:comp")
        self.assertEqual("logging_1", component.fn.__module__)
        self.assertIs(stdlib_logging, sys.modules["logging"])

    def test_virtual_file_execs_into_synthetic_module(self) -> None:
        filepath = str(self.test_dir / "does_not_exist.py")
        with (
            patch(
                "torchx.specs.finder.read_conf_file",
                return_value=self._HELPER + VIRTUAL_COMPONENT_SRC,
            ),
            patch(
                "torchx.specs.file_linter.read_conf_file",
                return_value=self._HELPER + VIRTUAL_COMPONENT_SRC,
            ),
        ):
            component = get_component(f"{filepath}:comp")
        self.assertTrue(component.fn.__module__.startswith("torchx_component_file_"))
        self.assertEqual(filepath, sys.modules[component.fn.__module__].__file__)
        app = component.fn()
        self.assertEqual("sibling", app.roles[0].name)


VIRTUAL_COMPONENT_SRC = """
from torchx.specs import AppDef, Role


def comp(msg: str = "hello") -> AppDef:
    \"\"\"Test component

    Args:
        msg: message
    \"\"\"
    return AppDef(msg, roles=[Role(name=ROLE_NAME, image="img", entrypoint="echo")])
"""


class GetBuiltinSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        # clear caches to avoid stale plugin registry state from other tests
        finder._components = None
        registry.cache_clear()

        self.test_dir = Path(tempfile.mkdtemp("torchx_specs_finder_test"))

        # this is so that the test can pick up penv python (fb-only)
        # which is added as a test resource
        self.orig_cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))

    def tearDown(self) -> None:
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir)

    def test_get_builtin_source_with_echo(self) -> None:
        echo_src = finder.get_builtin_source("utils.echo")

        # save it to a file and try running it
        echo_copy = self.test_dir / "echo_copy.py"
        with open(echo_copy, "w") as f:
            f.write(echo_src)

        runner = get_runner()
        app_handle = runner.run_component(
            scheduler="local_cwd",
            component=f"{str(echo_copy)}:echo",
            component_args=["--msg", "hello world"],
        )
        status = runner.wait(app_handle, wait_interval=0.1)
        self.assertIsNotNone(status)
        self.assertEqual(AppState.SUCCEEDED, status.state)

    def test_get_builtin_source_with_booth(self) -> None:
        # try copying and running a builtin that is NOT the first
        # defined function in the file

        booth_src = finder.get_builtin_source("utils.booth")

        # save it to a file and try running it
        booth_copy = self.test_dir / "booth_copy.py"
        with open(booth_copy, "w") as f:
            f.write(booth_src)

        runner = get_runner()

        trial_idx = 0
        tracker_base = str(self.test_dir / "tracking")

        app_handle = runner.run_component(
            scheduler="local_cwd",
            cfg={"prepend_cwd": True},
            component=f"{str(booth_copy)}:booth",
            component_args=[
                "--x1=1",
                "--x2=3",
                f"--trial_idx={trial_idx}",
                f"--tracker_base={tracker_base}",
            ],
        )
        status = runner.wait(app_handle, wait_interval=0.1)
        self.assertIsNotNone(status)
        self.assertEqual(AppState.SUCCEEDED, status.state)

        tracker = FsspecResultTracker(tracker_base)
        # booth function has global min of 0 at (1, 3)
        self.assertEqual(0, tracker[trial_idx]["booth_eval"])
