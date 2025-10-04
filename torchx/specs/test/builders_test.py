# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

from torchx.specs.api import AppDef, Resource, Role
from torchx.specs.builders import (
    _create_args_parser,
    BindMount,
    component_args_from_str,
    ComponentArgs,
    DeviceMount,
    make_app_handle,
    materialize_appdef,
    parse_mounts,
    VolumeMount,
)

from torchx.util.types import none_throws


class AppHandleTest(unittest.TestCase):
    def test_make(self) -> None:
        app_handle = make_app_handle(
            scheduler_backend="local",
            session_name="my_session",
            app_id="my_app_id_1234",
        )
        self.assertEqual("local://my_session/my_app_id_1234", app_handle)


def get_dummy_application(role: str) -> AppDef:
    trainer = Role(
        role,
        "test_image",
        entrypoint="main_script.py",
        args=["--train"],
        num_replicas=2,
    )
    return AppDef(name="test_app", roles=[trainer])


def example_empty_fn() -> AppDef:
    """Empty function that returns dummy app"""
    return get_dummy_application("trainer")


def example_fn_with_bool(flag: bool = False) -> AppDef:
    """Dummy app with or without flag

    Args:
        flag: flag param
    """
    if flag:
        return get_dummy_application("trainer-with-flag")
    else:
        return get_dummy_application("trainer-without-flag")


def example_fn_with_bool_optional(flag: Optional[bool] = None) -> AppDef:
    """Dummy app with or without flag

    Args:
        flag: flag param
    """
    if flag:
        return get_dummy_application("trainer-with-flag")
    else:
        return get_dummy_application("trainer-without-flag")


def example_empty_fn_no_docstring() -> AppDef:
    return get_dummy_application("trainer")


def example_test_complex_fn(
    app_name: str,
    containers: List[str],
    roles_scripts: Dict[str, str],
    num_cpus: Optional[List[int]] = None,
    num_gpus: Optional[Dict[str, int]] = None,
    nnodes: int = 4,
    first_arg: Optional[str] = None,
    nested_arg: Optional[Dict[str, List[str]]] = None,
    env: dict[str, str] | None = None,
    *roles_args: str,
) -> AppDef:
    """Creates complex application, testing all possible complex types

    Args:
        app_name: AppDef name
        containers: List of containers
        roles_scripts: Dict role_name -> role_script
    """
    num_roles = len(roles_scripts)
    if not num_cpus:
        num_cpus = [1] * num_roles
    if not num_gpus:
        num_gpus = {}
        for role in roles_scripts.keys():
            num_gpus[role] = 1
    roles = []
    for idx, (role_name, role_script) in enumerate(roles_scripts.items()):
        container_img = containers[idx]
        cpus = num_cpus[idx]
        gpus = num_gpus[role_name]
        if first_arg:
            args = [first_arg, *roles_args]
        elif nested_arg:
            args = nested_arg[role_name]
        else:
            args = [*roles_args]
        role = Role(
            role_name,
            image=container_img,
            entrypoint=role_script,
            args=args,
            resource=Resource(cpu=cpus, gpu=gpus, memMB=1),
            num_replicas=nnodes,
            env=env or {},
        )
        roles.append(role)
    return AppDef(app_name, roles)


_TEST_VAR_ARGS: Optional[Tuple[object, ...]] = None


def example_var_args(foo: str, *args: str, bar: str = "asdf") -> AppDef:
    """
    test component for mixing var args with kwargs.
    Args:
        foo: arg
        args: varargs
        bar: kwarg
    """
    global _TEST_VAR_ARGS
    _TEST_VAR_ARGS = (foo, args, bar)
    return AppDef(name="varargs")


_TEST_VAR_ARGS_FIRST: Optional[Tuple[object, ...]] = None


def example_var_args_first(*args: str, bar: str = "asdf") -> AppDef:
    """
    test component for mixing var args with kwargs.
    Args:
        args: varargs
        bar: kwarg
    """
    global _TEST_VAR_ARGS_FIRST
    _TEST_VAR_ARGS_FIRST = (args, bar)
    return AppDef(name="varargs")


_TEST_SINGLE_LETTER: Optional[str] = None


def example_single_letter(c: str) -> AppDef:
    global _TEST_SINGLE_LETTER
    _TEST_SINGLE_LETTER = c
    return AppDef(name="varargs")


class AppDefLoadTest(unittest.TestCase):
    def assert_apps(self, expected_app: AppDef, actual_app: AppDef) -> None:
        self.assertDictEqual(asdict(expected_app), asdict(actual_app))

    def _get_role_args(self) -> List[str]:
        return ["--train", "data_source", "random", "--epochs", "128"]

    def _get_nested_arg(self) -> Dict[str, List[str]]:
        return {"worker": ["1", "2"], "master": ["3", "4"]}

    def _get_expected_app_with_default(self) -> AppDef:
        role_args = self._get_role_args()
        return example_test_complex_fn(
            "test_app",
            ["img1", "img2"],
            {"worker": "worker.py", "master": "master.py"},
            None,
            None,
            4,
            None,
            None,
            None,
            *role_args,
        )

    def _get_args_with_default(self) -> List[str]:
        role_args = self._get_role_args()
        return [
            "--app_name",
            "test_app",
            "--containers",
            "img1,img2",
            "--roles_scripts",
            "worker=worker.py,master=master.py",
            "--",
            *role_args,
        ]

    def _get_expected_app_with_all_args(self) -> AppDef:
        role_args = self._get_role_args()
        return example_test_complex_fn(
            "test_app",
            ["img1", "img2"],
            {"worker": "worker.py", "master": "master.py"},
            [1, 2],
            {"worker": 1, "master": 4},
            8,
            "first_arg",
            None,
            {"FOO": "BAR", "HELLO": "WORLD"},
            *role_args,
        )

    def _get_app_args(self) -> List[str]:
        role_args = self._get_role_args()
        return [
            "--app_name",
            "test_app",
            "--containers",
            "img1,img2",
            "--roles_scripts",
            "worker=worker.py,master=master.py",
            "--num_cpus",
            "1,2",
            "--num_gpus",
            "worker=1,master=4",
            "--nnodes",
            "8",
            "--first_arg",
            "first_arg",
            "--env",
            "FOO=BAR,HELLO=WORLD",
            "--",
            *role_args,
        ]

    def _get_expected_app_with_nested_objects(self) -> AppDef:
        role_args = self._get_role_args()
        defaults = self._get_nested_arg()
        return example_test_complex_fn(
            "test_app",
            ["img1", "img2"],
            {"worker": "worker.py", "master": "master.py"},
            [1, 2],
            {"worker": 1, "master": 4},
            8,
            "first_arg",
            defaults,
            None,
            *role_args,
        )

    def _get_app_args_and_defaults_with_nested_objects(
        self,
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        role_args = self._get_role_args()
        defaults = self._get_nested_arg()
        return [
            "--app_name",
            "test_app",
            "--containers",
            "img1,img2",
            "--roles_scripts",
            "worker=worker.py,master=master.py",
            "--num_cpus",
            "1,2",
            "--num_gpus",
            "worker=1,master=4",
            "--nnodes",
            "8",
            "--first_arg",
            "first_arg",
            "--",
            *role_args,
        ], defaults

    def test_component_args_from_str(self) -> None:
        component_fn_args = [
            "--foo",
            "fooval",
            "--bar",
            "barval",
            "arg1",
            "arg2",
        ]
        parsed_args: ComponentArgs = component_args_from_str(
            example_var_args, component_fn_args
        )
        self.assertEqual(parsed_args.positional_args, {"foo": "fooval"})
        self.assertEqual(parsed_args.var_args, ["arg1", "arg2"])
        self.assertEqual(parsed_args.kwargs, {"bar": "barval"})

    def test_component_args_from_str_equals_separated(self) -> None:
        component_fn_args = [
            "--foo=fooval",
            "--bar=barval",
            "arg1",
            "arg2",
        ]
        parsed_args: ComponentArgs = component_args_from_str(
            example_var_args, component_fn_args
        )
        self.assertEqual(parsed_args.positional_args, {"foo": "fooval"})
        self.assertEqual(parsed_args.var_args, ["arg1", "arg2"])
        self.assertEqual(parsed_args.kwargs, {"bar": "barval"})

    def test_load_from_fn_empty(self) -> None:
        actual_app = materialize_appdef(example_empty_fn, [])
        expected_app = get_dummy_application("trainer")
        self.assert_apps(expected_app, actual_app)

    def test_load_from_fn_complex_all_args(self) -> None:
        expected_app = self._get_expected_app_with_all_args()
        app_args = self._get_app_args()
        actual_app = materialize_appdef(example_test_complex_fn, app_args)
        self.assert_apps(expected_app, actual_app)

    def test_required_args(self) -> None:
        with patch.object(sys, "exit") as exit_mock:
            try:
                materialize_appdef(example_test_complex_fn, [])
            except Exception:
                # ignore any errors, since function should fail
                pass
        exit_mock.assert_called_once()

    def test_load_from_fn_with_default(self) -> None:
        expected_app = self._get_expected_app_with_default()
        app_args = self._get_args_with_default()
        actual_app = materialize_appdef(example_test_complex_fn, app_args)
        self.assert_apps(expected_app, actual_app)

    def test_with_nested_object(self) -> None:
        expected_app = self._get_expected_app_with_nested_objects()
        app_args, defaults = self._get_app_args_and_defaults_with_nested_objects()
        actual_app = materialize_appdef(example_test_complex_fn, app_args, defaults)
        self.assert_apps(expected_app, actual_app)

    def test_varargs(self) -> None:
        materialize_appdef(
            example_var_args,
            [
                "--foo",
                "fooval",
                "--bar",
                "barval",
                "arg1",
                "arg2",
            ],
        )
        self.assertEqual(_TEST_VAR_ARGS, ("fooval", ("arg1", "arg2"), "barval"))

    def test_bool_true(self) -> None:
        app_def = materialize_appdef(
            example_fn_with_bool,
            [
                "--flag",
                "True",
            ],
        )
        self.assertEqual("trainer-with-flag", app_def.roles[0].name)
        app_def = materialize_appdef(
            example_fn_with_bool,
            [
                "--flag",
                "true",
            ],
        )
        self.assertEqual("trainer-with-flag", app_def.roles[0].name)

    def test_bool_false(self) -> None:
        app_def = materialize_appdef(
            example_fn_with_bool,
            [
                "--flag",
                "False",
            ],
        )
        self.assertEqual("trainer-without-flag", app_def.roles[0].name)
        app_def = materialize_appdef(
            example_fn_with_bool,
            [
                "--flag",
                "false",
            ],
        )
        self.assertEqual("trainer-without-flag", app_def.roles[0].name)

    def test_bool_none(self) -> None:
        app_def = materialize_appdef(
            example_fn_with_bool_optional,
            [],
        )
        self.assertEqual("trainer-without-flag", app_def.roles[0].name)

    def test_varargs_only_flag_first(self) -> None:
        materialize_appdef(
            example_var_args_first,
            [
                "--",
                "--foo",
                "fooval",
                "barval",
                "arg1",
                "arg2",
            ],
        )
        self.assertEqual(
            _TEST_VAR_ARGS_FIRST,
            (("--foo", "fooval", "barval", "arg1", "arg2"), "asdf"),
        )

    def test_varargs_only_arg_first(self) -> None:
        materialize_appdef(
            example_var_args_first,
            [
                "fooval",
                "--foo",
                "barval",
                "arg1",
                "arg2",
            ],
        )
        self.assertEqual(
            _TEST_VAR_ARGS_FIRST,
            (("fooval", "--foo", "barval", "arg1", "arg2"), "asdf"),
        )

    def test_single_letter(self) -> None:
        materialize_appdef(
            example_single_letter,
            [
                "-c",
                "arg1",
            ],
        )
        self.assertEqual(
            _TEST_SINGLE_LETTER,
            "arg1",
        )

        materialize_appdef(
            example_single_letter,
            [
                "--c",
                "arg2",
            ],
        )
        self.assertEqual(
            _TEST_SINGLE_LETTER,
            "arg2",
        )

    # pyre-ignore[3]
    def _get_argument_help(
        self, parser: argparse.ArgumentParser, name: str
    ) -> Optional[Tuple[str, Any]]:
        actions = parser._actions
        for action in actions:
            if action.dest == name:
                return action.help or "", action.default
        return None

    def test_argparster_complex_fn_partial(self) -> None:
        parser = _create_args_parser(example_test_complex_fn)
        self.assertTupleEqual(
            ("AppDef name", None),
            none_throws(self._get_argument_help(parser, "app_name")),
        )
        self.assertTupleEqual(
            ("List of containers", None),
            none_throws(self._get_argument_help(parser, "containers")),
        )
        self.assertTupleEqual(
            ("Dict role_name -> role_script", None),
            none_throws(self._get_argument_help(parser, "roles_scripts")),
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "num_cpus"))
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "num_gpus"))
        )
        self.assertTupleEqual(
            (" ", 4), none_throws(self._get_argument_help(parser, "nnodes"))
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "first_arg"))
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "roles_args"))
        )

    def test_argparser_remainder_main_args(self) -> None:
        parser = _create_args_parser(example_test_complex_fn)

        materialize_appdef(
            example_var_args,
            [
                "--foo",
                "fooval",
                "--bar",
                "barval",
                "arg1",
                "arg2",
            ],
            {"args": "arg3 arg4"},
        )
        self.assertEqual(_TEST_VAR_ARGS, ("fooval", ("arg1", "arg2"), "barval"))

        materialize_appdef(
            example_var_args,
            [
                "--foo",
                "fooval",
                "--bar",
                "barval",
            ],
            {"args": "arg3 arg4"},
        )
        self.assertEqual(_TEST_VAR_ARGS, ("fooval", ("arg3", "arg4"), "barval"))


class MountsTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(parse_mounts([]), [])

    def test_bindmount(self) -> None:
        self.assertEqual(
            parse_mounts(
                [
                    "type=bind",
                    "src=foo",
                    "dst=dst",
                    "type=bind",
                    "source=foo1",
                    "readonly",
                    "target=dst1",
                    "type=volume",
                    "destination=dst2",
                    "source=foo2",
                    "readonly",
                    "type=device",
                    "src=duck",
                    "type=device",
                    "src=foo",
                    "dst=bar",
                    "perm=rw",
                    "type=bind",
                    "src=~/foo",
                    "dst=dst",
                ]
            ),
            [
                BindMount(src_path="foo", dst_path="dst"),
                BindMount(src_path="foo1", dst_path="dst1", read_only=True),
                VolumeMount(src="foo2", dst_path="dst2", read_only=True),
                DeviceMount(src_path="duck", dst_path="duck", permissions="rwm"),
                DeviceMount(src_path="foo", dst_path="bar", permissions="rw"),
                BindMount(src_path=f"{str(Path.home())}/foo", dst_path="dst"),
            ],
        )

    def test_invalid(self) -> None:
        with self.assertRaisesRegex(KeyError, "type must be specified first"):
            parse_mounts(["src=foo"])
        with self.assertRaisesRegex(
            KeyError, "unknown mount option blah, must be one of.*type"
        ):
            parse_mounts(["blah=foo"])
        with self.assertRaisesRegex(KeyError, "src"):
            parse_mounts(["type=bind"])
        with self.assertRaisesRegex(
            ValueError, "invalid mount type.*must be one of.*bind"
        ):
            parse_mounts(["type=foo"])
