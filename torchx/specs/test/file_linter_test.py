# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[2]
# arguments untyped in certain functions for testing
# flake8: noqa: B950

import argparse
import os
import sys
import unittest
from typing import Dict, List, Optional
from unittest.mock import patch

from torchx.specs import AppDef

from torchx.specs.file_linter import (
    get_fn_docstring,
    TorchXArgumentHelpFormatter,
    validate,
)

IGNORED = AppDef(name="__IGNORED__")


# Note if the function is moved, the tests need to be updated with new lineno
def _test_empty_fn() -> AppDef:
    return IGNORED


# Note if the function is moved, the tests need to be updated with new lineno
def _test_fn_no_return() -> None:
    """
    Function description
    """
    pass


def _test_fn_return_int() -> int:
    """
    Function description
    """
    return 0


def _test_docstring(arg0: str, arg1: int, arg2: Dict[int, str]) -> AppDef:
    """Short Test description

    Long funct description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
    """
    return IGNORED


def _test_docstring_short() -> AppDef:
    """Short Test description"""
    return IGNORED


def _test_without_docstring(arg0: str) -> AppDef:
    return IGNORED


def _test_args_no_type_defs(arg0, arg1, arg2: Dict[int, str]) -> AppDef:
    """
    Test description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
        arg2: arg2 desc
    """
    return IGNORED


def _test_args_complex_types(
    arg0,
    arg1: Dict[int, List[str]],
    arg2: Dict[int, Dict[int, str]],
    arg3: Dict[List[int], str],
    arg4: Dict[Dict[int, str], str],
    arg5: List[List[str]],
    arg6: List[Dict[str, str]],
    arg7: Optional[Optional[str]],
) -> AppDef:
    """
    Test description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
        arg2: arg2 desc
        arg3: arg2 desc
    """
    return IGNORED


def _test_args_builtin_complex_types(
    arg0,
    arg1: dict[int, list[str]],
    arg2: dict[int, dict[int, str]],
    arg3: dict[list[int], str],
    arg4: dict[dict[int, str], str],
    arg5: list[list[str]],
    arg6: list[dict[str, str]],
    arg7: Optional[Optional[str]],
) -> AppDef:
    """
    Test description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
        arg2: arg2 desc
        arg3: arg2 desc
    """
    return IGNORED


if sys.version_info >= (3, 10):

    def _test_args_optional_types(
        arg0: int | None,
        arg1: None | int,
        arg2: dict[str, str] | None,
        arg3: list[str] | None,
        arg4: tuple[str, str] | None,
        arg5: Optional[int],
        arg6: Optional[dict[str, str]],
    ) -> AppDef:
        """
        Test both ways to specify optional for python-3.10+
        """
        return IGNORED


def _test_invalid_fn_with_varags_and_kwargs(*args, id: int) -> AppDef:
    """
    Test description

    Args:
        args: args desc
    """
    return IGNORED


def current_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), __file__)


class SpecsFileValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._path = current_file_path()
        with open(self._path, "r") as fp:
            source = fp.read()
        self._file_content = source

    def test_syntax_error(self) -> None:
        content = "!!foo====bar"
        with patch("torchx.specs.file_linter.read_conf_file") as read_conf_file_mock:
            read_conf_file_mock.return_value = content
            errors = validate(self._path, "unknown_function")
            self.assertEqual(1, len(errors))
            self.assertEqual("invalid syntax", errors[0].description)

    def test_validate_varargs_kwargs_fn(self) -> None:
        linter_errors = validate(self._path, "_test_invalid_fn_with_varags_and_kwargs")
        self.assertEqual(1, len(linter_errors))
        self.assertTrue(
            "Arg args missing type annotation", linter_errors[0].description
        )

    def test_validate_no_return(self) -> None:
        linter_errors = validate(self._path, "_test_fn_no_return")
        self.assertEqual(1, len(linter_errors))
        expected_desc = (
            "Function: _test_fn_no_return missing return annotation or "
            "has unknown annotations. Supported return annotation: AppDef"
        )
        self.assertEqual(expected_desc, linter_errors[0].description)

    def test_validate_incorrect_return(self) -> None:
        linter_errors = validate(self._path, "_test_fn_return_int")
        self.assertEqual(1, len(linter_errors))
        expected_desc = (
            "Function: _test_fn_return_int has incorrect return annotation, "
            "supported annotation: AppDef"
        )
        self.assertEqual(expected_desc, linter_errors[0].description)

    def test_no_validators_has_no_validation(self) -> None:
        linter_errors = validate(self._path, "_test_fn_return_int", [])
        self.assertEqual(0, len(linter_errors))

        linter_errors = validate(self._path, "_test_fn_no_return", [])
        self.assertEqual(0, len(linter_errors))

        linter_errors = validate(
            self._path, "_test_invalid_fn_with_varags_and_kwargs", []
        )
        self.assertEqual(0, len(linter_errors))

    def test_validate_empty_fn(self) -> None:
        linter_errors = validate(self._path, "_test_empty_fn")
        self.assertEqual(0, len(linter_errors))

    def test_validate_args_no_type_defs(self) -> None:
        fn = "_test_args_no_type_defs"
        linter_errors = validate(self._path, fn)
        error_msgs = [e.description for e in linter_errors]

        self.assertListEqual(
            [
                "Missing type annotation for argument 'arg0' in function '_test_args_no_type_defs'",
                "Missing type annotation for argument 'arg1' in function '_test_args_no_type_defs'",
            ],
            error_msgs,
        )

    def test_validate_args_complex_types(self) -> None:
        linter_errors = validate(self._path, "_test_args_complex_types")
        error_msgs = [e.description for e in linter_errors]
        self.assertListEqual(
            [
                "Missing type annotation for argument 'arg0' in function '_test_args_complex_types'",
                "Non-primitive value type 'List[str]' for argument 'arg1: Dict[int, List[str]]' in function '_test_args_complex_types'",
                "Non-primitive value type 'Dict[int, str]' for argument 'arg2: Dict[int, Dict[int, str]]' in function '_test_args_complex_types'",
                "Non-primitive key type 'List[int]' for argument 'arg3: Dict[List[int], str]' in function '_test_args_complex_types'",
                "Non-primitive key type 'Dict[int, str]' for argument 'arg4: Dict[Dict[int, str], str]' in function '_test_args_complex_types'",
                "Non-primitive element type 'List[str]' for argument 'arg5: List[List[str]]' in function '_test_args_complex_types'",
                "Non-primitive element type 'Dict[str, str]' for argument 'arg6: List[Dict[str, str]]' in function '_test_args_complex_types'",
                "Unsupported container type 'Optional' for argument 'arg7: Optional[Optional[str]]' in function '_test_args_complex_types'",
            ],
            error_msgs,
        )

    def test_validate_args_builtin_complex_types(self) -> None:
        linter_errors = validate(self._path, "_test_args_builtin_complex_types")
        error_msgs = [e.description for e in linter_errors]
        self.assertListEqual(
            [
                "Missing type annotation for argument 'arg0' in function '_test_args_builtin_complex_types'",
                "Non-primitive value type 'list[str]' for argument 'arg1: dict[int, list[str]]' in function '_test_args_builtin_complex_types'",
                "Non-primitive value type 'dict[int, str]' for argument 'arg2: dict[int, dict[int, str]]' in function '_test_args_builtin_complex_types'",
                "Non-primitive key type 'list[int]' for argument 'arg3: dict[list[int], str]' in function '_test_args_builtin_complex_types'",
                "Non-primitive key type 'dict[int, str]' for argument 'arg4: dict[dict[int, str], str]' in function '_test_args_builtin_complex_types'",
                "Non-primitive element type 'list[str]' for argument 'arg5: list[list[str]]' in function '_test_args_builtin_complex_types'",
                "Non-primitive element type 'dict[str, str]' for argument 'arg6: list[dict[str, str]]' in function '_test_args_builtin_complex_types'",
                "Unsupported container type 'Optional' for argument 'arg7: Optional[Optional[str]]' in function '_test_args_builtin_complex_types'",
            ],
            error_msgs,
        )

    # pyre-ignore[56]
    @unittest.skipUnless(
        sys.version_info >= (3, 10),
        "typing optional as [type]|None requires python-3.10+",
    )
    def test_validate_args_optional_type(self) -> None:
        linter_errors = validate(self._path, "_test_args_optional_types")
        self.assertFalse(linter_errors)

    def test_validate_docstring(self) -> None:
        func_desc, param_desc = get_fn_docstring(_test_docstring)
        self.assertEqual("Short Test description\nLong funct description", func_desc)
        self.assertEqual("arg0 desc", param_desc["arg0"])
        self.assertEqual("arg1 desc", param_desc["arg1"])
        self.assertEqual(" ", param_desc["arg2"])

    def test_validate_docstring_short(self) -> None:
        func_desc, param_desc = get_fn_docstring(_test_docstring_short)
        self.assertEqual("Short Test description", func_desc)

    def test_validate_docstring_no_docs(self) -> None:
        func_desc, param_desc = get_fn_docstring(_test_without_docstring)
        expected_fn_desc = """_test_without_docstring TIP: improve this help string by adding a docstring
to your component (see: https://meta-pytorch.org/torchx/latest/component_best_practices.html)"""
        self.assertEqual(expected_fn_desc, func_desc)
        self.assertEqual(" ", param_desc["arg0"])

    def test_validate_unknown_function(self) -> None:
        linter_errors = validate(self._path, "unknown_function")
        self.assertEqual(1, len(linter_errors))
        self.assertEqual(
            "Function unknown_function not found", linter_errors[0].description
        )

    def test_formatter(self) -> None:
        parser = argparse.ArgumentParser(
            prog="test prog",
            description="test desc",
        )
        parser.add_argument(
            "--foo",
            type=int,
            required=True,
            help="foo",
        )
        parser.add_argument(
            "--bar",
            type=int,
            help="bar",
            default=1,
        )
        formatter = TorchXArgumentHelpFormatter(prog="test")
        self.assertEqual(
            "show this help message and exit",
            formatter._get_help_string(parser._actions[0]),
        )
        self.assertEqual(
            "foo (required)", formatter._get_help_string(parser._actions[1])
        )
        self.assertEqual(
            "bar (default: 1)", formatter._get_help_string(parser._actions[2])
        )
