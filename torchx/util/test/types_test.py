# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from typing import cast, Optional, Union

from torchx.util.types import (
    decode,
    decode_from_string,
    decode_optional,
    get_argparse_param_type,
    is_bool,
    is_primitive,
    none_throws,
    to_dict,
    to_list,
)


def _test_complex_args(
    arg1: int,
    arg2: Optional[list[str]],
    arg3: Union[float, int],
) -> int:
    return 42


def _test_dict(arg1: dict[int, float]) -> int:
    return 42


def _test_nested_object(arg1: dict[str, list[str]]) -> int:
    return 42


def _test_list(arg1: list[float]) -> int:
    return 42


def _test_complex_list(arg1: list[list[float]]) -> int:
    return 42


def _test_unknown(arg1: Optional[Optional[float]]) -> int:
    return 42


class TypesTest(unittest.TestCase):
    def test_decode_optional(self) -> None:
        parameters = inspect.signature(_test_complex_args).parameters

        arg1_parameter = parameters["arg1"]
        arg1_type = decode_optional(arg1_parameter.annotation)
        self.assertTrue(arg1_type is int)
        arg2_type = decode_optional(parameters["arg2"].annotation)
        self.assertTrue(getattr(arg2_type, "__origin__", None) is list)
        arg3_parameter = parameters["arg3"]
        arg3_type = decode_optional(arg3_parameter.annotation)
        self.assertTrue(
            hasattr(arg3_type, "__origin__") and arg3_type.__origin__ is Union
        )

    def test_is_primitive(self) -> None:
        parameters = inspect.signature(_test_complex_args).parameters

        arg1_parameter = parameters["arg1"]
        self.assertTrue(is_primitive(arg1_parameter.annotation))

        arg2_parameter = parameters["arg2"]
        self.assertFalse(is_primitive(arg2_parameter.annotation))

    def test_is_bool(self) -> None:
        self.assertTrue(is_bool(bool))
        self.assertFalse(is_bool(int))

    def test_decode_from_string_dict(self) -> None:
        parameters = inspect.signature(_test_dict).parameters

        encoded_value = "1=1.0,2=42.1,3=10"

        value = decode_from_string(encoded_value, parameters["arg1"].annotation)
        value = cast(dict[int, float], value)
        self.assertEqual(3, len(value))
        self.assertEqual(float(1.0), value[1])
        self.assertEqual(float(42.1), value[2])
        self.assertEqual(float(10), value[3])

    def test_decode_from_string_list(self) -> None:
        parameters = inspect.signature(_test_list).parameters

        encoded_value = "1.0,42.2,3.9"

        value = decode_from_string(encoded_value, parameters["arg1"].annotation)
        value = cast(list[float], value)
        self.assertEqual(3, len(value))
        self.assertEqual(float(1.0), value[0])
        self.assertEqual(float(42.2), value[1])
        self.assertEqual(float(3.9), value[2])

    def test_decode(self) -> None:
        encoded_value = "1.0,42.2,3.9"

        dict_parameters = inspect.signature(_test_nested_object).parameters
        list_parameters = inspect.signature(_test_list).parameters

        value = {"a": ["1", "2"], "b": ["3", "4"]}
        self.assertDictEqual(value, decode(value, dict_parameters["arg1"].annotation))

        self.assertEqual(decode("true", bool), True)
        self.assertEqual(decode("false", bool), False)

        self.assertEqual(decode(None, int), None)

        self.assertEqual(
            decode_from_string(encoded_value, list_parameters["arg1"].annotation),
            decode(encoded_value, list_parameters["arg1"].annotation),
        )

    def test_decode_from_string_empty(self) -> None:
        parameters = inspect.signature(_test_list).parameters
        value = decode_from_string("", parameters["arg1"].annotation)
        self.assertEqual(None, value)

    def test_decode_from_string_complex_list(self) -> None:
        parameters = inspect.signature(_test_complex_list).parameters
        with self.assertRaises(ValueError):
            decode_from_string("1", parameters["arg1"].annotation)

    def test_decode_from_string_unknown(self) -> None:
        parameters = inspect.signature(_test_unknown).parameters
        with self.assertRaises(ValueError):
            decode_from_string("2", parameters["arg1"].annotation)

    def test_to_dict_empty(self) -> None:
        self.assertDictEqual({}, to_dict(""))

    def test_to_dict(self) -> None:
        self.assertDictEqual({"FOO": "v1"}, to_dict("FOO=v1"))

        self.assertDictEqual({"FOO": "v1,v2"}, to_dict("FOO=v1,v2"))
        self.assertDictEqual({"FOO": "v1;v2"}, to_dict("FOO=v1;v2"))

        # Handles empty strings as a special case
        self.assertDictEqual({"FOO": ""}, to_dict("FOO=''"))
        self.assertDictEqual({"FOO": ""}, to_dict('FOO=""'))

        # trailing delimiters preserved
        # a delim without the next key should be interpreted as the value for FOO
        self.assertDictEqual({"FOO": ","}, to_dict("FOO=,"))
        self.assertDictEqual({"FOO": ";"}, to_dict("FOO=;"))
        self.assertDictEqual({"FOO": "v1,v2,"}, to_dict("FOO=v1,v2,"))
        self.assertDictEqual({"FOO": "v1,v2;"}, to_dict("FOO=v1,v2;"))
        self.assertDictEqual({"FOO": "v1;v2;"}, to_dict("FOO=v1;v2;"))
        self.assertDictEqual({"FOO": "v1;v2,"}, to_dict("FOO=v1;v2,"))

        # ; and , can be used interchangeably to delimit list values and kv pairs
        self.assertDictEqual({"FOO": "v1,v2", "BAR": "v3"}, to_dict("FOO=v1,v2,BAR=v3"))
        self.assertDictEqual({"FOO": "v1;v2", "BAR": "v3"}, to_dict("FOO=v1;v2;BAR=v3"))
        self.assertDictEqual({"FOO": "v1,v2", "BAR": "v3"}, to_dict("FOO=v1,v2;BAR=v3"))
        self.assertDictEqual({"FOO": "v1;v2", "BAR": "v3"}, to_dict("FOO=v1;v2,BAR=v3"))

        # Parser cannot appropriately handle the cases where
        # value contains "=". Create test for this to keep record. This especially can
        # be a problem for some base64 encoded values with trailing "=" or "=="
        # For some components (dist.ddp), an `env_file` option is provided to read env
        # from file
        with self.assertRaises(AssertionError):
            self.assertDictEqual(
                {"FOO": "v1;v2", "BAR": "v3=="}, to_dict("FOO=v1;v2,BAR=v3==")
            )
            self.assertDictEqual(
                {"FOO": "v1;v2", "BAR": "v3="}, to_dict("FOO=v1;v2,BAR=v3=")
            )

        # test some non-trivial + edge cases
        self.assertDictEqual(
            {
                "foo": "bar1,bar2,bar3",
                "my_key": "my_value",
                "new_foo": "new_bar1,new_bar2",
            },
            to_dict("foo=bar1,bar2,bar3,my_key=my_value,new_foo=new_bar1,new_bar2"),
        )

        self.assertDictEqual(
            {"foo": "bar", "key1": "value1,"},
            to_dict("key1=value1,,foo=bar"),
        )

        self.assertDictEqual(
            {"FOO": "value with = and , and ;"},
            to_dict('FOO="value with = and , and ;"'),
        )

    def test_to_dict_malformed_literal(self) -> None:
        for malformed in ["FOO", "FOO,", "FOO;", "FOO=", "FOO=;BAR=v1"]:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    print(to_dict(malformed))

    def test_to_list_empty(self) -> None:
        self.assertListEqual([], to_list(""))

    def test_get_argparse_param_type(self) -> None:
        def fake_component(
            i: int,
            f: float,
            s: str,
            b: bool,
            l: list[str],
            m: dict[str, str],
            o: Optional[int],
        ) -> None:
            # component has to return an AppDef
            # but ok here since we're simply testing the parameter types
            pass

        params = inspect.signature(fake_component).parameters
        self.assertEqual(int, get_argparse_param_type(params["i"]))
        self.assertEqual(float, get_argparse_param_type(params["f"]))
        self.assertEqual(str, get_argparse_param_type(params["s"]))
        self.assertEqual(str, get_argparse_param_type(params["b"]))
        self.assertEqual(str, get_argparse_param_type(params["l"]))
        self.assertEqual(str, get_argparse_param_type(params["m"]))
        self.assertEqual(str, get_argparse_param_type(params["o"]))

    def test_none_throws(self) -> None:
        self.assertEqual(none_throws(10), 10)
        self.assertEqual(none_throws("str"), "str")
        with self.assertRaisesRegex(AssertionError, "Unexpected.*None"):
            none_throws(None)
