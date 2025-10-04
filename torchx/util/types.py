# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import re
from types import UnionType
from typing import Any, Callable, Optional, Tuple, TypeVar, Union


def to_list(arg: str) -> list[str]:
    conf = []
    if len(arg.strip()) == 0:
        return []
    for el in arg.split(","):
        conf.append(el.strip())
    return conf


def to_dict(arg: str) -> dict[str, str]:
    """
    Parses the given ``arg`` string literal into a ``dict[str, str]`` of
    key-value pairs delimited by ``"="`` (equals). The values may be a
    list literal where the list elements are delimited by ``","`` (comma)
    or ``";"`` (semi-colon). The same delimiters (``","`` and ``";"``) are used
    to specify multiple key-value pairs in the ``arg`` literal (see examples below).
    When values are lists, the last delimiter is used as kv-pair delimiter
    (e.g. ``FOO=v1,v2,BAR=v3``). Empty values of ``arg`` returns an empty map.

    Values can be quoted with single or double quotes to include special characters
    (``"="``, ``","``, ``";"``) without them being interpreted as separators.

    Note that values that encode list literals are returned as list literals
    NOT actual lists. The caller must further process each value in the returned
    map, to cast/decode the value literals as specific types. In this case,
    the list delimiters are preserved (see examples below).


    Examples:

    .. code-block:: python

     to_dict("") == {}

     to_dict("FOO=v1") == {"FOO": "v1"}

     to_dict("FOO=''") == {"FOO": ""}
     to_dict('FOO=""') == {"FOO": ""}

     to_dict("FOO=v1,v2") == {"FOO": "v1,v2"]}
     to_dict("FOO=v1;v2") == {"FOO": "v1;v2"]}
     to_dict("FOO=v1;v2") == {"FOO": "v1;v2,"]}
     to_dict("FOO=v1;v2") == {"FOO": "v1;v2,"]}

     to_dict("FOO=v1,v2,BAR=v3") == {"FOO": "v1,v2", "BAR": "v3"}
     to_dict("FOO=v1;v2,BAR=v3") == {"FOO": "v1;v2", "BAR": "v3"}
     to_dict("FOO=v1;v2;BAR=v3") == {"FOO": "v1;v2", "BAR": "v3"}

     to_dict('FOO="value with = and , and ;"') == {"FOO": "value with = and , and ;"}
    """

    def parse_val_key(vk: str) -> Tuple[str, str]:
        # ``vk`` is assumed to be in value<delim>key format
        delims = [",", ";"]
        idx = max([vk.rfind(d) for d in delims])
        if idx == -1 or idx == 0 or len(vk) == idx + 1:
            # no delimiter (hence cannot parse value-key pair from vk)
            # -- or -- missing val (starts with a delim)
            # -- or -- trailing delim, no key (e.g. "val1,val2,")
            raise ValueError(
                f"`{vk}` cannot be split into `val<delim>key` with delims={delims}"
            )
        else:
            return vk[0:idx].strip(), vk[idx + 1 :].strip()

    def to_val(val: str) -> str:
        if (val.startswith("'") and val.endswith("'")) or (
            val.startswith('"') and val.endswith('"')
        ):
            return val[1:-1]
        return val if val != '""' and val != "''" else ""

    arg_map: dict[str, str] = {}

    if not arg:
        return arg_map

    # find quoted values
    quoted_pattern = r'([\'"])((?:\\.|(?!\1).)*?)\1'
    quoted_values: list[str] = []

    def replace_quoted(match):
        quoted_values.append(match.group(0))
        return f"__QUOTED_{len(quoted_values) - 1}__"

    # replace quoted values with placeholders
    processed_arg = re.sub(quoted_pattern, replace_quoted, arg)

    # split cfgs
    cfg_kv_delim = "="

    # ["FOO", "v1;v2,BAR", v3, "BAZ", "v4,v5"]
    split_arg = [
        s.strip() for s in processed_arg.split(cfg_kv_delim) if s.strip()
    ]  # remove empty
    split_arg_len = len(split_arg)

    if split_arg_len < 2:  # no kv -> malformed str
        raise ValueError(f"`{arg}` does not have at least one `key=value` pair")

    # since we split on "=" so we end up with ["KEY1", "val1,KEY2", "val2,KEY_n", "val_n"]
    key = split_arg[0]  # first element is always a key
    # middle elements are value_{n}<delim>key_{n+1}
    for vk in split_arg[1 : split_arg_len - 1]:  # python deals with
        val, key_next = parse_val_key(vk)
        for i, quoted in enumerate(quoted_values):
            val = val.replace(f"__QUOTED_{i}__", quoted)
        arg_map[key] = to_val(val)
        key = key_next

    val = split_arg[-1]  # last element is always a value
    for i, quoted in enumerate(quoted_values):
        val = val.replace(f"__QUOTED_{i}__", quoted)
    arg_map[key] = to_val(val)

    return arg_map


# pyre-ignore-all-errors[3, 2]
def _decode_string_to_dict(
    encoded_value: str, param_type: type[dict[Any, Any]]
) -> dict[Any, Any]:
    # pyre-ignore[16]
    if not hasattr(param_type, "__args__") or len(param_type.__args__) != 2:
        raise ValueError(f"param_type must be a `dict` type, but was `{param_type}`")

    key_type, value_type = param_type.__args__
    arg_values = {}
    for key, value in to_dict(encoded_value).items():
        arg_values[key_type(key)] = value_type(value)
    return arg_values


def _decode_string_to_list(
    encoded_value: str, param_type: type[list[Any]]
) -> list[Any]:
    # pyre-ignore[16]
    if not hasattr(param_type, "__args__") or len(param_type.__args__) != 1:
        raise ValueError(f"param_type must be a `list` type, but was `{param_type}`")
    value_type = param_type.__args__[0]
    if not is_primitive(value_type):
        raise ValueError("List types support only primitives: int, str, float")
    arg_values = []
    for value in to_list(encoded_value):
        arg_values.append(value_type(value))
    return arg_values


def decode(encoded_value: Any, annotation: Any):
    if encoded_value is None:
        return None
    if is_bool(annotation):
        return encoded_value and encoded_value.lower() == "true"
    if not is_primitive(annotation) and type(encoded_value) == str:
        return decode_from_string(encoded_value, annotation)
    return encoded_value


def decode_from_string(
    encoded_value: str, annotation: Any
) -> Union[dict[Any, Any], list[Any], None]:
    """Decodes string representation to the underlying type(Dict or List)

    Given a string representation of the value, the method decodes it according
    to the ``annotation`` type:

    If ``annotation`` is list, the expected format of ``encoded_value`` is:
    value1,value2,..

    If ``annotation`` is dict, the expected format of ``encoded_value`` is:
    key1=value,key2=value2

    Args:
        encoded_value: String encoded complex type, if empty value provided, method
            will return None
        annotation: Complex type. Currently supported list or dict

    Returns:
        List or Dict filled with decoded ``encoded_value``. The type of
        elements in list or dict will be the same as the one in ``annotation``
    """

    if not encoded_value:
        return None
    value_type = annotation
    if hasattr(value_type, "__origin__"):
        value_origin = value_type.__origin__
        if value_origin is dict:
            return _decode_string_to_dict(encoded_value, value_type)
        elif value_origin is list:
            return _decode_string_to_list(encoded_value, value_type)
    raise ValueError("Unknown")


def is_bool(param_type: Any) -> bool:
    """Check if the ``param_type`` is bool"""
    return param_type == bool


def is_primitive(param_type: Any) -> bool:
    """Check if the ``param_type`` belongs to a list of
    (int, str, float)

    Args:
        param_type: Parameter type

    Returns:
        True if ``param_type`` is int, str or float, otherwise False
    """
    TYPES = (int, str, float)
    return param_type in TYPES


def decode_optional(param_type: Any) -> Any:
    """Returns underlying type if optional

    Args:
        param_type: Parameter type

    Returns:
        If ``param_type`` is type Optional[INNER_TYPE], method returns INNER_TYPE
        Otherwise returns ``param_type``
    """

    if not hasattr(param_type, "__origin__"):
        if isinstance(param_type, UnionType):
            # handle BinOp style Optional (e.g. `T | None`)
            if len(param_type.__args__) == 2 and param_type.__args__[1] is type(None):
                return param_type.__args__[0]
            else:
                return param_type
        else:
            return param_type

    if param_type.__origin__ is not Union:
        return param_type

    args = param_type.__args__
    if len(args) == 2 and args[1] is type(None):
        return args[0]
    else:
        return param_type


def get_argparse_param_type(parameter: inspect.Parameter) -> Callable[[str], object]:
    if is_primitive(parameter.annotation):
        return parameter.annotation
    else:
        return str


_T = TypeVar("_T")


def none_throws(optional: Optional[_T], message: str = "Unexpected `None`") -> _T:
    """Convert an optional to its value. Raises an `AssertionError` if the value is `None`
    Copied from https://github.com/facebook/pyre-check/blob/main/pyre_extensions/refinement.py for performance reasons.
    """
    if optional is None:
        raise AssertionError(message)
    return optional
