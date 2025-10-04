# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from configparser import ConfigParser

from importlib.metadata import EntryPoint
from types import ModuleType

from unittest.mock import MagicMock, patch

from torchx.util.entrypoints import load, load_group


def EntryPoint_from_config(config: ConfigParser) -> list[EntryPoint]:
    # from stdlib, Copyright (c) Python Authors
    return [
        EntryPoint(name, value, group)
        for group in config.sections()
        for name, value in config.items(group)
    ]


def EntryPoint_from_text(text: str) -> list[EntryPoint]:
    # from stdlib, Copyright (c) Python Authors
    config = ConfigParser(delimiters="=")
    config.read_string(text)
    return EntryPoint_from_config(config)


def foobar() -> str:
    return "foobar"


def barbaz() -> str:
    return "barbaz"


_EP_TXT: str = """
[entrypoints.test]
foo = torchx.util.test.entrypoints_test:foobar
"""

_EP_GRP_TXT: str = """
[ep.grp.test]
foo = torchx.util.test.entrypoints_test:foobar
bar = torchx.util.test.entrypoints_test:barbaz
"""

_EP_GRP_IGN_ATTR_TXT: str = """
[ep.grp.missing.attr.test]
baz = torchx.util.test.entrypoints_test:missing_attr
"""

_EP_GRP_MOD_TXT: str = """
[ep.grp.mod.test]
baz = torchx.util.test.entrypoints_test
"""

_EP_GRP_IGN_MOD_TXT: str = """
[ep.grp.missing.mod.test]
baz = torchx.util.test.entrypoints_test.missing_module
"""

_EPS: list[EntryPoint] = (
    EntryPoint_from_text(_EP_TXT)
    + EntryPoint_from_text(_EP_GRP_TXT)
    + EntryPoint_from_text(_EP_GRP_IGN_ATTR_TXT)
    + EntryPoint_from_text(_EP_GRP_MOD_TXT)
    + EntryPoint_from_text(_EP_GRP_IGN_MOD_TXT)
)

try:
    from importlib.metadata import EntryPoints
except ImportError:
    # python<=3.9
    _ENTRY_POINTS: dict[str, list[EntryPoint]] = {}
    for ep in _EPS:
        _ENTRY_POINTS.setdefault(ep.group, []).append(ep)
else:
    # python>=3.10
    _ENTRY_POINTS: EntryPoints = EntryPoints(_EPS)

_METADATA_EPS: str = "torchx.util.entrypoints.metadata.entry_points"


class EntryPointsTest(unittest.TestCase):
    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load(self, _: MagicMock) -> None:
        print(type(load("entrypoints.test", "foo")))
        self.assertEqual("foobar", load("entrypoints.test", "foo")())

        with self.assertRaisesRegex(KeyError, "baz"):
            load("entrypoints.test", "baz")()

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_with_default(self, _: MagicMock) -> None:
        self.assertEqual("barbaz", load("entrypoints.test", "missing", barbaz)())
        self.assertEqual("barbaz", load("entrypoints.missing", "foo", barbaz)())
        self.assertEqual("barbaz", load("entrypoints.missing", "missing", barbaz)())

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group(self, _: MagicMock) -> None:
        eps = load_group("ep.grp.test")
        self.assertEqual(2, len(eps), eps)
        self.assertEqual("foobar", eps["foo"]())
        self.assertEqual("barbaz", eps["bar"]())

        eps = load_group("ep.grp.test.missing")
        self.assertIsNone(eps)

        eps = load_group("ep.grp.mod.test")
        module = eps["baz"]()
        self.assertEqual(ModuleType, type(module))
        self.assertEqual("torchx.util.test.entrypoints_test", module.__name__)

        # module's deferred load function should ignore *args and **kwargs
        self.assertEqual(module, eps["baz"]("ignored", should="ignore"))

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group_with_default(self, _: MagicMock) -> None:
        eps = load_group("ep.grp.test", {"foo": barbaz, "bar": foobar})
        self.assertEqual(2, len(eps))
        self.assertEqual("foobar", eps["foo"]())
        self.assertEqual("barbaz", eps["bar"]())

        eps = load_group("ep.grp.test.missing", {"foo": barbaz, "bar": foobar})
        self.assertEqual(2, len(eps))
        self.assertEqual("barbaz", eps["foo"]())
        self.assertEqual("foobar", eps["bar"]())

        eps = load_group(
            "ep.grp.test.missing", {"foo": barbaz, "bar": foobar}, skip_defaults=True
        )
        self.assertIsNone(eps)

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group_missing(self, _: MagicMock) -> None:
        with self.assertRaises(AttributeError):
            load_group("ep.grp.missing.attr.test")["baz"]()

        with self.assertRaises(ModuleNotFoundError):
            load_group("ep.grp.missing.mod.test")["baz"]()
