# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest import mock

from torchx.components.structured_arg import StructuredJArgument, StructuredNameArgument

WARNINGS_WARN = "torchx.components.structured_arg.warnings.warn"


class ArgNameTest(unittest.TestCase):
    def test_create(self) -> None:
        # call by module
        self.assertEqual(
            StructuredNameArgument("default-experiment", "baz"),
            StructuredNameArgument.parse_from(name="/", m="bar.baz"),
        )
        self.assertEqual(
            StructuredNameArgument("default-experiment", "foo"),
            StructuredNameArgument.parse_from(name="/foo", m="bar.baz"),
        )
        self.assertEqual(
            StructuredNameArgument("foo", "baz"),
            StructuredNameArgument.parse_from(name="foo/", m="bar.baz"),
        )
        self.assertEqual(
            StructuredNameArgument("foo", "bar"),
            StructuredNameArgument.parse_from(name="foo/bar", m="bar.baz"),
        )

        # call by script
        self.assertEqual(
            StructuredNameArgument("default-experiment", "baz"),
            StructuredNameArgument.parse_from(name="/", script="bar/baz.py"),
        )
        self.assertEqual(
            StructuredNameArgument("default-experiment", "foo"),
            StructuredNameArgument.parse_from(name="/foo", script="bar/baz.py"),
        )
        self.assertEqual(
            StructuredNameArgument("default-experiment", "foo"),
            StructuredNameArgument.parse_from(name="/foo", script="bar/baz.py"),
        )
        self.assertEqual(
            StructuredNameArgument("foo", "baz"),
            StructuredNameArgument.parse_from(name="foo/", script="bar/baz.py"),
        )
        self.assertEqual(
            StructuredNameArgument("foo", "bar"),
            StructuredNameArgument.parse_from(name="foo/bar", script="bar/baz.py"),
        )

        self.assertEqual(
            StructuredNameArgument("default-experiment", "foobar"),
            StructuredNameArgument.parse_from(name="foobar", m="bar.baz"),
        )

        with self.assertRaisesRegex(ValueError, "No main module or script specified"):
            StructuredNameArgument.parse_from(name="foo/bar")

        with self.assertRaisesRegex(ValueError, "Both main module and script set"):
            StructuredNameArgument.parse_from(name="foo/bar", m="a.b", script="a/b.py")


class ArgJTest(unittest.TestCase):
    def test_create(self) -> None:
        self.assertEqual(
            StructuredJArgument(nnodes=2, nproc_per_node=8),
            StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2"),
        )
        self.assertEqual(
            StructuredJArgument(nnodes=2, nproc_per_node=4),
            StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2x4"),
        )
        self.assertEqual(
            StructuredJArgument(nnodes=2, nproc_per_node=16),
            StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2x16"),
        )
        self.assertEqual(
            StructuredJArgument(nnodes=2, nproc_per_node=8),
            StructuredJArgument.parse_from(h="aws_trn1.2xlarge", j="2x8"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "nproc_per_node cannot be inferred from GPU count. `aws_trn1.32xlarge` is not a GPU instance.",
        ):
            StructuredJArgument.parse_from(h="aws_trn1.32xlarge", j="2")

        with self.assertRaisesRegex(ValueError, "Invalid format for `-j"):
            StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2x2x2")

        with mock.patch(WARNINGS_WARN) as warn_mock:
            StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2x4")
            warn_mock.assert_called()

        with mock.patch(WARNINGS_WARN) as warn_mock:
            StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2x9")
            warn_mock.assert_called()
