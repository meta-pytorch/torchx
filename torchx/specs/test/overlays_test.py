# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

from torchx.specs.overlays import apply_overlay


class OverlaysTest(unittest.TestCase):
    def test_apply_overlay_empty_base(self) -> None:
        base = {}
        overlay = {
            "a0": {"b": "c"},
            "a1": ["b", "c"],
            "a2": "b",
        }
        apply_overlay(base, overlay)
        self.assertDictEqual(base, overlay)

    def test_apply_overlay_dict_attr(self) -> None:
        base = {"a0": {"d": "e"}}
        overlay = {"a0": {"b": "c"}}

        apply_overlay(base, overlay)
        self.assertDictEqual(
            base,
            {
                "a0": {
                    "b": "c",
                    "d": "e",
                },
            },
        )

        base = {"a0": {"b": "d"}}
        apply_overlay(base, overlay)
        self.assertDictEqual(base, {"a0": {"b": "c"}})

    def test_apply_overlay_list_attr(self) -> None:
        base = {"a0": []}
        overlay = {"a0": ["b", "c"]}

        apply_overlay(base, overlay)
        self.assertDictEqual(base, {"a0": ["b", "c"]})

        base = {"a0": ["1", "b", "2"]}
        apply_overlay(base, overlay)
        # lists simply append - they do not dedup
        self.assertDictEqual(base, {"a0": ["1", "b", "2", "b", "c"]})

        base = {"a0": ["1", ["2", "3"]]}
        overlay = {"a0": ["b", ["c", "d"]]}
        apply_overlay(base, overlay)
        # lists simply append - they do NOT recusively apply
        self.assertDictEqual(base, {"a0": ["1", ["2", "3"], "b", ["c", "d"]]})

    def test_overlay_type_mismatch(self) -> None:

        with self.assertRaises(AssertionError):
            apply_overlay({"a": [1, 2]}, {"a": "b"})

        with self.assertRaises(AssertionError):
            apply_overlay({"a": {"b": 1}}, {"a": {"b": "c"}})
