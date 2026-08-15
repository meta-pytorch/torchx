#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from torchx.specs import CapabilityKey, Resource

NETWORK_BANDWIDTH_GBPS: CapabilityKey[int] = CapabilityKey(
    "aws", "network_bandwidth_gbps", int
)


class CapabilityKeyTest(unittest.TestCase):
    def test_golden_key_spelling(self) -> None:
        # the plain string key is the wire format — assert the literal
        self.assertEqual(
            "aws.network_bandwidth_gbps",
            NETWORK_BANDWIDTH_GBPS.key,
            "wire key must be `<namespace>.<name>`",
        )

    def test_set_get_round_trip(self) -> None:
        res = Resource(cpu=4, gpu=0, memMB=1024)
        NETWORK_BANDWIDTH_GBPS.set(res, 400)

        self.assertEqual(
            {"aws.network_bandwidth_gbps": 400},
            res.capabilities,
            "set() must store under the plain string key (wire format)",
        )
        self.assertEqual(
            400,
            NETWORK_BANDWIDTH_GBPS.get(res),
            "get() must round-trip the stored value",
        )

    def test_get_default(self) -> None:
        res = Resource(cpu=4, gpu=0, memMB=1024)

        self.assertIsNone(
            NETWORK_BANDWIDTH_GBPS.get(res),
            "missing capability defaults to None",
        )
        self.assertEqual(
            100,
            NETWORK_BANDWIDTH_GBPS.get(res, default=100),
            "missing capability returns the passed default",
        )

    def test_get_type_mismatch_raises_naming_the_key(self) -> None:
        res = Resource(cpu=4, gpu=0, memMB=1024)
        res.capabilities["aws.network_bandwidth_gbps"] = "400"

        with self.assertRaisesRegex(
            TypeError,
            r"aws\.network_bandwidth_gbps.*expected `int`.*got `str`",
        ):
            NETWORK_BANDWIDTH_GBPS.get(res)

    def test_get_rejects_bool_for_int_typed_key(self) -> None:
        res = Resource(cpu=4, gpu=0, memMB=1024)
        res.capabilities["aws.network_bandwidth_gbps"] = True

        with self.assertRaisesRegex(
            TypeError,
            r"aws\.network_bandwidth_gbps.*expected `int`.*got `bool`",
        ):
            NETWORK_BANDWIDTH_GBPS.get(res)

    def test_get_bool_typed_key_accepts_bool(self) -> None:
        efa_enabled: CapabilityKey[bool] = CapabilityKey("aws", "efa_enabled", bool)
        res = Resource(cpu=4, gpu=0, memMB=1024)
        efa_enabled.set(res, True)

        self.assertIs(
            True,
            efa_enabled.get(res),
            "bool-typed key must round-trip a stored bool",
        )

    def test_resource_copy_preserves_capability(self) -> None:
        res = Resource(cpu=4, gpu=0, memMB=1024)
        NETWORK_BANDWIDTH_GBPS.set(res, 400)

        copied = Resource.copy(res)

        self.assertEqual(
            400,
            NETWORK_BANDWIDTH_GBPS.get(copied),
            "Resource.copy must preserve capabilities set via CapabilityKey",
        )
