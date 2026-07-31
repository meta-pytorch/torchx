#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torchx.specs as specs
from torchx.components.serve import torchserve


class ServeTest(unittest.TestCase):
    def test_torchserve(self) -> None:
        want = specs.AppDef(
            name="torchx-torchserve",
            roles=[
                specs.Role(
                    name="worker",
                    image="torchx:latest",
                    entrypoint="python",
                    args=[
                        "-m",
                        "torchx.apps.serve.serve",
                        "--model_path",
                        "the_model_path",
                        "--management_api",
                        "http://localhost:1234",
                        "--initial_workers",
                        "1",
                    ],
                    port_map={"model-download": 8222},
                    resource=specs.Resource(cpu=1, gpu=0, memMB=1024),
                ),
            ],
        )
        out = torchserve(
            "the_model_path",
            "http://localhost:1234",
            image="torchx:latest",
            params={
                "initial_workers": "1",
            },
        )
        self.assertEqual(out, want)
