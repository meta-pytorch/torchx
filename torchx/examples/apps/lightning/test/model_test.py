# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchx.examples.apps.lightning.model import TinyImageNetModel


class ModelTest(unittest.TestCase):
    def test_lr(self) -> None:
        model = TinyImageNetModel(lr=0.1234)
        self.assertEqual(model.configure_optimizers().defaults["lr"], 0.1234)

    def test_layer_sizes(self) -> None:
        model = TinyImageNetModel(
            layer_sizes=[
                1,
                2,
                1,
                1,
            ],
        )
        out = model(torch.zeros((1, 3, 64, 64)))
        self.assertIsNotNone(out)
