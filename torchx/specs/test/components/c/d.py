# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torchx
from torchx import specs


def e() -> specs.AppDef:
    return specs.AppDef(
        name="bar.e",
        roles=[
            specs.Role(
                name="bar.e",
                image=torchx.IMAGE,
                entrypoint="echo",
                args=["hello world"],
            )
        ],
    )
