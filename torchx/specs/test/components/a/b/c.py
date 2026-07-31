# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torchx
from torchx import specs


def d() -> specs.AppDef:
    return specs.AppDef(
        name="foo.b.c.d",
        roles=[
            specs.Role(
                name="foo.b.c.d",
                image=torchx.IMAGE,
                entrypoint="echo",
                args=["hello world"],
            )
        ],
    )
