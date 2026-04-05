# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Test fixture: implicit namespace subpackage (no ``__init__.py``)."""

from torchx.plugins import register
from torchx.specs.api import Resource


@register.named_resource()
def implicit_gpu() -> Resource:
    return Resource(cpu=32, gpu=4, memMB=65536)
