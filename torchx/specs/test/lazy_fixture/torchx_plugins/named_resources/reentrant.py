# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Re-entrancy fixture: a namespace plugin that imports ``torchx.specs`` at
module top-level. Discovery is triggered from inside ``torchx.specs`` (first
named-resource lookup), so this import re-enters the already-initialized
module — it must resolve cleanly, not crash or deadlock the scan."""

import torchx.specs  # noqa: F401  (the top-level re-entrant import IS the fixture)
from torchx.plugins import register
from torchx.specs.api import Resource


@register.named_resource()
def reentrant_gpu() -> Resource:
    return Resource(cpu=1, gpu=1, memMB=1024)
