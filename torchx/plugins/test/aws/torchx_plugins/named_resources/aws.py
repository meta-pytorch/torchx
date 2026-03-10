# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""AWS named resources for testing cross-package discovery."""

from torchx.plugins import register
from torchx.specs.api import Resource


@register.named_resource()
def aws_p5_48xlarge(fractional: float = 1.0) -> Resource:
    """AWS p5.48xlarge — 8x H100 80GB, 192 vCPU, 2048 GB RAM."""
    gpu = int(8 * fractional)
    return Resource(cpu=192, gpu=gpu, memMB=2097152)
