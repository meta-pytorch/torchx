# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""GCP named resources for testing cross-package discovery."""

from torchx.plugins import register
from torchx.specs.api import Resource


@register.named_resource()
def gcp_a3_highgpu_8g(fractional: float = 1.0) -> Resource:
    """GCP a3-highgpu-8g — 8x H100 80GB, 252 vCPU, 2048 GB RAM."""
    gpu = int(8 * fractional)
    return Resource(cpu=252, gpu=gpu, memMB=2097152)
