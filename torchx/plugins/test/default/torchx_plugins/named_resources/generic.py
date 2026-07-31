# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""Generic named resources for testing namespace package discovery."""

from torchx.plugins import halve_mem_down_to, powers_of_two_gpus, register, WHOLE
from torchx.specs.api import Resource

GiB: int = 1024


@register.named_resource(aliases=["t4g"], fractionals=powers_of_two_gpus)
def gpu(fractional: float = WHOLE) -> Resource:
    return Resource(
        cpu=int(64 * fractional),
        gpu=int(8 * fractional),
        memMB=int(1024 * GiB * fractional),
    )


@register.named_resource(fractionals=halve_mem_down_to(minGiB=8))
def cpu(fractional: float = WHOLE) -> Resource:
    return Resource(
        cpu=int(16 * fractional),
        gpu=int(2 * fractional),
        memMB=int(64 * GiB * fractional),
    )
