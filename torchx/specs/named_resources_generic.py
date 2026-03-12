# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Defines generic named resources that are not specific to any cloud provider's
instance types. These generic named resources are meant to be used as
default values for components and examples and are NOT meant to be used
long term as the specific capabilities (e.g. number of cpu, gpu, memMB)
are subject to change.

.. note::
    The named resources in this file DO NOT map device capabilities such as
    special network interfaces (e.g. EFA devices on AWS).
.. warning::
    Do not use for launching applications that require specific capabilities
    (e.g. needs exactly 4 x A100 GPUs with 40GB of memory connected with NVLink).

Different cloud provides offer different types of
instance types hence practically speaking one should register their own
named resources that accurately capture the instances they have at their disposal
rather than using these defaults long term.

.. note::
    The cpu/gpu/memory ratios in these default resources are based on current
    HW trends and do not map exactly to a particular instance type!

.. warning::
    The specific capabilities of these default resources are subject to change
    at any time based on current hardware spec trends.
    Therefore, the user should NEVER assume that the specific number of cpu, gpu, and memMB
    will always remain the same. For instance, never assume that ``gpu.small`` will always
    have 8 cpus.

"""
from typing import Callable, Mapping

from torchx.specs.api import Resource

GiB: int = 1024

NAMED_RESOURCES: Mapping[str, Callable[[], Resource]] = {
    # typically system CPU memory is >= GPU memory (most modern GPUs have 32GB device mem)
    # most cloud provides offer 1, 2, 4, 8 GPUs per host
    "gpu.small": lambda: Resource(cpu=8, gpu=1, memMB=32 * GiB),
    "gpu.medium": lambda: Resource(cpu=16, gpu=2, memMB=64 * GiB),
    "gpu.large": lambda: Resource(cpu=32, gpu=4, memMB=128 * GiB),
    "gpu.xlarge": lambda: Resource(cpu=64, gpu=8, memMB=256 * GiB),
    # for cpu defaults - based on AWS's T2 (general purpose) instance type
    "cpu.nano": lambda: Resource(cpu=1, gpu=0, memMB=512),
    "cpu.micro": lambda: Resource(cpu=1, gpu=0, memMB=1 * GiB),
    "cpu.small": lambda: Resource(cpu=1, gpu=0, memMB=2 * GiB),
    "cpu.medium": lambda: Resource(cpu=2, gpu=0, memMB=4 * GiB),
    "cpu.large": lambda: Resource(cpu=2, gpu=0, memMB=8 * GiB),
    "cpu.xlarge": lambda: Resource(cpu=8, gpu=0, memMB=32 * GiB),
}
