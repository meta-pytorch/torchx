# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch


def has_cuda_devices() -> bool:
    """
    Checks if the host that is running this function has CUDA (GPU) devices
    and that the installed version of PyTorch is CUDA-capable (PyTorch can be installed
    as CPU-only). If this function returns ``True`` then it means that the caller
    can actually allocate tensors on the GPU.

    .. note::
        ``torch.cuda.is_available()`` returns ``True`` if a GPU-capable
         version of torch is installed (even on a CPU-only host).
         So that call alone will not tell us whether we can actually
         put tensors and modules on a CUDA device.
         Hence this method checks that the host has both:
         1) GPU-capable version of torch installed
         2) Actually has at least one physical CUDA device

    """

    #
    return torch.cuda.is_available() and torch.cuda.device_count() > 0
