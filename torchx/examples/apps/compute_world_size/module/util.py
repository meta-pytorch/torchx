#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig


def compute_world_size(cfg: DictConfig) -> int:
    # required env vars for initializing pg with the default init_method (env://)
    # read from hydra config in config/defaults.yaml if not set already
    # this can happen is compute_world_size is run directly (not with torchrun)
    os.environ.setdefault("RANK", str(cfg.main.rank))
    os.environ.setdefault("WORLD_SIZE", str(cfg.main.world_size))
    os.environ.setdefault("MASTER_ADDR", cfg.main.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.main.master_port))

    backend = cfg.main.backend

    print(f"initializing `{backend}` process group")
    dist.init_process_group(backend=backend)
    print("successfully initialized process group")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    t = F.one_hot(torch.tensor(rank), num_classes=world_size)
    dist.all_reduce(t)
    computed_world_size = int(torch.sum(t).item())
    print(
        f"rank: {rank}, actual world_size: {world_size}, computed world_size: {computed_world_size}"
    )
    return computed_world_size
