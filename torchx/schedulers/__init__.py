#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
from typing import Mapping, Protocol

from torchx.schedulers.api import Scheduler
from torchx.util.entrypoints import load_group

DEFAULT_SCHEDULER_MODULES: Mapping[str, str] = {
    "local_docker": "torchx.schedulers.docker_scheduler",
    "local_cwd": "torchx.schedulers.local_scheduler",
    "slurm": "torchx.schedulers.slurm_scheduler",
    "kubernetes": "torchx.schedulers.kubernetes_scheduler",
    "kubernetes_mcad": "torchx.schedulers.kubernetes_mcad_scheduler",
    "aws_batch": "torchx.schedulers.aws_batch_scheduler",
    "aws_sagemaker": "torchx.schedulers.aws_sagemaker_scheduler",
    "lsf": "torchx.schedulers.lsf_scheduler",
}


class SchedulerFactory(Protocol):
    # pyre-fixme: Scheduler opts
    def __call__(self, session_name: str, **kwargs: object) -> Scheduler: ...


def _defer_load_scheduler(path: str) -> SchedulerFactory:
    # pyre-ignore[24]: Scheduler opts
    def run(*args: object, **kwargs: object) -> Scheduler:
        module = importlib.import_module(path)
        return module.create_scheduler(*args, **kwargs)

    return run


def get_scheduler_factories(
    group: str = "torchx.schedulers", skip_defaults: bool = False
) -> dict[str, SchedulerFactory]:
    """
    get_scheduler_factories returns all the available schedulers names under `group` and the
    method to instantiate them.

    The first scheduler in the dictionary is used as the default scheduler.
    """

    if skip_defaults:
        default_schedulers = {}
    else:
        default_schedulers: dict[str, SchedulerFactory] = {}
        for scheduler, path in DEFAULT_SCHEDULER_MODULES.items():
            default_schedulers[scheduler] = _defer_load_scheduler(path)

    return load_group(group, default=default_schedulers)


def get_default_scheduler_name() -> str:
    """
    default_scheduler_name returns the first scheduler defined in
    get_scheduler_factories.
    """
    return next(iter(get_scheduler_factories().keys()))
