#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import importlib
from typing import Mapping, Protocol

from torchx import plugins
from torchx.schedulers.api import Scheduler

DEFAULT_SCHEDULER_MODULES: Mapping[str, str] = {
    "local_docker": "torchx.schedulers.docker_scheduler",
    "local_cwd": "torchx.schedulers.local_scheduler",
    "slurm": "torchx.schedulers.slurm_scheduler",
    "kubernetes": "torchx.schedulers.kubernetes_scheduler",
}


class SchedulerFactory(Protocol):
    def __call__(self, session_name: str, **kwargs: object) -> Scheduler: ...


def _defer_load_scheduler(path: str) -> SchedulerFactory:
    def run(*args: object, **kwargs: object) -> Scheduler:
        module = importlib.import_module(path)
        return module.create_scheduler(*args, **kwargs)

    return run


def get_scheduler_factories(
    *, skip_defaults: bool = False
) -> dict[str, SchedulerFactory]:
    """
    get_scheduler_factories returns all the available schedulers names and the
    method to instantiate them: the built-ins plus everything registered
    through :py:mod:`torchx.plugins`, which wins a name clash.

    The first scheduler in the dictionary is used as the default scheduler,
    and registered plugins come first.

    Pass ``skip_defaults=True`` for the registered plugins only.
    """

    if skip_defaults:
        default_schedulers = {}
    else:
        default_schedulers: dict[str, SchedulerFactory] = {}
        for scheduler, path in DEFAULT_SCHEDULER_MODULES.items():
            default_schedulers[scheduler] = _defer_load_scheduler(path)

    factories: dict[str, SchedulerFactory] = dict(
        plugins.registry().get(plugins.PluginType.SCHEDULER)
    )
    for name, factory in default_schedulers.items():
        factories.setdefault(name, factory)
    return factories


def get_default_scheduler_name() -> str:
    """
    default_scheduler_name returns the first scheduler defined in
    get_scheduler_factories.
    """
    return next(iter(get_scheduler_factories().keys()))
