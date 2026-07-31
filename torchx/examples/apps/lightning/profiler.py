#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Simple Logging Profiler
===========================

This is a simple profiler that's used as part of the trainer app example. This
logs the Lightning training stage durations a logger such as Tensorboard. This
output is used for HPO optimization with Ax.
"""

import time

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.profilers.profiler import Profiler


class SimpleLoggingProfiler(Profiler):
    """
    This profiler records the duration of actions (in seconds) and reports the
    mean duration of each action to the specified logger. Reported metrics are
    in the format `duration_<event>`.
    """

    def __init__(self, logger: Logger) -> None:
        super().__init__()

        self.current_actions: dict[str, float] = {}
        self.logger = logger

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.logger.log_metrics({"duration_" + action_name: duration})

    def summary(self) -> str:
        return ""


# sphinx_gallery_thumbnail_path = '_static/img/gallery-lib.png'
