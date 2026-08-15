#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging

from torchx.cli.cmd_base import AppHandleSubCommand
from torchx.runner import Runner

logger: logging.Logger = logging.getLogger(__name__)


class CmdDelete(AppHandleSubCommand):
    def run_with_runner(self, args: argparse.Namespace, runner: Runner) -> None:
        runner.delete(args.app_handle)
