#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import dataclasses
import logging
import pprint

from torchx.cli.cmd_base import AppHandleSubCommand
from torchx.runner import Runner

logger: logging.Logger = logging.getLogger(__name__)


class CmdDescribe(AppHandleSubCommand):
    def run_with_runner(self, args: argparse.Namespace, runner: Runner) -> None:
        app = runner.describe(args.app_handle)

        if app:
            pprint.pprint(dataclasses.asdict(app), indent=2, width=80)
        else:
            self.exit_missing_app(args.app_handle)
