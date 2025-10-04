#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging

from tabulate import tabulate

from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories

logger: logging.Logger = logging.getLogger(__name__)


HANDLE_HEADER = "APP HANDLE"
STATUS_HEADER = "APP STATUS"
NAME_HEADER = "APP NAME"


class CmdList(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        scheduler_names = get_scheduler_factories().keys()
        subparser.add_argument(
            "-s",
            "--scheduler",
            type=str,
            default=get_default_scheduler_name(),
            choices=list(scheduler_names),
            help=f"Name of the scheduler to use. One of: [{','.join(scheduler_names)}].",
        )

    def run(self, args: argparse.Namespace) -> None:
        with get_runner() as runner:
            apps = runner.list(args.scheduler)
            apps_data = [[app.app_handle, app.name, str(app.state)] for app in apps]
            print(
                tabulate(apps_data, headers=[HANDLE_HEADER, NAME_HEADER, STATUS_HEADER])
            )
