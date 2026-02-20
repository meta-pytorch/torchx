#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging

from torchx.cli.cmd_base import SubCommand
from torchx.runner.api import get_runner
from torchx.util.colors import ENDC, GREEN

logger: logging.Logger = logging.getLogger(__name__)


class CmdRunopts(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "scheduler",
            type=str,
            nargs="?",
            help="scheduler to dump the runopts for, dumps for all schedulers if not specified",
        )

    def run(self, args: argparse.Namespace) -> None:
        filter = args.scheduler
        with get_runner() as runner:
            for scheduler in runner.scheduler_backends():
                if filter and scheduler != filter:
                    continue
                try:
                    opts = runner.scheduler_run_opts(scheduler)
                    print(f"{GREEN}{scheduler}{ENDC}:\n{repr(opts)}\n")
                except ModuleNotFoundError as e:
                    print(f"{GREEN}{scheduler}{ENDC}: {e}\n")
