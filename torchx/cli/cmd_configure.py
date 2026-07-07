#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import sys

from torchx.cli.cmd_base import SubCommand
from torchx.runner.config import dump
from torchx.schedulers import get_scheduler_factories


logger: logging.Logger = logging.getLogger(__name__)


class CmdConfigure(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "-s",
            "--schedulers",
            type=str,
            help="comma delimited list of schedulers to dump runopts for, if not specified, dumps for all schedulers",
        )
        subparser.add_argument(
            "--print",
            action="store_true",
            help="if specified, prints the config file to stdout instead of saving it to a file",
        )
        subparser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="if specified, includes required and optional runopts (default only dumps required)",
        )

    def run(self, args: argparse.Namespace) -> None:
        if args.schedulers:
            schedulers = args.schedulers.split(",")
        else:
            schedulers = get_scheduler_factories().keys()

        required_only = not args.all

        if args.print:
            # pyrefly: ignore [bad-argument-type]
            dump(f=sys.stdout, schedulers=schedulers, required_only=required_only)
        else:
            with open(".torchxconfig", "w") as f:
                # pyrefly: ignore [bad-argument-type]
                dump(f=f, schedulers=schedulers, required_only=required_only)
