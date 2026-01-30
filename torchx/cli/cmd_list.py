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
from torchx.cli.argparse_util import ArgOnceAction, torchxconfig_list
from torchx.cli.cmd_base import SubCommand
from torchx.runner import config, get_runner
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.specs import CfgVal

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
            action=torchxconfig_list,
            help=f"Name of the scheduler to use. One of: [{','.join(scheduler_names)}].",
        )
        subparser.add_argument(
            "-cfg",
            "--scheduler_args",
            type=str,
            action=ArgOnceAction,
            help="Arguments to pass to the scheduler (Ex: `cluster=foo,user=bar`)."
            " For a list of scheduler run options run: `torchx runopts`",
        )

    def run(self, args: argparse.Namespace) -> None:
        with get_runner() as runner:
            # Parse scheduler config from CLI args first (takes precedence)
            cfg: dict[str, CfgVal] = {}
            if args.scheduler_args:
                scheduler_opts = runner.scheduler_run_opts(args.scheduler)
                cfg = scheduler_opts.cfg_from_str(args.scheduler_args)

            # Fill in gaps from .torchxconfig (doesn't override CLI args)
            config.apply(scheduler=args.scheduler, cfg=cfg)

            apps = runner.list(args.scheduler, cfg if cfg else None)
            apps_data = [[app.app_handle, app.name, str(app.state)] for app in apps]
            print(
                tabulate(apps_data, headers=[HANDLE_HEADER, NAME_HEADER, STATUS_HEADER])
            )
