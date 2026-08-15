#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging

from torchx.cli.cmd_base import AppHandleSubCommand
from torchx.runner import Runner

logger: logging.Logger = logging.getLogger(__name__)


_ROLE_FORMAT_TEMPLATE = "\n  ${role}:${replicas}"

_REPLICA_FORMAT_TEMPLATE_DETAILED = """\n  ${role}[${replica_id}]:
    state: ${state}
    timestamp: ${timestamp} (exit_code: ${exit_code})
    hostname: ${hostname}
    error_msg: ${error_msg}"""

_LINE_WIDTH = 100


def parse_list_arg(arg: str) -> list[str] | None:
    if not arg:
        return None
    return arg.split(",")


class CmdStatus(AppHandleSubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        super().add_arguments(subparser)
        subparser.add_argument(
            "--roles", type=str, default="", help="comma separated roles to filter"
        )
        subparser.add_argument(
            "--json",
            action="store_true",
            help="output the status in JSON format",
        )

    def run_with_runner(self, args: argparse.Namespace, runner: Runner) -> None:
        app_status = runner.status(args.app_handle)
        filter_roles = parse_list_arg(args.roles)
        if app_status:
            if args.json:
                print(json.dumps(app_status.to_json(filter_roles)))
            else:
                print(app_status.format(filter_roles))
        else:
            self.exit_missing_app(args.app_handle)
