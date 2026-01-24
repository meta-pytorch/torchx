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
import sys

from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.specs.api import parse_app_handle

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


class CmdStatus(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "app_handle",
            type=str,
            help="torchx app handle (e.g. local://session-name/app-id)",
        )
        subparser.add_argument(
            "--roles", type=str, default="", help="comma separated roles to filter"
        )
        subparser.add_argument(
            "--json",
            action="store_true",
            help="output the status in JSON format",
        )

    def run(self, args: argparse.Namespace) -> None:
        app_handle = args.app_handle
        scheduler, _, app_id = parse_app_handle(app_handle)
        runner = get_runner()
        app_status = runner.status(app_handle)
        filter_roles = parse_list_arg(args.roles)
        if app_status:
            if args.json:
                print(json.dumps(app_status.to_json(filter_roles)))
            else:
                print(app_status.format(filter_roles))
        else:
            logger.error(
                f"AppDef: {app_id},"
                f" does not exist or has been removed from {scheduler}'s data plane"
            )
            sys.exit(1)
