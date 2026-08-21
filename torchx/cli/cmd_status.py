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
from torchx.specs.api import AppStatus, parse_app_handle

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
    """``torchx status`` -- prints the status of an app.

    With ``--json`` stdout carries exactly one JSON object with a stable
    schema (keys are only ever added): ``{"handle": str, "app_id": str,
    "scheduler": str, "state": str, "num_restarts": int, "roles": list,
    "msg": str, "structured_error_msg": str, "url": str | None,
    "ui_url": str | None}``. ``url`` and ``ui_url`` carry the same value;
    ``ui_url`` is the canonical name.
    """

    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        super().add_arguments(subparser)
        subparser.add_argument(
            "--roles", type=str, default="", help="comma separated roles to filter"
        )
        subparser.add_argument(
            "--json",
            action="store_true",
            help="output the status as a single machine-readable JSON object"
            " (see the command docstring for the schema)",
        )

    def run_with_runner(self, args: argparse.Namespace, runner: Runner) -> None:
        self.print_status(args, runner.status(args.app_handle))

    def print_status(
        self, args: argparse.Namespace, app_status: AppStatus | None
    ) -> None:
        filter_roles = parse_list_arg(args.roles)
        if app_status:
            if args.json:
                scheduler, _, app_id = parse_app_handle(args.app_handle)
                status_json = {
                    **app_status.to_json(filter_roles),
                    "handle": args.app_handle,
                    "app_id": app_id,
                    "scheduler": scheduler,
                    "ui_url": app_status.ui_url,
                }
                print(json.dumps(status_json))
            else:
                print(app_status.format(filter_roles))
        else:
            self.exit_missing_app(args.app_handle)
