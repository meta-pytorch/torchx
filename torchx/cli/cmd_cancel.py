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
from torchx.runner import get_runner

logger: logging.Logger = logging.getLogger(__name__)


class CmdCancel(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "app_handle",
            type=str,
            help="torchx app handle (e.g. local://session-name/app-id)",
        )

    def run(self, args: argparse.Namespace) -> None:
        app_handle = args.app_handle
        runner = get_runner()
        runner.cancel(app_handle)
