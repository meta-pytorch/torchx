# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import sys
from argparse import ArgumentParser
from typing import Dict, List

import torchx
from torchx.cli.cmd_base import SubCommand
from torchx.cli.cmd_cancel import CmdCancel
from torchx.cli.cmd_configure import CmdConfigure
from torchx.cli.cmd_delete import CmdDelete
from torchx.cli.cmd_describe import CmdDescribe
from torchx.cli.cmd_list import CmdList
from torchx.cli.cmd_log import CmdLog
from torchx.cli.cmd_run import CmdBuiltins, CmdRun
from torchx.cli.cmd_runopts import CmdRunopts
from torchx.cli.cmd_status import CmdStatus
from torchx.cli.cmd_tracker import CmdTracker
from torchx.cli.colors import BLUE, ENDC, GRAY
from torchx.util.entrypoints import load_group


sub_parser_description = """Use the following commands to run operations, e.g.:
torchx run ${JOB_NAME}
"""


def get_default_sub_cmds() -> Dict[str, SubCommand]:
    return {
        "builtins": CmdBuiltins(),
        "cancel": CmdCancel(),
        "configure": CmdConfigure(),
        "delete": CmdDelete(),
        "describe": CmdDescribe(),
        "list": CmdList(),
        "log": CmdLog(),
        "run": CmdRun(),
        "runopts": CmdRunopts(),
        "status": CmdStatus(),
        "tracker": CmdTracker(),
    }


def get_sub_cmds() -> Dict[str, SubCommand]:
    """
    Find available subcommands for `torchx cli`.
    The method consits of two parts:
    1. Fetch default commands
    2. Find any commands in the entrypoints.txt under the `torchx.cli.cmds` group

    The commands defined in `torchx.cli.cmds` take priority, meaning that if
    the same command is defined in defaults and `torchx.cli.cmds`, the one from
    entrypoints will take priority.

    """
    sub_cmds = get_default_sub_cmds()

    override_sub_cmds = load_group(
        "torchx.cli.cmds",
        default={},
    )
    for cmd_name, cmd_cls in override_sub_cmds.items():
        sub_cmds[cmd_name] = cmd_cls()
    return sub_cmds


def create_parser(subcmds: Dict[str, SubCommand]) -> ArgumentParser:
    """
    Helper function parsing the command line options.
    """

    parser = ArgumentParser(description="torchx CLI")
    parser.add_argument(
        "--log_level",
        type=str,
        help="Python logging log level",
        default=os.getenv("LOGLEVEL", "INFO"),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="torchx-{version}".format(version=torchx.version.__version__),
    )
    subparser = parser.add_subparsers(
        title="sub-commands",
        description=sub_parser_description,
    )

    for subcmd_name, cmd in subcmds.items():
        cmd_parser = subparser.add_parser(subcmd_name)
        cmd.add_arguments(cmd_parser)
        cmd_parser.set_defaults(func=cmd.run)

    return parser


def run_main(subcmds: Dict[str, SubCommand], argv: List[str] = sys.argv[1:]) -> None:
    parser = create_parser(subcmds)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format=f"{BLUE}torchx{ENDC} {GRAY}%(asctime)s %(levelname)-8s{ENDC} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if "func" not in args:
        parser.print_help()
        sys.exit(1)
    args.func(args)


def main(argv: List[str] = sys.argv[1:]) -> None:
    run_main(get_sub_cmds(), argv)


if __name__ == "__main__":
    main()
