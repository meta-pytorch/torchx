#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import re
import sys
import threading
import time
from queue import Queue
from typing import TextIO

from torchx import specs
from torchx.cli.cmd_base import SubCommand
from torchx.cli.colors import ENDC, GREEN
from torchx.runner import get_runner, Runner
from torchx.schedulers.api import Stream
from torchx.specs.api import is_started
from torchx.specs.builders import make_app_handle
from torchx.util.log_tee_helpers import (
    _find_role_replicas as find_role_replicas,
    _prefix_line,
)
from torchx.util.types import none_throws

logger: logging.Logger = logging.getLogger(__name__)

ID_FORMAT = "SCHEDULER://[SESSION_NAME]/APP_ID/[ROLE_NAME/[REPLICA_IDS,...]]"


def validate(job_identifier: str) -> None:
    if not re.match(r"^\w+://[^/]*/[^/]+(/[^/]+(/(\d+,?)+)?)?$", job_identifier):
        logger.error(
            f"{job_identifier} is not of the form {ID_FORMAT}",
        )
        sys.exit(1)


def print_log_lines(
    file: TextIO,
    runner: Runner,
    app_handle: str,
    role_name: str,
    replica_id: int,
    regex: str,
    should_tail: bool,
    exceptions: "Queue[Exception]",
    streams: Stream | None,
) -> None:
    try:
        for line in runner.log_lines(
            app_handle,
            role_name,
            replica_id,
            regex,
            should_tail=should_tail,
            streams=streams,
        ):
            prefix = f"{GREEN}{role_name}/{replica_id}{ENDC} "
            print(_prefix_line(prefix, line), file=file, end="", flush=True)
    except Exception as e:
        exceptions.put(e)
        raise


def get_logs(
    file: TextIO,
    identifier: str,
    regex: str | None,
    should_tail: bool = False,
    runner: Runner | None = None,
    streams: Stream | None = None,
) -> None:
    validate(identifier)
    scheduler_backend, _, path_str = identifier.partition("://")

    # path is of the form ["", "app_id", "master", "0"]
    path = path_str.split("/")
    session_name = path[0] or "default"
    app_id = path[1]
    role_name = path[2] if len(path) > 2 else None

    if not runner:
        runner = get_runner()
    app_handle = make_app_handle(scheduler_backend, session_name, app_id)

    if len(path) == 4:
        replica_ids = [(role_name, int(id)) for id in path[3].split(",") if id]
    else:
        display_waiting = True
        while True:
            status = runner.status(app_handle)
            if status and is_started(status.state):
                break
            elif display_waiting:
                logger.info("Waiting for app state response before fetching logs...")
                display_waiting = False
            time.sleep(1)

        app = none_throws(runner.describe(app_handle))
        # print all replicas for the role
        replica_ids = find_role_replicas(app, role_name)

        if not replica_ids:
            valid_ids = "\n".join(
                [
                    f"  {idx}: {scheduler_backend}://{app_id}/{role.name}"
                    for idx, role in enumerate(app.roles)
                ]
            )

            logger.error(
                f"No role [{role_name}] found for app: {app.name}."
                f" Did you mean one of the following:\n{valid_ids}",
            )
            sys.exit(1)

    threads = []
    exceptions = Queue()
    for role_name, replica_id in replica_ids:
        thread = threading.Thread(
            target=print_log_lines,
            args=(
                file,
                runner,
                app_handle,
                role_name,
                replica_id,
                regex,
                should_tail,
                exceptions,
                streams,
            ),
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # Retrieve all exceptions, print all except one and raise the first recorded exception
    threads_exceptions = []
    while not exceptions.empty():
        threads_exceptions.append(exceptions.get())

    if len(threads_exceptions) > 0:
        for i in range(1, len(threads_exceptions)):
            logger.error(threads_exceptions[i])

        raise threads_exceptions[0]


class CmdLog(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--regex",
            type=str,
            help="regex filter",
        )
        subparser.add_argument(
            "-t",
            "--tail",
            action="store_true",
            help="Tail logs",
        )
        subparser.add_argument(
            "--streams",
            type=Stream,
            choices=list(s.value for s in Stream),
            default=None,
            help="IO streams to use. Default is scheduler specific.",
        )
        subparser.add_argument(
            "identifier",
            type=str,
            metavar=ID_FORMAT,
            help="identifiers for the roles and replicas to log",
        )

    def run(self, args: argparse.Namespace) -> None:
        get_logs(
            sys.stdout, args.identifier, args.regex, args.tail, streams=args.streams
        )
