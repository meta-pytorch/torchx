# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
If you're wrapping the TorchX API with your own CLI, these functions can
help show the logs of the job within your CLI, just like
`torchx log`
"""

import logging
import threading
from queue import Queue
from typing import TextIO, TYPE_CHECKING

from torchx.util.types import none_throws

if TYPE_CHECKING:
    from torchx.runner.api import Runner
    from torchx.schedulers.api import Stream
    from torchx.specs.api import AppDef

logger: logging.Logger = logging.getLogger(__name__)

# A torchX job can have stderr/stdout for many replicas, of many roles
# The scheduler API has functions that allow us to get,
# with unspecified detail, the log lines of a given replica of
# a given role.
#
# So, to neatly tee the results, we:
# 1) Determine every role ID / replica ID pair we want to monitor
# 2) Request the given stderr / stdout / combined streams from them (1 thread each)
# 3) Concatenate each of those streams to a given destination file


def _find_role_replicas(
    app: "AppDef",
    role_name: str | None,
) -> list[tuple[str, int]]:
    """
    Enumerate all (role, replica id) pairs in the given AppDef.
    Replica IDs are 0-indexed, and range up to num_replicas,
    for each role.
    If role_name is provided, filters to only that name.
    """
    role_replicas = []
    for role in app.roles:
        if role_name is None or role_name == role.name:
            for i in range(role.num_replicas):
                role_replicas.append((role.name, i))
    return role_replicas


def _prefix_line(prefix: str, line: str) -> str:
    """
    _prefix_line ensure the prefix is still present even when dealing with return characters
    """
    if "\r" in line:
        line = line.replace("\r", f"\r{prefix}")
    if "\n" in line[:-1]:
        line = line[:-1].replace("\n", f"\n{prefix}") + line[-1:]
    if not line.startswith("\r"):
        line = f"{prefix}{line}"
    return line


def _print_log_lines_for_role_replica(
    dst: TextIO,
    app_handle: str,
    regex: str | None,
    runner: "Runner",
    which_role: str,
    which_replica: int,
    exceptions: "Queue[Exception]",
    should_tail: bool,
    streams: "Stream | None",
    colorize: bool = False,
) -> None:
    """
    Helper function that'll run in parallel - one
    per monitored replica of a given role.

    Based on print_log_lines .. but not designed for TTY
    """
    try:
        for line in runner.log_lines(
            app_handle,
            which_role,
            which_replica,
            regex,
            should_tail=should_tail,
            streams=streams,
        ):
            if colorize:
                color_begin = "\033[32m"
                color_end = "\033[0m"
            else:
                color_begin = ""
                color_end = ""
            prefix = f"{color_begin}{which_role}/{which_replica}{color_end} "
            print(_prefix_line(prefix, line.strip()), file=dst, end="\n", flush=True)
    except Exception as e:
        exceptions.put(e)
        raise


def _start_threads_to_monitor_role_replicas(
    dst: TextIO,
    app_handle: str,
    regex: str | None,
    runner: "Runner",
    which_role: str | None = None,
    should_tail: bool = False,
    streams: "Stream | None" = None,
    colorize: bool = False,
) -> None:
    threads = []

    app = none_throws(runner.describe(app_handle))
    replica_ids = _find_role_replicas(app, role_name=which_role)

    # Holds exceptions raised by all threads, in a thread-safe
    # object
    exceptions = Queue()

    if not replica_ids:
        valid_roles = [role.name for role in app.roles]
        raise ValueError(
            f"{which_role} is not a valid role name. Available: {valid_roles}"
        )

    for role_name, replica_id in replica_ids:
        threads.append(
            threading.Thread(
                target=_print_log_lines_for_role_replica,
                kwargs={
                    "dst": dst,
                    "runner": runner,
                    "app_handle": app_handle,
                    "which_role": role_name,
                    "which_replica": replica_id,
                    "regex": regex,
                    "should_tail": should_tail,
                    "exceptions": exceptions,
                    "streams": streams,
                    "colorize": colorize,
                },
                daemon=True,
            )
        )

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Retrieve all exceptions, print all except one and raise the first recorded exception
    threads_exceptions = []
    while not exceptions.empty():
        threads_exceptions.append(exceptions.get())

    if len(threads_exceptions) > 0:
        for i in range(1, len(threads_exceptions)):
            logger.error(threads_exceptions[i])

        raise threads_exceptions[0]


def tee_logs(
    dst: TextIO,
    app_handle: str,
    regex: str | None,
    runner: "Runner",
    should_tail: bool = False,
    streams: "Stream | None" = None,
    colorize: bool = False,
) -> threading.Thread:
    """
    Makes a thread, which in turn will start 1 thread per replica
    per role, that tees that role-replica's logs to the given
    destination buffer.

    You'll need to start and join with this parent thread.

    dst:  TextIO to tee the logs into
    app_handle: The return value of runner.run() or runner.schedule()
    regex: Regex to filter the logs that are tee-d
    runner: The Runner you used to schedule the job
    should_tail: If true, continue until we run out of logs. Otherwise, just fetch
                 what's available
    streams: Whether to fetch STDERR, STDOUT, or the temporally COMBINED (default) logs
    """
    thread = threading.Thread(
        target=_start_threads_to_monitor_role_replicas,
        kwargs={
            "dst": dst,
            "runner": runner,
            "app_handle": app_handle,
            "regex": None,
            "should_tail": True,
            "colorize": colorize,
        },
        daemon=True,
    )
    return thread
