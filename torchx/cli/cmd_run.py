# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging
import os
import sys
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, MISSING as DATACLASS_MISSING
from itertools import groupby
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import torchx.specs as specs
from torchx.cli.argparse_util import ArgOnceAction, torchxconfig_run
from torchx.cli.cmd_base import SubCommand
from torchx.cli.cmd_log import get_logs
from torchx.runner import config, get_runner, Runner
from torchx.runner.config import load_sections
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.specs import CfgVal, Workspace
from torchx.specs.finder import (
    _Component,
    ComponentNotFoundException,
    ComponentValidationException,
    get_builtin_source,
    get_components,
)
from torchx.util.log_tee_helpers import tee_logs
from torchx.util.types import none_throws


MISSING_COMPONENT_ERROR_MSG = (
    "missing component name, either provide it from the CLI or in .torchxconfig"
)

LOCAL_SCHEDULER_WARNING_MSG = (
    "`local` scheduler is deprecated and will be"
    " removed in the near future,"
    " please use other variants of the local scheduler"
    " (e.g. `local_cwd`)"
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TorchXRunArgs:
    component_name: str
    scheduler: str
    scheduler_args: Dict[str, Any]
    scheduler_cfg: Dict[str, CfgVal] = field(default_factory=dict)
    dryrun: bool = False
    wait: bool = False
    log: bool = False
    workspace: str = ""
    parent_run_id: Optional[str] = None
    tee_logs: bool = False
    component_args: Dict[str, Any] = field(default_factory=dict)
    component_args_str: List[str] = field(default_factory=list)


def torchx_run_args_from_json(json_data: Dict[str, Any]) -> TorchXRunArgs:
    all_fields = [f.name for f in fields(TorchXRunArgs)]
    required_fields = {
        f.name
        for f in fields(TorchXRunArgs)
        if f.default is DATACLASS_MISSING and f.default_factory is DATACLASS_MISSING
    }
    missing_fields = required_fields - json_data.keys()
    if missing_fields:
        raise ValueError(
            f"The following required fields are missing: {', '.join(missing_fields)}"
        )

    # Fail if there are fields that aren't part of the run command
    filtered_json_data = {k: v for k, v in json_data.items() if k in all_fields}
    extra_fields = set(json_data.keys()) - set(all_fields)
    if extra_fields:
        raise ValueError(
            f"The following fields are not part of the run command: {', '.join(extra_fields)}.",
            "Please check your JSON and try launching again.",
        )

    torchx_args = TorchXRunArgs(**filtered_json_data)
    if torchx_args.workspace == "":
        torchx_args.workspace = f"{Path.cwd()}"
    return torchx_args


def torchx_run_args_from_argparse(
    args: argparse.Namespace,
    component_name: str,
    component_args: List[str],
    scheduler_cfg: Dict[str, CfgVal],
) -> TorchXRunArgs:
    return TorchXRunArgs(
        component_name=component_name,
        scheduler=args.scheduler,
        scheduler_args={},
        scheduler_cfg=scheduler_cfg,
        dryrun=args.dryrun,
        wait=args.wait,
        log=args.log,
        workspace=args.workspace,
        parent_run_id=args.parent_run_id,
        tee_logs=args.tee_logs,
        component_args_str=component_args,
    )


def _parse_component_name_and_args(
    component_name_and_args: List[str],
    subparser: argparse.ArgumentParser,
    dirs: Optional[List[str]] = None,  # for testing only
) -> Tuple[str, List[str]]:
    """
    Given a list of nargs parsed from commandline, parses out the component name
    and component args. If component name is not found in the list, then
    the default component is loaded from the [cli:run] component section in
    .torchxconfig. If no default config is specified in .torchxconfig, then
    this method errors out to the specified subparser.

    This method deals with the following input list:

    1. [$component_name, *$component_args]
        - Example: ["utils.echo", "--msg", "hello"] or ["utils.echo"]
        - Note: component name and args both in list
    2. [*$component_args]
        - Example: ["--msg", "hello"] or []
        - Note: component name loaded from .torchxconfig, args in list
        - Note: assumes list is only args if the first element
                looks like an option (e.g. starts with "-")

    """
    component = config.get_config(prefix="cli", name="run", key="component", dirs=dirs)
    component_args = []

    # make a copy of the input list to guard against side-effects
    args = list(component_name_and_args)

    if len(args) > 0:
        # `--` is used to delimit between run's options and nargs which includes component args
        # argparse returns the delimiter as part of the nargs so just ignore it if present
        if args[0] == "--":
            args = args[1:]

    if len(args) > 0:  # check len again in case we removed the leading "--" above
        if args[0].startswith("-"):
            component_args = args
        else:  # first element is NOT an option; then it must be a component name
            component = args[0]
            component_args = args[1:]

    # Error if there are repeated command line arguments each group of arguments,
    # where the groups are separated by "--"
    arg_groups = [list(g) for _, g in groupby(component_args, key=lambda x: x == "--")]
    for arg_group in arg_groups:
        all_options = [
            x
            for x in arg_group
            if x.startswith("-") and x.strip() != "-" and x.strip() != "--"
        ]
        arg_count = Counter(all_options)
        duplicates = [arg for arg, count in arg_count.items() if count > 1]
        if len(duplicates) > 0:
            subparser.error(f"Repeated Command Line Arguments: {duplicates}")

    if not component:
        subparser.error(MISSING_COMPONENT_ERROR_MSG)

    return component, component_args


class CmdBuiltins(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--print",
            type=str,
            help="prints the builtin's component def to stdout",
        )

    def _builtins(self) -> Dict[str, _Component]:
        return get_components()

    def run(self, args: argparse.Namespace) -> None:
        builtin_name = args.print
        if not builtin_name:
            builtin_components = self._builtins()
            num_builtins = len(builtin_components)
            print(f"Found {num_builtins} builtin components:")
            for i, component in enumerate(builtin_components.values()):
                print(f" {i + 1:2d}. {component.name}")
        else:
            print(get_builtin_source(builtin_name))


class CmdRun(SubCommand):
    def __init__(self) -> None:
        self._subparser: Optional[argparse.ArgumentParser] = None
        self._stdin_data_json: Optional[Dict[str, Any]] = None

    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        scheduler_names = get_scheduler_factories().keys()
        self._subparser = subparser
        subparser.add_argument(
            "-s",
            "--scheduler",
            type=str,
            default=get_default_scheduler_name(),
            choices=list(scheduler_names),
            action=torchxconfig_run,
            help="Name of the scheduler to use.",
        )
        subparser.add_argument(
            "-cfg",
            "--scheduler_args",
            type=str,
            action=ArgOnceAction,
            help="Arguments to pass to the scheduler (Ex:`cluster=foo,user=bar`)."
            " For a list of scheduler run options run: `torchx runopts`",
        )
        subparser.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Does not actually submit the app,"
            " just prints the scheduler request",
        )
        subparser.add_argument(
            "--wait",
            action="store_true",
            default=False,
            help="Wait for the app to finish before exiting.",
        )
        subparser.add_argument(
            "--log",
            action="store_true",
            default=False,
            help="Stream logs while waiting for app to finish.",
        )
        subparser.add_argument(
            "--workspace",
            "--buck-target",
            default=f"{Path.cwd()}",
            action=torchxconfig_run,
            help="local workspace to build/patch (buck-target of main binary if using buck)",
        )
        subparser.add_argument(
            "--parent_run_id",
            type=str,
            action=ArgOnceAction,
            help="optional parent run ID that this run belongs to."
            " It can be used to group runs for experiment tracking purposes",
        )
        subparser.add_argument(
            "--tee_logs",
            action="store_true",
            default=False,
            help="Add additional prefix to log lines to indicate which replica is printing the log",
        )
        subparser.add_argument(
            "--stdin",
            action="store_true",
            default=False,
            help="Read JSON input from stdin to parse into torchx run args and run the component.",
        )
        subparser.add_argument(
            "component_name_and_args",
            nargs=argparse.REMAINDER,
        )

    def _run_inner(self, runner: Runner, args: TorchXRunArgs) -> None:
        if args.scheduler == "local":
            logger.warning(LOCAL_SCHEDULER_WARNING_MSG)

        config.apply(scheduler=args.scheduler, cfg=args.scheduler_cfg)
        component_args = (
            args.component_args_str
            if args.component_args_str != []
            else args.component_args
        )
        try:
            workspace = Workspace.from_str(args.workspace) if args.workspace else None

            if args.dryrun:
                dryrun_info = runner.dryrun_component(
                    args.component_name,
                    component_args,
                    args.scheduler,
                    workspace=workspace,
                    cfg=args.scheduler_cfg,
                    parent_run_id=args.parent_run_id,
                )
                print(
                    "\n=== APPLICATION ===\n"
                    f"{pformat(asdict(dryrun_info._app), indent=2, width=80)}"
                )

                print("\n=== SCHEDULER REQUEST ===\n" f"{dryrun_info}")
            else:
                app_handle = runner.run_component(
                    args.component_name,
                    component_args,
                    args.scheduler,
                    workspace=args.workspace,
                    cfg=args.scheduler_cfg,
                    parent_run_id=args.parent_run_id,
                )
                # DO NOT delete this line. It is used by slurm tests to retrieve the app id
                print(app_handle)

                if args.scheduler.startswith("local"):
                    self._wait_and_exit(
                        runner, app_handle, log=True, tee_logs=args.tee_logs
                    )
                else:
                    logger.info(f"Launched app: {app_handle}")
                    app_status = runner.status(app_handle)
                    if app_status:
                        logger.info(app_status.format())
                    if args.wait or args.log:
                        self._wait_and_exit(
                            runner, app_handle, log=args.log, tee_logs=args.tee_logs
                        )

        except (ComponentValidationException, ComponentNotFoundException) as e:
            error_msg = (
                f"\nFailed to run component `{args.component_name}` got errors: \n {e}"
            )
            logger.error(error_msg)
            sys.exit(1)
        except specs.InvalidRunConfigException as e:
            error_msg = (
                "Invalid scheduler configuration: %s\n"
                "To configure scheduler options, either:\n"
                "  1. Use the `-cfg` command-line argument, e.g., `-cfg key1=value1,key2=value2`\n"
                "  2. Set up a `.torchxconfig` file. For more details, visit: https://meta-pytorch.org/torchx/main/runner.config.html\n"
                "Run `torchx runopts %s` to check all available configuration options for the "
                "`%s` scheduler."
            )
            print(error_msg % (e, args.scheduler, args.scheduler), file=sys.stderr)
            sys.exit(1)

    def _run_from_cli_args(self, runner: Runner, args: argparse.Namespace) -> None:
        scheduler_opts = runner.scheduler_run_opts(args.scheduler)
        cfg = scheduler_opts.cfg_from_str(args.scheduler_args)

        component, component_args = _parse_component_name_and_args(
            args.component_name_and_args,
            none_throws(self._subparser),
        )
        torchx_run_args = torchx_run_args_from_argparse(
            args, component, component_args, cfg
        )
        self._run_inner(runner, torchx_run_args)

    def _run_from_stdin_args(self, runner: Runner, stdin_data: Dict[str, Any]) -> None:
        torchx_run_args = torchx_run_args_from_json(stdin_data)
        scheduler_opts = runner.scheduler_run_opts(torchx_run_args.scheduler)
        cfg = scheduler_opts.cfg_from_json_repr(
            json.dumps(torchx_run_args.scheduler_args)
        )
        torchx_run_args.scheduler_cfg = cfg
        self._run_inner(runner, torchx_run_args)

    def _get_torchx_stdin_args(
        self, args: argparse.Namespace
    ) -> Optional[Dict[str, Any]]:
        if not args.stdin:
            return None
        if self._stdin_data_json is None:
            self._stdin_data_json = self.torchx_json_from_stdin(args)
        return self._stdin_data_json

    def torchx_json_from_stdin(
        self, args: Optional[argparse.Namespace] = None
    ) -> Dict[str, Any]:
        try:
            stdin_data_json = json.load(sys.stdin)
            if args and args.dryrun:
                stdin_data_json["dryrun"] = True
            if not isinstance(stdin_data_json, dict):
                logger.error(
                    "Invalid JSON input for `torchx run` command. Expected a dictionary."
                )
                sys.exit(1)
            return stdin_data_json
        except (json.JSONDecodeError, EOFError):
            logger.error(
                "Unable to parse JSON input for `torchx run` command, please make sure it's a valid JSON input."
            )
            sys.exit(1)

    def verify_no_extra_args(self, args: argparse.Namespace) -> None:
        """
        Verifies that only --stdin was provided when using stdin mode.
        """
        if not args.stdin:
            return

        subparser = none_throws(self._subparser)
        conflicting_args = []

        # Check each argument against its default value
        for action in subparser._actions:
            if action.dest == "stdin":  # Skip stdin itself
                continue
            if action.dest == "help":  # Skip help
                continue
            if action.dest == "dryrun":  # Skip dryrun
                continue

            current_value = getattr(args, action.dest, None)
            default_value = action.default

            # For arguments that differ from default
            if current_value != default_value:
                # Handle special cases where non-default doesn't mean explicitly set
                if action.dest == "component_name_and_args" and current_value == []:
                    continue  # Empty list is still default
                print(f"*********\n {default_value} = {current_value}")
                conflicting_args.append(f"--{action.dest.replace('_', '-')}")

        if conflicting_args:
            subparser.error(
                f"Cannot specify {', '.join(conflicting_args)} when using --stdin. "
                "All configuration should be provided in JSON input."
            )

    def _run(self, runner: Runner, args: argparse.Namespace) -> None:
        self.verify_no_extra_args(args)
        if args.stdin:
            stdin_data_json = self._get_torchx_stdin_args(args)
            if stdin_data_json is not None:
                self._run_from_stdin_args(runner, stdin_data_json)
        else:
            self._run_from_cli_args(runner, args)

    def run(self, args: argparse.Namespace) -> None:
        os.environ["TORCHX_CONTEXT_NAME"] = os.getenv("TORCHX_CONTEXT_NAME", "cli_run")
        component_defaults = load_sections(prefix="component")

        with get_runner(component_defaults=component_defaults) as runner:
            self._run(runner, args)

    def _wait_and_exit(
        self, runner: Runner, app_handle: str, log: bool, tee_logs: bool = False
    ) -> None:
        logger.info("Waiting for the app to finish...")

        log_thread = (
            self._start_log_thread(runner, app_handle, tee_logs_enabled=tee_logs)
            if log
            else None
        )

        status = runner.wait(app_handle, wait_interval=1)
        if not status:
            raise RuntimeError(f"unknown status, wait returned {status}")

        logger.info(f"Job finished: {status.state}")

        if log_thread:
            log_thread.join()

        if status.state != specs.AppState.SUCCEEDED:
            logger.error(status)
            sys.exit(1)
        else:
            logger.debug(status)

    def _start_log_thread(
        self, runner: Runner, app_handle: str, tee_logs_enabled: bool = False
    ) -> threading.Thread:
        if tee_logs_enabled:
            thread = tee_logs(
                dst=sys.stderr,
                app_handle=app_handle,
                regex=None,
                runner=runner,
                should_tail=True,
                streams=None,
                colorize=not sys.stderr.closed and sys.stderr.isatty(),
            )
        else:
            thread = threading.Thread(
                target=get_logs,
                kwargs={
                    "file": sys.stderr,
                    "runner": runner,
                    "identifier": app_handle,
                    "regex": None,
                    "should_tail": True,
                },
            )
            thread.daemon = True
        thread.start()
        return thread
