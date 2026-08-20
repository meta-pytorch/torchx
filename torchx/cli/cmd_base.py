# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import argparse
import logging
import sys
from typing import NoReturn

from torchx.runner import get_runner, Runner
from torchx.specs.api import parse_app_handle

logger: logging.Logger = logging.getLogger(__name__)


class SubCommand(abc.ABC):
    """
    Base sub command class, all subcommands should implement this base class
    """

    @abc.abstractmethod
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        """
        Adds the arguments to this sub command
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, args: argparse.Namespace) -> None:
        """
        Runs the sub command. Parsed arguments are available as ``args``.
        """
        raise NotImplementedError()


class AppHandleSubCommand(SubCommand):
    """Base for subcommands that act on a single ``app_handle`` positional arg
    (e.g. ``cancel``, ``delete``, ``describe``, ``status``).

    Adds the ``app_handle`` argument and runs :py:meth:`run_with_runner`
    inside a ``with get_runner()`` block so scheduler clients are closed.
    """

    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "app_handle",
            type=str,
            help="torchx app handle (e.g. local://session-name/app-id)",
        )

    def run(self, args: argparse.Namespace) -> None:
        with get_runner() as runner:
            self.run_with_runner(args, runner)

    @abc.abstractmethod
    def run_with_runner(self, args: argparse.Namespace, runner: Runner) -> None:
        """Command body; ``runner`` is closed when this returns."""
        raise NotImplementedError()

    def exit_missing_app(self, app_handle: str) -> NoReturn:
        """Logs the shared app-not-found error and exits non-zero."""
        scheduler, _, app_id = parse_app_handle(app_handle)
        logger.error(
            "AppDef `%s` does not exist or has been removed from `%s`'s data plane",
            app_id,
            scheduler,
        )
        sys.exit(1)
