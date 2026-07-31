# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import argparse


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
