# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import sys
from argparse import Action, ArgumentParser, Namespace
from typing import Any, Sequence

from torchx.runner import config

logger: logging.Logger = logging.getLogger(__name__)


class torchxconfig(Action):
    """
    Custom argparse action that loads default torchx CLI options
    from .torchxconfig file.

    """

    called_args: set[str] = set()

    # since this action is used for each argparse argument
    # load the config section for the subcmd once
    _subcmd_configs: dict[str, dict[str, str]] = {}

    def __init__(
        self,
        subcmd: str,
        dest: str,
        option_strings: Sequence[str],
        required: bool = False,
        # pyre-ignore[2] declared as Any in superclass Action
        default: Any = None,
        **kwargs: Any,
    ) -> None:
        cfg = self._subcmd_configs.setdefault(
            subcmd,
            config.get_configs(
                prefix="cli",
                name=subcmd,
            ),
        )

        # if found in .torchxconfig make it the default for this argument
        # otherwise use the default defined from add_argument(...)
        default = cfg.get(dest, default)

        # ``required`` means that it NEEDS to be present  in the CLI args
        # if we found it in .torchxconfig then we don't "require" it to be
        # in the CLI args so set it to False
        if default:
            required = False

        super().__init__(
            dest=dest,
            default=default,
            option_strings=option_strings,
            required=required,
            **kwargs,
        )

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: Any,  # pyre-ignore[2] declared as Any in superclass Action
        option_string: str | None = None,
    ) -> None:
        if option_string is not None:
            if option_string in self.called_args:
                logger.error(f"{option_string} is specified more than once")
                sys.exit(1)
            self.called_args.add(option_string)
        setattr(namespace, self.dest, values)


# argparse takes the action as a Type[Action] so we can't have custom constructors
# hence for each subcommand we need to subclass the base _torchxconfig Action
# this is also how store_true and store_false builtin actions are implemented in argparse
class torchxconfig_run(torchxconfig):
    """
    Custom action that gets the default argument from .torchxconfig.
    """

    def __init__(
        self,
        dest: str,
        option_strings: Sequence[str],
        required: bool = False,
        # pyre-ignore[2] declared as Any in superclass Action
        default: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "run",
            dest=dest,
            default=default,
            required=required,
            option_strings=option_strings,
            **kwargs,
        )


class ArgOnceAction(Action):
    """
    Custom argparse action only allows argument to be specified once
    """

    called_args: set[str] = set()

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: list[str],
        option_string: str | None = None,
    ) -> None:
        if option_string is not None:
            if option_string in self.called_args:
                logger.error(f"{option_string} is specified more than once")
                sys.exit(1)
            self.called_args.add(option_string)
        setattr(namespace, self.dest, values)
