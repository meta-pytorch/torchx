# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from os import path
from pathlib import Path

from torchx.util import entrypoints


COMPONENTS_DIR: Path = Path("torchx/components")


def get_abspath(relpath: str) -> str:
    module = __name__.replace(".", path.sep)  # torchx/cli/cmd_run
    module_path, _ = path.splitext(__file__)  # $root/torchx/cli/cmd_run
    root = module_path.replace(module, "")
    return path.join(root, relpath)


def _get_file_contents(conf_file: str) -> str | None:
    """
    Reads the ``conf_file`` relative to the root of the project.
    Returns ``None`` if ``$root/$conf_file`` does not exist.
    Example: ``get_file("torchx/cli/config/foo.txt")``
    """
    abspath = get_abspath(conf_file)
    if path.exists(abspath):
        with open(abspath, "r") as f:
            return f.read()
    else:
        return None


def read_conf_file(conf_file: str) -> str:
    builtin_conf = entrypoints.load(
        "torchx.file",
        "get_file_contents",
        default=_get_file_contents,
    )(str(conf_file))
    # user provided conf file precedes the builtin config
    # just print a warning but use the user provided one
    if path.exists(conf_file):
        with open(conf_file, "r") as f:
            return f.read()
    elif builtin_conf:  # conf_file does not exist fallback to builtin
        return builtin_conf
    else:  # neither conf_file nor builtin exists, raise error
        raise FileNotFoundError(
            f"{conf_file} does not exist and is not a builtin."
            " For a list of available builtins run `torchx builtins`"
        )
