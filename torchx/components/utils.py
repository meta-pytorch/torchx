# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility components that are `ready-to-use` out of the box. These are
components that simply execute well known binaries (e.g. ``/bin/cp``)
and are meant to be used as tutorial materials or glue operations between
meaningful stages in a workflow.
"""

import shlex

import torchx.specs as specs
from torchx.version import TORCHX_IMAGE


def echo(
    msg: str = "hello world", image: str = "/tmp", num_replicas: int = 1
) -> specs.AppDef:
    """
    Echos a message to stdout (calls /bin/echo)

    Args:
        msg: message to echo
        image: image to use
        num_replicas: number of replicas to run

    """
    return specs.AppDef(
        name="echo",
        roles=[
            specs.Role(
                name="echo",
                image=image,
                entrypoint="/bin/echo",
                args=[msg],
                num_replicas=num_replicas,
            )
        ],
    )


def touch(file: str) -> specs.AppDef:
    """
    Touches a file (calls /bin/touch)

    Args:
        file: file to create

    """
    return specs.AppDef(
        name="touch",
        roles=[
            specs.Role(
                name="touch",
                image="/tmp",
                entrypoint="/bin/touch",
                args=[file],
                num_replicas=1,
            )
        ],
    )


def sh(*args: str, image: str = "/tmp", num_replicas: int = 1) -> specs.AppDef:
    """
    Runs the provided command via sh. Currently sh does not support
    environment variable substitution.

    Args:
        args: bash arguments
        image: image to use
        num_replicas: number of replicas to run

    """

    escaped_args = " ".join(shlex.quote(arg) for arg in args)

    return specs.AppDef(
        name="sh",
        roles=[
            specs.Role(
                name="sh",
                image=image,
                entrypoint="/bin/sh",
                args=["-c", escaped_args],
                num_replicas=num_replicas,
            )
        ],
    )


def copy(src: str, dst: str, image: str = TORCHX_IMAGE) -> specs.AppDef:
    """
    copy copies the file from src to dst. src and dst can be any valid fsspec
    url.

    This does not support recursive copies or directories.

    Args:
        src: the source fsspec file location
        dst: the destination fsspec file location
        image: the image that contains the copy app
    """

    return specs.AppDef(
        name="torchx-utils-copy",
        roles=[
            specs.Role(
                name="torchx-utils-copy",
                image=image,
                entrypoint="python3",
                args=[
                    "torchx/apps/utils/copy_main.py",
                    "--src",
                    src,
                    "--dst",
                    dst,
                ],
            ),
        ],
    )
