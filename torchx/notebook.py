#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This contains TorchX utilities for creating and running components and apps from
an Jupyter/IPython Notebook.
"""

import posixpath

import fsspec
from IPython.core.magic import register_cell_magic


def get_workspace() -> str:
    """
    get_workspace returns the TorchX notebook workspace fsspec path.
    """
    return "memory://torchx-workspace/"


@register_cell_magic
def workspacefile(line: str, cell: str) -> None:
    workspace = get_workspace()
    fs, path = fsspec.core.url_to_fs(workspace)
    path = posixpath.join(path, line)

    base = posixpath.dirname(path)
    if not fs.exists(base):
        fs.mkdirs(base, exist_ok=True)

    with fs.open(path, "wt") as f:
        f.write(cell)

    print(f"Added {line} to workspace {workspace}")
