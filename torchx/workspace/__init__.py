# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Status: Beta

Workspaces are used to apply local changes on top of existing images so you can
execute your code on a remote cluster. This module contains the interfaces used
by workspace implementations.

These workspaces are defined as an ``fsspec`` path which the directories and
files under will be used to generate a patch.

Example workspace paths:

    * ``file://.`` the current working directory
    * ``memory://foo-bar/`` an in-memory workspace for notebook/programmatic usage
"""

from torchx.workspace.api import walk_workspace, WorkspaceMixin  # noqa: F401
