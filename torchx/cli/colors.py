#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import sys

# only print colors if outputting directly to a terminal
if not sys.stdout.closed and sys.stdout.isatty():
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    ORANGE = "\033[38:2:238:76:44m"
    GRAY = "\033[2m"
    ENDC = "\033[0m"
else:
    GREEN = ""
    ORANGE = ""
    BLUE = ""
    GRAY = ""
    ENDC = ""


def prefix_container_name(container_name: str, role_name: str, replica_id: int) -> str:
    """
    Generate a colored prefix for a container name.
    Returns empty string for default container (role_name-replica_id), colored name for others.
    """
    default_container = f"{role_name}-{replica_id}"
    if container_name == default_container:
        return ""
    return f"{BLUE}{container_name}{ENDC} "
