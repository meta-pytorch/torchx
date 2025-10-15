#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import subprocess
from typing import Optional

from .base_version_gen import BASE_VERSION


def get_mercurial_hash() -> Optional[str]:
    """Get the current mercurial revision hash."""
    try:
        # Try to get the mercurial hash from the current working directory
        result = subprocess.run(
            ["hg", "id", "-i"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(__file__),
        )
        # Remove the '+' suffix that indicates uncommitted changes
        hg_hash = result.stdout.strip().rstrip("+")
        return hg_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If hg command fails or is not available, return None
        return None


def get_fb_version() -> str:
    """
    Get the torchx version string.

    Returns:
        Version string following semantic versioning specs
    """
    hg_hash = get_mercurial_hash()
    if hg_hash:
        # Follow semantic versioning local version identifier format
        # https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers
        version = f"{BASE_VERSION}+fb.{hg_hash}"
    else:
        # Fallback if we can't get mercurial hash
        version = f"{BASE_VERSION}+fb"

    return version
