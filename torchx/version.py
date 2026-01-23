# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torchx._version import BASE_VERSION
from torchx.util.entrypoints import load

# Follows PEP-0440 version scheme guidelines
# https://www.python.org/dev/peps/pep-0440/#version-scheme
#
# Examples:
# 0.1.0.devN # Developmental release
# 0.1.0aN  # Alpha release
# 0.1.0bN  # Beta release
# 0.1.0rcN  # Release Candidate
# 0.1.0  # Final release
__version__: str = BASE_VERSION


# Use the github container registry images corresponding to the current package
# version.
def _get_torchx_image(torchx_version: str) -> str:
    return f"ghcr.io/pytorch/torchx:{__version__}"


# Check if there's an environment override on the default image.
TORCHX_IMAGE: str = load(
    "torchx.version", "get_torchx_image", default=_get_torchx_image
)(__version__)
