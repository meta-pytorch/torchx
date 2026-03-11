# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""Deliberately mis-registered plugin for testing error surfacing.

This module lives under ``torchx_plugins/schedulers/`` but registers a
*tracker* plugin.  The registry should detect the type mismatch and
record an error instead of silently including it.
"""

from torchx.plugins import register


@register.tracker()
def mlflow() -> str:
    """This should NOT be discovered as a scheduler."""
    return "wrong"
