# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Centralized environment variable constants for TorchX.

All ``TORCHX_*`` environment variable names used across the TorchX codebase are
defined here as module-level constants.  Individual modules that previously
defined these constants now re-export them from this module for backward
compatibility.

New code should ``from torchx import settings`` and reference constants as
``settings.ENV_TORCHX_*``.
"""

# -- tracker env vars (previously in torchx.tracker.api) --
ENV_TORCHX_TRACKERS: str = "TORCHX_TRACKERS"
ENV_TORCHX_PARENT_RUN_ID: str = "TORCHX_PARENT_RUN_ID"
ENV_TORCHX_JOB_ID: str = "TORCHX_JOB_ID"

# -- config env vars (previously in torchx.runner.config) --
ENV_TORCHXCONFIG: str = "TORCHXCONFIG"

# -- session env vars (previously in torchx.util.session) --
ENV_TORCHX_INTERNAL_SESSION_ID: str = "TORCHX_INTERNAL_SESSION_ID"

# -- image env vars (previously in torchx.schedulers.docker_scheduler) --
ENV_TORCHX_IMAGE: str = "TORCHX_IMAGE"
