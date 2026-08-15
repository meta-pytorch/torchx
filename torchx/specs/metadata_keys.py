# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Well-known TorchX metadata key and env-var spellings.

These literals are a **wire contract**: schedulers stamp them into submitted
jobs' metadata and downstream telemetry joins on the exact spellings. Import
them from here instead of re-spelling the strings at each writer site.
"""

#: Env var identifying the calling context (e.g. ``cli_run``, a pipeline
#: name). Stamped into job metadata as :py:attr:`app_metadata.CONTEXT`.
TORCHX_CONTEXT_NAME: str = "TORCHX_CONTEXT_NAME"

#: Sentinel written when a metadata value is not available (e.g.
#: :py:data:`TORCHX_CONTEXT_NAME` unset at submit time).
NA: str = "<<NA>>"


class app_metadata:
    """Keys stamped into a submitted job's app-level metadata by schedulers."""

    CONTEXT: str = "torchx/context"
    VERSION: str = "torchx/version"
    SCHEDULER: str = "torchx/scheduler"

    #: Prefix for per-role metadata keys: ``torchx/roles.<role_name>...``.
    #: The schedulers that stamp these keys validate role (task-group) names
    #: against a charset that forbids ``.`` and ``/``, so the first ``.``
    #: after the prefix always terminates the role name, keeping these keys
    #: unambiguous to parse.
    ROLES_PREFIX: str = "torchx/roles."
