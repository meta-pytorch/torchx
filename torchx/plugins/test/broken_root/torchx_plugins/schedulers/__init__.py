# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Broken root-namespace fixture: importing ``torchx_plugins.schedulers``
itself fails with a ``ModuleNotFoundError`` for a missing dependency.

Simulates a plugin distribution that (incorrectly) ships an ``__init__.py``
whose body raises — distinct from the "namespace not installed" case.
"""

import missing_scheduler_dep  # noqa: F401
