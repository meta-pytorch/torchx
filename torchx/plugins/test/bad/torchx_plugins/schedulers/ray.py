# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Ray scheduler plugin that fails because ``ray`` is not installed.

Simulates a plugin with a missing dependency.  The registry should
capture this as an import-level error and continue scanning other
modules.
"""

import ray  # noqa: F401  # type: ignore[import-not-found]
