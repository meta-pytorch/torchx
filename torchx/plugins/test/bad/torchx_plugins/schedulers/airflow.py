# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Airflow scheduler plugin that fails at import time.

Simulates a plugin whose module-level initialization raises an
unexpected error (e.g. trying to connect to the Airflow REST API
before the scheduler factory is even called).  The registry should
capture this as an import-level error and continue scanning other
modules.
"""

raise RuntimeError("cannot reach Airflow REST API at http://localhost:8080")
