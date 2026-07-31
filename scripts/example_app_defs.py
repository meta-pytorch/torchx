#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
App Defs for integration tests.
"""

from torchx.components.dist import ddp as dist_ddp
from torchx.components.integration_tests.component_provider import ComponentProvider
from torchx.components.utils import python as utils_python
from torchx.specs import AppDef


class CvTrainerComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return dist_ddp(
            *("--output_path", "/tmp", "--skip_export", "--log_path", "/tmp"),
            image=self._image,
            j="1x1",
            m="torchx.examples.apps.lightning.train",
            memMB=2048,
        )


class DatapreprocComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return utils_python(
            *("--output_path", "/tmp/test", "--limit", "100"),
            image=self._image,
            m="torchx.examples.apps.datapreproc.datapreproc",
        )
