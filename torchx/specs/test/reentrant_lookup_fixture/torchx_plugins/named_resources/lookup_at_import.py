# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Re-entrancy fixture: a namespace plugin that LOOKS UP a named resource at
module top-level. Discovery is triggered from inside
``_NamedResourcesLibrary._load`` (first named-resource lookup), so this
module's import re-enters ``_load`` mid-scan — pinned behavior: the lookup
raises ``RuntimeError`` and the scanner records this module as a plugin
load error."""

from torchx.plugins import register
from torchx.specs import named_resources
from torchx.specs.api import Resource

# the re-entrant lookup IS the fixture: raises RuntimeError mid-scan
_NULL: Resource = named_resources["NULL"]


@register.named_resource()
def lookup_at_import_gpu() -> Resource:
    return Resource(cpu=1, gpu=1, memMB=1024)
