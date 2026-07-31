# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import shlex
from typing import Iterable


def join(args: Iterable[str]) -> str:
    """
    This is equivalent to Python 3.8+'s shlex.join method but also works on Python 3.7.
    """
    return " ".join(shlex.quote(arg) for arg in args)
