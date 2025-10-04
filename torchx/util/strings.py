# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re


def normalize_str(data: str) -> str:
    """
    Invokes ``lower`` on thes string and removes all
    characters that do not satisfy ``[a-z0-9\\-]`` pattern.
    This method is mostly used to make sure kubernetes scheduler gets
    the job name that does not violate its restrictions.
    """
    if data.startswith("-"):
        data = data[1:]
    pattern = r"[a-z0-9\-]"
    return "".join(re.findall(pattern, data.lower()))
