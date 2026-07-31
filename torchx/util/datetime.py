# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from datetime import datetime, timedelta


def get_past_epoch_time(days_past: int) -> int:
    past_day = datetime.now() - timedelta(days_past)
    return int(past_day.timestamp())
