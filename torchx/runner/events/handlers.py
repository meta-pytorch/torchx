#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging


_log_handlers: dict[str, logging.Handler] = {
    "console": logging.StreamHandler(),
    "null": logging.NullHandler(),
}


def get_logging_handler(destination: str = "null") -> logging.Handler:
    return _log_handlers[destination]
