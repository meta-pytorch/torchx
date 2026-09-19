#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

from torchx.plugins import PluginType, registry

_log_handlers: dict[str, logging.Handler] = {
    "console": logging.StreamHandler(),
    "null": logging.NullHandler(),
}


def get_logging_handler(destination: str = "null") -> logging.Handler:
    """Return the :py:class:`logging.Handler` that records events for ``destination``.

    A ``torchx.event_handlers`` plugin registered under the same name wins over
    the built-in ``console`` and ``null`` handlers, so a deployment can route
    TorchX events to its own telemetry sink without patching this module.

    Raises:
        KeyError: if no plugin and no built-in handler carries that name.
    """
    factory = registry().get(PluginType.EVENT_HANDLER).get(destination)
    if factory is None:
        return _log_handlers[destination]
    handler: logging.Handler = factory()
    return handler
