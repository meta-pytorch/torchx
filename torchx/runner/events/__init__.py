#!/usr/bin/env/python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Module contains events processing mechanisms that are integrated with the standard python logging.

Example of usage:

::

  from torchx import events
  event = TorchxEvent(..)
  events.record(event)

"""

import json
import logging
import sys
import time
import traceback
from types import TracebackType
from typing import Dict, Optional, Type

from torchx.runner.events.handlers import get_logging_handler
from torchx.util.session import get_session_id_or_create_new

from .api import SourceType, TorchxEvent  # noqa F401

_events_logger: Optional[logging.Logger] = None

log: logging.Logger = logging.getLogger(__name__)


def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """
    Constructs python logger based on the destination type or extends if provided.
    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
        logger: Logger to be extended with the events handler. Method constructs
            a new logger if None provided.
    """
    global _events_logger

    if _events_logger:
        return _events_logger
    else:
        logging_handler = get_logging_handler(destination)
        logging_handler.setLevel(logging.DEBUG)
        _events_logger = logging.getLogger(f"torchx-events-{destination}")
        # Do not propagate message to the root logger
        _events_logger.propagate = False
        _events_logger.addHandler(logging_handler)

        assert _events_logger  # make type-checker happy
        return _events_logger


def record(event: TorchxEvent, destination: str = "null") -> None:
    try:
        serialized_event = event.serialize()
    except Exception:
        log.exception("failed to serialize event, will not record event")
    else:
        _get_or_create_logger(destination).info(serialized_event)


class log_event:
    """
    Context for logging torchx events. Creates TorchxEvent and records it in
    the default destination at the end of the context execution. If exception occurs
    the event will be recorded as well with the error message.

    Example of usage:

    ::

    with log_event("api_name", ..):
        ...

    """

    def __init__(
        self,
        api: str,
        scheduler: Optional[str] = None,
        app_id: Optional[str] = None,
        app_image: Optional[str] = None,
        app_metadata: Optional[Dict[str, str]] = None,
        runcfg: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> None:
        self._torchx_event: TorchxEvent = self._generate_torchx_event(
            api,
            scheduler or "",
            app_id,
            app_image=app_image,
            app_metadata=app_metadata,
            runcfg=runcfg,
            workspace=workspace,
        )
        self._start_cpu_time_ns = 0
        self._start_wall_time_ns = 0
        self._start_epoch_time_usec = 0

    def __enter__(self) -> "log_event":
        self._start_cpu_time_ns = time.process_time_ns()
        self._start_wall_time_ns = time.perf_counter_ns()
        self._torchx_event.start_epoch_time_usec = int(time.time() * 1_000_000)

        return self

    def __exit__(
        self,
        exec_type: Optional[Type[BaseException]],
        exec_value: Optional[BaseException],
        traceback_type: Optional[TracebackType],
    ) -> Optional[bool]:
        self._torchx_event.cpu_time_usec = (
            time.process_time_ns() - self._start_cpu_time_ns
        ) // 1000
        self._torchx_event.wall_time_usec = (
            time.perf_counter_ns() - self._start_wall_time_ns
        ) // 1000
        if traceback_type:
            self._torchx_event.raw_exception = traceback.format_exc()
            typ, value, tb = sys.exc_info()
            if tb:
                last_frame = traceback.extract_tb(tb)[-1]
                self._torchx_event.exception_source_location = json.dumps(
                    {
                        "filename": last_frame.filename,
                        "lineno": last_frame.lineno,
                        "name": last_frame.name,
                    }
                )
        if exec_type:
            self._torchx_event.exception_type = exec_type.__name__
        if exec_value:
            self._torchx_event.exception_message = str(exec_value)
        record(self._torchx_event)

    def _generate_torchx_event(
        self,
        api: str,
        scheduler: str,
        app_id: Optional[str] = None,
        app_image: Optional[str] = None,
        app_metadata: Optional[Dict[str, str]] = None,
        runcfg: Optional[str] = None,
        source: SourceType = SourceType.UNKNOWN,
        workspace: Optional[str] = None,
    ) -> TorchxEvent:
        return TorchxEvent(
            session=get_session_id_or_create_new(),
            scheduler=scheduler,
            api=api,
            app_id=app_id,
            app_image=app_image,
            app_metadata=app_metadata,
            runcfg=runcfg,
            source=source,
            workspace=workspace,
        )
