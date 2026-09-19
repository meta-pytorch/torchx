#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import unittest
from typing import cast
from unittest.mock import MagicMock, patch

from torchx.plugins import registry
from torchx.runner import events
from torchx.runner.events import (
    SourceType,
    TorchxEvent,
    _get_or_create_logger,
    log_event,
    record,
)
from torchx.runner.events.handlers import get_logging_handler
from torchx.util import entrypoints

SESSION_ID = "123"


class TorchxEventLibTest(unittest.TestCase):
    def assert_event(
        self, actual_event: TorchxEvent, expected_event: TorchxEvent
    ) -> None:
        self.assertEqual(actual_event.session, expected_event.session)
        self.assertEqual(actual_event.scheduler, expected_event.scheduler)
        self.assertEqual(actual_event.api, expected_event.api)
        self.assertEqual(actual_event.app_id, expected_event.app_id)
        self.assertEqual(actual_event.app_image, expected_event.app_image)
        self.assertEqual(actual_event.runcfg, expected_event.runcfg)
        self.assertEqual(actual_event.source, expected_event.source)
        self.assertEqual(actual_event.app_metadata, expected_event.app_metadata)

    @patch("torchx.runner.events.get_logging_handler")
    def test_get_or_create_logger(self, logging_handler_mock: MagicMock) -> None:
        logging_handler_mock.return_value = logging.NullHandler()
        logger = _get_or_create_logger("test_destination")
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)

    @patch("torchx.runner.events.get_logging_handler")
    def test_get_or_create_logger_per_destination(
        self, logging_handler_mock: MagicMock
    ) -> None:
        logging_handler_mock.side_effect = [
            logging.NullHandler(),
            logging.StreamHandler(),
        ]
        null_logger = _get_or_create_logger("test_destination_null")
        stream_logger = _get_or_create_logger("test_destination_stream")
        self.assertIsNot(
            null_logger,
            stream_logger,
            "each destination must get its own logger; the first call must not"
            " pin every later destination to its logger",
        )
        self.assertIsInstance(
            stream_logger.handlers[0],
            logging.StreamHandler,
            "the second destination must carry its own handler",
        )
        self.assertIs(
            stream_logger,
            _get_or_create_logger("test_destination_stream"),
            "loggers must be cached per destination",
        )

    def test_event_created(self) -> None:
        test_metadata = {"test_key": "test_value"}
        event = TorchxEvent(
            session=SESSION_ID,
            scheduler="test_scheduler",
            api="test_api",
            app_image="test_app_image",
            app_metadata=test_metadata,
            workspace="test_workspace",
        )
        self.assertEqual(SESSION_ID, event.session)
        self.assertEqual("test_scheduler", event.scheduler)
        self.assertEqual("test_api", event.api)
        self.assertEqual("test_app_image", event.app_image)
        self.assertEqual(SourceType.UNKNOWN, event.source)
        self.assertEqual("test_workspace", event.workspace)
        self.assertEqual(test_metadata, event.app_metadata)

    def test_event_deser(self) -> None:
        test_metadata = {"test_key": "test_value"}
        event = TorchxEvent(
            session="test_session",
            scheduler="test_scheduler",
            api="test_api",
            app_image="test_app_image",
            app_metadata=test_metadata,
            workspace="test_workspace",
            source=SourceType.EXTERNAL,
        )
        json_event = event.serialize()
        deser_event = TorchxEvent.deserialize(json_event)
        self.assert_event(event, deser_event)

    def test_event_deser_invalid_type(self) -> None:
        with self.assertRaises(
            TypeError, msg="non str/TorchxEvent input must raise TypeError"
        ):
            TorchxEvent.deserialize(cast(str, 123))


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class GetLoggingHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        registry.cache_clear()
        events._events_loggers.clear()
        self.addCleanup(registry.cache_clear)
        self.addCleanup(events._events_loggers.clear)

    def test_builtin_destinations(self) -> None:
        self.assertIsInstance(get_logging_handler("null"), logging.NullHandler)
        self.assertIsInstance(get_logging_handler("console"), logging.StreamHandler)

    def test_unregistered_destination_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_logging_handler("no_such_destination")

    def test_plugin_shadows_builtin_of_the_same_name(self) -> None:
        handler = _RecordingHandler()
        with patch.object(
            entrypoints, "load_group", return_value={"null": lambda: handler}
        ):
            self.assertIs(
                get_logging_handler("null"),
                handler,
                "a `torchx.event_handlers` plugin must win over the built-in"
                " handler registered under the same destination",
            )

    def test_recorded_events_reach_the_registered_plugin(self) -> None:
        handler = _RecordingHandler()
        destination = "plugin_sink"
        events_logger = logging.getLogger(f"torchx-events-{destination}")
        self.addCleanup(lambda: events_logger.removeHandler(handler))
        with patch.object(
            entrypoints, "load_group", return_value={destination: lambda: handler}
        ):
            record(
                TorchxEvent(session=SESSION_ID, scheduler="local", api="test_api"),
                destination=destination,
            )

        self.assertEqual(
            [SESSION_ID],
            [TorchxEvent.deserialize(m).session for m in handler.messages],
            "the event must be emitted to the plugin handler, not a built-in one",
        )

    def test_recorded_events_survive_a_warning_root_logger(self) -> None:
        handler = _RecordingHandler()
        destination = "warning_root_sink"
        events_logger = logging.getLogger(f"torchx-events-{destination}")
        self.addCleanup(lambda: events_logger.removeHandler(handler))
        root_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        self.addCleanup(lambda: logging.getLogger().setLevel(root_level))

        with patch.object(
            entrypoints, "load_group", return_value={destination: lambda: handler}
        ):
            record(
                TorchxEvent(session=SESSION_ID, scheduler="local", api="test_api"),
                destination=destination,
            )

        self.assertEqual(
            [SESSION_ID],
            [TorchxEvent.deserialize(m).session for m in handler.messages],
            "an app that never configures logging leaves the root logger at"
            " WARNING; the events logger must carry its own INFO level so"
            " `record` is not dropped before the handler",
        )


@patch("torchx.runner.events.record")
@patch("torchx.runner.events.get_session_id_or_create_new")
class LogEventTest(unittest.TestCase):
    def assert_torchx_event(self, expected: TorchxEvent, actual: TorchxEvent) -> None:
        self.assertEqual(expected.session, actual.session)
        self.assertEqual(expected.app_id, actual.app_id)
        self.assertEqual(expected.api, actual.api)
        self.assertEqual(expected.app_image, actual.app_image)
        self.assertEqual(expected.source, actual.source)
        self.assertEqual(expected.workspace, actual.workspace)
        self.assertEqual(expected.app_metadata, actual.app_metadata)

    def test_create_context(
        self, get_session_id_or_create_new_mock: MagicMock, record_mock: MagicMock
    ) -> None:
        get_session_id_or_create_new_mock.return_value = SESSION_ID
        test_dict = {"test_key": "test_value"}
        cfg = json.dumps(test_dict)
        context = log_event(
            "test_call",
            "local",
            "test_app_id",
            app_image="test_app_image_id",
            app_metadata=test_dict,
            runcfg=cfg,
            workspace="test_workspace",
        )
        expected_torchx_event = TorchxEvent(
            SESSION_ID,
            "local",
            "test_call",
            "test_app_id",
            app_image="test_app_image_id",
            app_metadata=test_dict,
            runcfg=cfg,
            workspace="test_workspace",
        )

        self.assert_torchx_event(expected_torchx_event, context._torchx_event)

    def test_record_event(
        self, get_session_id_or_create_new_mock: MagicMock, record_mock: MagicMock
    ) -> None:
        get_session_id_or_create_new_mock.return_value = SESSION_ID
        test_dict = {"test_key": "test_value"}
        cfg = json.dumps(test_dict)
        with log_event(
            "test_call",
            "local",
            "test_app_id",
            app_image="test_app_image_id",
            app_metadata=test_dict,
            runcfg=cfg,
            workspace="test_workspace",
        ) as ctx:
            pass

        expected_torchx_event = TorchxEvent(
            SESSION_ID,
            "local",
            "test_call",
            "test_app_id",
            app_image="test_app_image_id",
            app_metadata=test_dict,
            runcfg=cfg,
            workspace="test_workspace",
            cpu_time_usec=ctx._torchx_event.cpu_time_usec,
            wall_time_usec=ctx._torchx_event.wall_time_usec,
        )
        self.assert_torchx_event(expected_torchx_event, ctx._torchx_event)

    def test_record_event_with_exception(
        self, get_session_id_or_create_new_mock: MagicMock, record_mock: MagicMock
    ) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        with self.assertRaises(RuntimeError):
            with log_event("test_call", "local", "test_app_id", cfg) as ctx:
                raise RuntimeError("test error")
        # pyrefly: ignore [not-iterable]
        self.assertTrue("test error" in ctx._torchx_event.raw_exception)
