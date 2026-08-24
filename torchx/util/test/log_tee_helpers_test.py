# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import threading
import unittest
from collections.abc import Iterator
from queue import Queue
from typing import NoReturn
from unittest.mock import MagicMock, patch

from torchx.schedulers.api import Stream
from torchx.specs.api import AppDef, Role
from torchx.util.log_tee_helpers import (
    _print_log_lines_for_role_replica,
    _start_threads_to_monitor_role_replicas,
    tee_logs,
)


class PrintLogLinesForRoleReplicaTest(unittest.TestCase):
    def _print(
        self,
        lines: list[str],
        colorize: bool,
        exceptions: "Queue[Exception]",
    ) -> tuple[io.StringIO, MagicMock]:
        dst = io.StringIO()
        runner = MagicMock()
        runner.log_lines.return_value = iter(lines)
        _print_log_lines_for_role_replica(
            dst=dst,
            app_handle="local://test_session/test_app",
            regex=None,
            runner=runner,
            which_role="worker",
            which_replica=1,
            exceptions=exceptions,
            should_tail=False,
            streams=None,
            colorize=colorize,
        )
        return dst, runner

    def test_prefixes_every_line_with_role_and_replica(self) -> None:
        exceptions: "Queue[Exception]" = Queue()
        dst, _ = self._print(
            ["alpha\n", "beta\n"], colorize=False, exceptions=exceptions
        )
        self.assertEqual(
            "worker/1 alpha\nworker/1 beta\n",
            dst.getvalue(),
            msg="every log line must carry a plain `role/replica ` prefix when colorize is off",
        )

    def test_colorize_wraps_prefix_in_ansi_green(self) -> None:
        exceptions: "Queue[Exception]" = Queue()
        dst, _ = self._print(["alpha\n"], colorize=True, exceptions=exceptions)
        self.assertEqual(
            "\033[32mworker/1\033[0m alpha\n",
            dst.getvalue(),
            msg="colorize=True must wrap only the `role/replica` prefix in green ANSI codes",
        )

    def test_empty_log_stream_writes_nothing(self) -> None:
        exceptions: "Queue[Exception]" = Queue()
        dst, _ = self._print([], colorize=False, exceptions=exceptions)
        self.assertEqual(
            "",
            dst.getvalue(),
            msg="a replica with no log lines must produce no output (not even a bare prefix)",
        )

    def test_log_lines_error_is_recorded_and_reraised(self) -> None:
        exceptions: "Queue[Exception]" = Queue()
        dst = io.StringIO()
        runner = MagicMock()
        error = RuntimeError("stream broke")
        runner.log_lines.side_effect = error
        with self.assertRaises(RuntimeError):
            _print_log_lines_for_role_replica(
                dst=dst,
                app_handle="local://test_session/test_app",
                regex=None,
                runner=runner,
                which_role="worker",
                which_replica=0,
                exceptions=exceptions,
                should_tail=False,
                streams=None,
                colorize=False,
            )
        self.assertIs(
            error,
            exceptions.get_nowait(),
            msg="the raised exception must also be queued so the parent thread can see it",
        )


class _InlineThread(threading.Thread):
    def start(self) -> None:
        self.run()

    def join(self, timeout: float | None = None) -> None:
        pass


class StartThreadsToMonitorRoleReplicasTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = AppDef(
            name="test_app",
            roles=[
                Role(name="trainer", image="/tmp", num_replicas=2),
                Role(name="reader", image="/tmp", num_replicas=1),
            ],
        )
        self.runner = MagicMock()
        self.runner.describe.return_value = self.app

    def test_tees_logs_of_every_role_replica(self) -> None:
        def fake_log_lines(
            app_handle: str,
            role_name: str,
            replica_id: int,
            regex: str | None,
            should_tail: bool = False,
            streams: "Stream | None" = None,
        ) -> Iterator[str]:
            return iter([f"hello from {role_name}/{replica_id}\n"])

        self.runner.log_lines.side_effect = fake_log_lines
        dst = io.StringIO()
        with patch("torchx.util.log_tee_helpers.threading.Thread", new=_InlineThread):
            _start_threads_to_monitor_role_replicas(
                dst=dst,
                app_handle="local://test_session/test_app",
                regex="foo.*",
                runner=self.runner,
                should_tail=True,
                streams=Stream.STDERR,
            )
        self.assertEqual(
            [
                "reader/0 hello from reader/0",
                "trainer/0 hello from trainer/0",
                "trainer/1 hello from trainer/1",
            ],
            sorted(dst.getvalue().splitlines()),
            msg="one prefixed line per (role, replica) pair must reach the destination",
        )
        self.runner.log_lines.assert_any_call(
            "local://test_session/test_app",
            "trainer",
            0,
            "foo.*",
            should_tail=True,
            streams=Stream.STDERR,
        )

    def test_unknown_role_raises_with_valid_role_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"nonexistent is not a valid role name.*trainer.*reader",
            msg="filtering to an unknown role must fail fast naming the valid roles",
        ):
            _start_threads_to_monitor_role_replicas(
                dst=io.StringIO(),
                app_handle="local://test_session/test_app",
                regex=None,
                runner=self.runner,
                which_role="nonexistent",
            )
        self.runner.log_lines.assert_not_called()

    def test_single_replica_error_propagates_without_extra_logging(self) -> None:
        self.runner.log_lines.side_effect = RuntimeError("reader stream broke")
        with patch("threading.excepthook"):
            with self.assertRaisesRegex(RuntimeError, "reader stream broke"):
                with self.assertNoLogs("torchx.util.log_tee_helpers", level="ERROR"):
                    _start_threads_to_monitor_role_replicas(
                        dst=io.StringIO(),
                        app_handle="local://test_session/test_app",
                        regex=None,
                        runner=self.runner,
                        which_role="reader",
                    )

    def test_first_replica_error_raised_rest_logged(self) -> None:
        errors: dict[int, RuntimeError] = {
            0: RuntimeError("replica 0 broke"),
            1: RuntimeError("replica 1 broke"),
        }

        def fail_log_lines(
            app_handle: str,
            role_name: str,
            replica_id: int,
            regex: str | None,
            should_tail: bool = False,
            streams: "Stream | None" = None,
        ) -> NoReturn:
            raise errors[replica_id]

        self.runner.log_lines.side_effect = fail_log_lines
        with patch("threading.excepthook"):
            with self.assertLogs("torchx.util.log_tee_helpers", level="ERROR") as logs:
                with self.assertRaises(RuntimeError) as cm:
                    _start_threads_to_monitor_role_replicas(
                        dst=io.StringIO(),
                        app_handle="local://test_session/test_app",
                        regex=None,
                        runner=self.runner,
                        which_role="trainer",
                    )
        raised = str(cm.exception)
        logged = [r.getMessage() for r in logs.records]
        self.assertEqual(
            1,
            len(logged),
            msg="with 2 failing replicas exactly one exception is logged (the other is raised)",
        )
        self.assertEqual(
            {"replica 0 broke", "replica 1 broke"},
            {raised, logged[0]},
            msg="every replica failure must surface: one raised, all others logged",
        )


class TeeLogsTest(unittest.TestCase):
    def test_tee_logs_forwards_params(self) -> None:
        dst = io.StringIO()
        runner = MagicMock()
        with patch(
            "torchx.util.log_tee_helpers._start_threads_to_monitor_role_replicas"
        ) as monitor_mock:
            thread = tee_logs(
                dst=dst,
                app_handle="local://test_session/test_app",
                regex="foo.*",
                runner=runner,
                should_tail=False,
                streams=Stream.STDERR,
                colorize=True,
            )
            thread.start()
            thread.join()

        monitor_mock.assert_called_once_with(
            dst=dst,
            runner=runner,
            app_handle="local://test_session/test_app",
            regex="foo.*",
            should_tail=False,
            streams=Stream.STDERR,
            colorize=True,
        )
