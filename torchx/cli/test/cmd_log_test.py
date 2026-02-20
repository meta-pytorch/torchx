#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import sys
import unittest
from datetime import datetime
from typing import Iterator
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_log import _prefix_line, ENDC, get_logs, GREEN, validate
from torchx.runner.api import Runner
from torchx.schedulers.api import Stream
from torchx.specs import AppDef, AppHandle, AppState, AppStatus, parse_app_handle, Role


class SentinelError(Exception):
    """
    Used to mock sys.exit
    """

    pass


RUNNER = "torchx.cli.cmd_log.get_runner"


class MockRunner(Runner):
    def __init__(self) -> None:
        pass

    def __call__(self, name: str | None = None) -> "MockRunner":
        return self

    def describe(self, app_handle: str) -> AppDef:
        scheduler_backend, session_name, app_id = parse_app_handle(app_handle)
        return AppDef(
            name=app_id,
            roles=[
                Role(name="master", image="test_image", num_replicas=1),
                Role(name="trainer", image="test_image", num_replicas=3),
            ],
        )

    def status(self, app_handle: str) -> AppStatus:
        return AppStatus(state=AppState.RUNNING)

    def log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        should_tail: bool = False,
        streams: Stream | None = None,
    ) -> Iterator[str]:
        import re

        if regex is None:
            regex = ".*"

        log_lines = ["INFO foo\n", "ERROR bar\n", "WARN baz\n"]
        return iter([line for line in log_lines if re.match(regex, line)])


class MockPendingRunner(MockRunner):
    def __init__(self) -> None:
        self._status_checks = 0

    def status(self, app_handle: str) -> AppStatus:
        if self._status_checks < 2:
            self._status_checks += 1
            return AppStatus(state=AppState.PENDING)
        else:
            return super().status(app_handle)


class CmdLogTest(unittest.TestCase):
    @patch("sys.exit", side_effect=SentinelError)
    def test_cmd_log_bad_job_identifier(self, exit_mock: MagicMock) -> None:
        with self.assertRaises(SentinelError):
            get_logs(sys.stdout, "local:///SparseNNAppDef/", "QPS.*")
        exit_mock.assert_called_once_with(1)

    @patch(RUNNER, new_callable=MockRunner)
    @patch("sys.exit", side_effect=SentinelError)
    def test_cmd_log_unknown_role(
        self, exit_mock: MagicMock, mock_runner: MagicMock
    ) -> None:
        with self.assertRaises(SentinelError):
            get_logs(
                sys.stdout,
                "local_docker://default/SparseNNAppDef/unknown_role",
                regex=None,
            )

        exit_mock.assert_called_once_with(1)

    @patch(RUNNER, new_callable=MockRunner)
    def test_cmd_log_all_roles(self, mock_runner: MagicMock) -> None:
        out = io.StringIO()
        get_logs(
            out,
            "local_docker://test-session/SparseNNAppDef",
            regex="INFO",
        )
        self.assertSetEqual(
            {
                f"{GREEN}master/0{ENDC} INFO foo",
                f"{GREEN}trainer/0{ENDC} INFO foo",
                f"{GREEN}trainer/1{ENDC} INFO foo",
                f"{GREEN}trainer/2{ENDC} INFO foo",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(out.getvalue().split("\n")),
        )

    @patch(RUNNER, new_callable=MockRunner)
    def test_cmd_log_all_replicas(self, mock_runner: MagicMock) -> None:
        out = io.StringIO()
        get_logs(
            out, "local_docker://test-session/SparseNNAppDef/trainer", regex="INFO"
        )
        self.assertSetEqual(
            {
                f"{GREEN}trainer/0{ENDC} INFO foo",
                f"{GREEN}trainer/1{ENDC} INFO foo",
                f"{GREEN}trainer/2{ENDC} INFO foo",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(out.getvalue().split("\n")),
        )

    @patch(RUNNER, new_callable=MockRunner)
    def test_cmd_log_one_replica(self, mock_runner: MagicMock) -> None:
        out = io.StringIO()
        get_logs(
            out, "local_docker://test-session/SparseNNAppDef/trainer/0", regex=None
        )
        self.assertSetEqual(
            {
                f"{GREEN}trainer/0{ENDC} INFO foo",
                f"{GREEN}trainer/0{ENDC} ERROR bar",
                f"{GREEN}trainer/0{ENDC} WARN baz",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(out.getvalue().split("\n")),
        )

    @patch(RUNNER, new_callable=MockRunner)
    def test_cmd_log_some_replicas(self, mock_runner: MagicMock) -> None:
        out = io.StringIO()
        get_logs(
            out, "local_docker://test-session/SparseNNAppDef/trainer/0,2", regex="WARN"
        )
        self.assertSetEqual(
            {
                f"{GREEN}trainer/0{ENDC} WARN baz",
                f"{GREEN}trainer/2{ENDC} WARN baz",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(out.getvalue().split("\n")),
        )

    @patch(RUNNER, new_callable=MockPendingRunner)
    def test_cmd_log_wait_to_start(self, mock_runner: MagicMock) -> None:
        out = io.StringIO()
        with self.assertLogs(level="INFO") as cm:
            get_logs(
                out, "local_docker://test-session/SparseNNAppdef/trainer", regex=None
            )
        self.assertIn("Waiting", "\n".join(cm.output))

    @patch(RUNNER, new_callable=MockRunner)
    def test_print_log_lines_throws(self, mock_runner: MagicMock) -> None:
        # makes sure that when the function executed in the threadpool
        # errors out; we raise the exception all the way through
        with patch.object(mock_runner, "log_lines") as log_lines_mock:
            log_lines_mock.side_effect = RuntimeError
            with self.assertRaises(RuntimeError):
                get_logs(
                    sys.stdout,
                    "local://test-session/SparseNNAppDef/trainer/0,1",
                    regex=None,
                )

    def test_validate(self) -> None:
        validate("kubernetes://session/queue:name-1234")
        validate("kubernetes://session/queue:name-1234/role")
        validate("kubernetes://session/queue:name-1234/role/1")
        validate("kubernetes://session/queue:name-1234/role/1,2,3")
        validate("two_part://session/queue:name-1234/role/1,2,3")

        with self.assertRaisesRegex(SystemExit, "1"):
            validate("kubernetes://session")

        with self.assertRaisesRegex(SystemExit, "1"):
            validate("session/name/role")

        with self.assertRaisesRegex(SystemExit, "1"):
            validate("kubernetes://session/queue:name-1234/role/")

    @patch(RUNNER, new_callable=MockRunner)
    def test_cmd_log_broken_pipe(self, mock_runner: MagicMock) -> None:
        out = MagicMock(spec=io.StringIO)
        out.write = MagicMock(side_effect=BrokenPipeError)
        # should not raise; BrokenPipeError from the output stream is silenced
        get_logs(
            out,
            "local_docker://test-session/SparseNNAppDef/trainer/0",
            regex=None,
        )

    def test_prefix_line(self) -> None:
        self.assertEqual(_prefix_line("<prefix>", "test\n"), "<prefix>test\n")
        self.assertEqual(_prefix_line("<prefix>", "\rtest\n"), "\r<prefix>test\n")
        self.assertEqual(
            _prefix_line("<prefix>", "foo\rtest\n"), "<prefix>foo\r<prefix>test\n"
        )
        self.assertEqual(
            _prefix_line("<prefix>", "foo\ntest\n"), "<prefix>foo\n<prefix>test\n"
        )
