#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest
from unittest.mock import MagicMock, patch

import fsspec
import psutil
from torchx.apps.utils.process_monitor import main, TIMEOUT_EXIT_CODE


class ProcessTest(unittest.TestCase):
    def tearDown(self) -> None:
        current_process = psutil.Process()
        self.assertEqual(
            len(current_process.children()), 0, "zombie children processes!"
        )

    def test_returncode(self) -> None:
        with self.assertRaisesRegex(SystemExit, r"^0$"):
            main(
                [
                    "--timeout",
                    "60",
                    "echo",
                    "hello",
                ]
            )

        with self.assertRaisesRegex(SystemExit, r"^123$"):
            main(["--timeout", "60", "--", "bash", "-c", "exit 123"])

    def test_proccess_args(self) -> None:
        with self.assertRaisesRegex(SystemExit, r"^0$"):
            main(
                [
                    "echo",
                    "--some",
                    "-args",
                ]
            )

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_timeout(self, stdout: MagicMock) -> None:
        with self.assertRaisesRegex(SystemExit, r"^-15$"):
            main(
                [
                    "--timeout",
                    "0.0001",
                    "--poll_rate",
                    "0.0001",
                    "sleep",
                    "60",
                ]
            )
        self.assertIn("reached timeout, terminating...", stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_timeout_kill(self, stdout: MagicMock) -> None:
        with self.assertRaisesRegex(SystemExit, r"^-9$"):
            main(
                [
                    "--timeout",
                    "0.1",
                    "--poll_rate",
                    "0.1",
                    "--kill_timeout",
                    "0.1",
                    "--",
                    "bash",
                    "-c",
                    "trap 'echo received term' TERM; sleep 10",
                ]
            )
        self.assertIn("reached safe termination timeout, killing...", stdout.getvalue())

    def test_start_on_file(self) -> None:
        start_on_file = "memory://start"
        fs, path = fsspec.core.url_to_fs(start_on_file)

        args = [
            "--timeout",
            "0.001",
            "--poll_rate",
            "0.001",
            "--start_on_file",
            start_on_file,
            "--",
            "echo",
            "banana",
        ]

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaisesRegex(SystemExit, f"^{TIMEOUT_EXIT_CODE}$"):
                main(args)
            self.assertIn(
                "reached timeout before launching, terminating...", stdout.getvalue()
            )

        fs.touch(path)

        with self.assertRaisesRegex(SystemExit, r"^0$"):
            main(args)

    def test_exit_on_file(self) -> None:
        start_on_file = "memory://end"
        fs, path = fsspec.core.url_to_fs(start_on_file)
        args = [
            "--poll_rate",
            "0.001",
            "--exit_on_file",
            start_on_file,
            "--",
            "sleep",
            "60",
        ]

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaisesRegex(SystemExit, r"^-15$"):
                main(
                    [
                        "--timeout",
                        "0.001",
                    ]
                    + args
                )
            self.assertIn("reached timeout, terminating", stdout.getvalue())

        fs.touch(path)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with self.assertRaisesRegex(SystemExit, r"^-15$"):
                main(
                    [
                        "--timeout",
                        "60",
                    ]
                    + args
                )
            self.assertIn("exists, terminating", stdout.getvalue())
