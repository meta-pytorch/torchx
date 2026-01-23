#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import dataclasses
import io
import os
import shutil
import signal
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

from torchx.cli.argparse_util import ArgOnceAction, torchxconfig
from torchx.cli.cmd_run import (
    _parse_component_name_and_args,
    CmdBuiltins,
    CmdRun,
    torchx_run_args_from_argparse,
    torchx_run_args_from_json,
    TorchXRunArgs,
)
from torchx.runner.config import ENV_TORCHXCONFIG
from torchx.schedulers.local_scheduler import SignalException
from torchx.specs import AppDryRunInfo, CfgVal


@contextmanager
def cwd(path: str) -> Generator[None, None, None]:
    orig_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_cwd)


class CmdRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

        # create empty .torchxconfig so that user .torchxconfig is not picked up
        empty_config = self.tmpdir / ".torchxconfig"
        empty_config.touch()
        self.mock_env = patch.dict(os.environ, {ENV_TORCHXCONFIG: str(empty_config)})
        self.mock_env.start()

        self.parser = argparse.ArgumentParser()
        self.cmd_run = CmdRun()
        self.cmd_run.add_arguments(self.parser)

    def tearDown(self) -> None:
        self.mock_env.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        ArgOnceAction.called_args = set()
        torchxconfig.called_args = set()

    def test_run_with_multiple_scheduler_args(self) -> None:
        args = ["--scheduler_args", "first_args", "--scheduler_args", "second_args"]
        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(args)
        self.assertEqual(cm.exception.code, 1)

    def test_run_with_multiple_schedule_args(self) -> None:
        args = [
            "--scheduler",
            "local_cwd",
            "--scheduler",
            "local_cwd",
        ]

        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(args)
        self.assertEqual(cm.exception.code, 1)

    def test_run_with_user_conf_abs_path(self) -> None:
        args = self.parser.parse_args(
            [
                "--scheduler",
                "local_cwd",
                str(Path(__file__).parent / "components.py:touch"),
                "--file",
                str(self.tmpdir / "foobar.txt"),
            ]
        )
        self.cmd_run.run(args)
        self.assertEqual("cli_run", os.environ["TORCHX_CONTEXT_NAME"])
        self.assertTrue(os.path.isfile(str(self.tmpdir / "foobar.txt.test")))

    def test_run_with_relpath(self) -> None:
        # should pick up test/examples/touch.torchx (not the builtin)
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--scheduler",
                    "local_cwd",
                    str(Path(__file__).parent / "components.py:touch_v2"),
                    "--file",
                    str(self.tmpdir / "foobar.txt"),
                ]
            )

            self.cmd_run.run(args)
            self.assertTrue(os.path.isfile(str(self.tmpdir / "foobar.txt.testv2")))

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_run_with_log(self, stderr: io.StringIO) -> None:
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--log",
                    "--wait",
                    "--scheduler",
                    "local_cwd",
                    "-cfg",
                    f"log_dir={self.tmpdir}",
                    str(Path(__file__).parent / "components.py:echo_stderr"),
                    "--msg",
                    "toast",
                ]
            )

            self.cmd_run.run(args)
        self.assertRegex(stderr.getvalue(), "echo/0.*toast")

    @patch(
        "torchx.runner.Runner.wait", side_effect=SignalException("msg", signal.SIGTERM)
    )
    @patch("torchx.schedulers.local_scheduler.LocalScheduler.close")
    def test_run_terminate_on_received_signal(
        self,
        mock_scheduler_close: MagicMock,
        _,
    ) -> None:
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--scheduler",
                    "local_cwd",
                    str(Path(__file__).parent / "components.py:touch_v2"),
                    "--file",
                    str(self.tmpdir / "foobar.txt"),
                ]
            )

            with self.assertRaises(SignalException):
                self.cmd_run.run(args)

            mock_scheduler_close.assert_called()

    def test_run_missing(self) -> None:
        args = self.parser.parse_args(
            [
                "--scheduler",
                "local_cwd",
                "1234_does_not_exist.torchx",
            ]
        )

        with self.assertRaises(SystemExit):
            self.cmd_run.run(args)

    def test_conf_file_missing(self) -> None:
        # guards against existing .torchxconfig files
        # in user's $HOME or the CWD where the test is launched from
        with patch(
            "torchx.runner.config.DEFAULT_CONFIG_DIRS", return_value=[self.tmpdir]
        ):
            args = self.parser.parse_args(
                [
                    "--scheduler",
                    "local_cwd",
                ]
            )
            with self.assertRaises(SystemExit):
                self.cmd_run.run(args)

    @patch("torchx.runner.Runner.run")
    def test_run_dryrun(self, mock_runner_run: MagicMock) -> None:
        args = self.parser.parse_args(
            [
                "--dryrun",
                "--parent_run_id",
                "experiment_1",
                "--scheduler",
                "local_cwd",
                "utils.echo",
                "--image",
                "/tmp",
            ]
        )
        self.cmd_run.run(args)
        mock_runner_run.assert_not_called()

    @patch("torchx.runner.Runner.dryrun_component")
    def test_store_experiment_id(self, mock_runner_run: MagicMock) -> None:
        args = self.parser.parse_args(
            [
                "--dryrun",
                "--parent_run_id",
                "experiment_1",
                "--scheduler",
                "local_cwd",
                "utils.echo",
                "--image",
                "/tmp",
            ]
        )

        app_run_info_stub = AppDryRunInfo("req", lambda x: x)
        req_type_dataclass = dataclasses.make_dataclass("T", [])
        app_run_info_stub._app = req_type_dataclass()
        mock_runner_run.return_value = app_run_info_stub

        self.cmd_run.run(args)

        # compatible with python 3.7
        call_kwargs = mock_runner_run.call_args[-1]
        self.assertEqual(call_kwargs["parent_run_id"], "experiment_1")

    def test_parse_component_name_and_args_no_default(self) -> None:
        # set dirs to test tmpdir so tests don't accidentally pick up user's $HOME/.torchxconfig
        dirs = [str(self.tmpdir)]

        sp = argparse.ArgumentParser(prog="test")
        self.assertEqual(
            ("utils.echo", []),
            _parse_component_name_and_args(["utils.echo"], sp, dirs),
        )
        self.assertEqual(
            ("utils.echo", []),
            _parse_component_name_and_args(["--", "utils.echo"], sp, dirs),
        )
        self.assertEqual(
            ("utils.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["utils.echo", "--msg", "hello"], sp, dirs),
        )

        self.assertEqual(
            ("utils.echo", ["--msg", "hello", "--", "--"]),
            _parse_component_name_and_args(
                ["utils.echo", "--msg", "hello", "--", "--"], sp, dirs
            ),
        )

        self.assertEqual(
            ("utils.echo", ["--msg", "hello", "-", "-"]),
            _parse_component_name_and_args(
                ["utils.echo", "--msg", "hello", "-", "-"], sp, dirs
            ),
        )

        self.assertEqual(
            ("utils.echo", ["--msg", "hello", "-  ", "-  "]),
            _parse_component_name_and_args(
                ["utils.echo", "--msg", "hello", "-  ", "-  "], sp, dirs
            ),
        )

        self.assertEqual(
            (
                "fb.python.binary",
                [
                    "--img",
                    "lex_ig_o3_package",
                    "-m",
                    "dper_lib.instagram.pyper_v2.teams.stories.train",
                    "--",
                    "-m",
                ],
            ),
            _parse_component_name_and_args(
                [
                    "fb.python.binary",
                    "--img",
                    "lex_ig_o3_package",
                    "-m",
                    "dper_lib.instagram.pyper_v2.teams.stories.train",
                    "--",
                    "-m",
                ],
                sp,
                dirs,
            ),
        )

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["--"], sp, dirs)

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["--msg", "hello"], sp, dirs)

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["-m", "hello"], sp, dirs)

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["-m", "hello", "-m", "repeate"], sp, dirs)

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(
                ["--msg", "hello", "--msg", "repeate"], sp, dirs
            )

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(
                ["--msg  ", "hello", "--msg     ", "repeate"], sp, dirs
            )

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(
                ["--m", "hello", "--", "--msg", "msg", "--msg", "repeate"], sp, dirs
            )

    def test_parse_component_name_and_args_with_default(self) -> None:
        sp = argparse.ArgumentParser(prog="test")
        dirs = [str(self.tmpdir)]

        with open(Path(self.tmpdir) / ".torchxconfig", "w") as f:
            f.write(
                """#
[cli:run]
component = custom.echo
            """
            )

        self.assertEqual(
            ("utils.echo", []), _parse_component_name_and_args(["utils.echo"], sp, dirs)
        )
        self.assertEqual(
            ("utils.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["utils.echo", "--msg", "hello"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", []),
            _parse_component_name_and_args([], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", []),
            _parse_component_name_and_args(["--"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["--", "--msg", "hello"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["--msg", "hello"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", ["-m", "hello"]),
            _parse_component_name_and_args(["-m", "hello"], sp, dirs),
        )

    def test_verify_no_extra_args_stdin_only(self) -> None:
        """Test that only --stdin is allowed when using stdin mode."""
        args = self.parser.parse_args(["--stdin"])
        # Should not raise any exception
        self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_no_stdin(self) -> None:
        """Test that verification is skipped when not using stdin."""
        args = self.parser.parse_args(["--scheduler", "local_cwd", "utils.echo"])
        # Should not raise any exception
        self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_component_name(self) -> None:
        """Test that component name conflicts with stdin."""
        args = self.parser.parse_args(["--stdin", "utils.echo"])
        with self.assertRaises(SystemExit):
            self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_scheduler_args(self) -> None:
        """Test that scheduler_args conflicts with stdin."""
        args = self.parser.parse_args(["--stdin", "--scheduler_args", "cluster=test"])
        with self.assertRaises(SystemExit):
            self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_scheduler(self) -> None:
        """Test that non-default scheduler conflicts with stdin."""
        args = self.parser.parse_args(["--stdin", "--scheduler", "kubernetes"])
        with self.assertRaises(SystemExit):
            self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_boolean_flags(self) -> None:
        """Test that boolean flags conflict with stdin."""
        boolean_flags = ["--wait", "--log", "--tee_logs"]
        for flag in boolean_flags:
            args = self.parser.parse_args(["--stdin", flag])
            with self.assertRaises(SystemExit):
                self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_dryrun_pass(self) -> None:
        """Test that dryrun is allowed."""
        args = self.parser.parse_args(["--stdin", "--dryrun"])
        self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_value_args(self) -> None:
        """Test that arguments with values conflict with stdin."""
        args = self.parser.parse_args(["--stdin", "--workspace", "/custom/path"])
        with self.assertRaises(SystemExit):
            self.cmd_run.verify_no_extra_args(args)

        args = self.parser.parse_args(["--stdin", "--parent_run_id", "experiment_123"])
        with self.assertRaises(SystemExit):
            self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_multiple_conflicts(self) -> None:
        """Test that multiple conflicting arguments with stdin are detected."""
        args = self.parser.parse_args(
            ["--stdin", "--dryrun", "--wait", "--scheduler_args", "cluster=test"]
        )
        with self.assertRaises(SystemExit):
            self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_default_scheduler(self) -> None:
        """Test that using default scheduler with stdin doesn't conflict."""
        # Get the default scheduler and use it explicitly - should not conflict
        from torchx.schedulers import get_default_scheduler_name

        default_scheduler = get_default_scheduler_name()

        args = self.parser.parse_args(["--stdin", "--scheduler", default_scheduler])
        # Should not raise any exception since it's the same as default
        self.cmd_run.verify_no_extra_args(args)

    def test_verify_no_extra_args_stdin_with_default_workspace(self) -> None:
        """Test that using default workspace with stdin doesn't conflict."""
        # Get the actual default workspace from a fresh parser
        fresh_parser = argparse.ArgumentParser()
        fresh_cmd_run = CmdRun()
        fresh_cmd_run.add_arguments(fresh_parser)

        # Find the workspace argument's default value
        workspace_default = None
        for action in fresh_parser._actions:
            if action.dest == "workspace":
                workspace_default = action.default
                break

        self.assertIsNotNone(
            workspace_default, "workspace argument should have a default"
        )

        # Use the actual default - this should not conflict with stdin
        args = fresh_parser.parse_args(["--stdin", "--workspace", workspace_default])
        # Should not raise any exception since it's the same as default
        fresh_cmd_run.verify_no_extra_args(args)


class CmdBuiltinTest(unittest.TestCase):
    def test_run(self) -> None:
        # nothing to assert, just make sure the supported schedulers print runopts

        parser = argparse.ArgumentParser()
        cmd_builtins = CmdBuiltins()
        cmd_builtins.add_arguments(parser)
        args = parser.parse_args([])
        cmd_builtins.run(args)

    def test_builtins(self) -> None:
        cmd_builtins = CmdBuiltins()
        builtins = cmd_builtins._builtins()
        # make sure there's at least one
        # there will always be one (example.torchx)
        self.assertTrue(len(builtins) > 0)
        for component in builtins.values():
            self.assertListEqual([], component.validation_errors)

    def test_print_builtin(self) -> None:
        parser = argparse.ArgumentParser()

        cmd_builtins = CmdBuiltins()
        cmd_builtins.add_arguments(parser)

        cmd_builtins.run(parser.parse_args(["--print", "dist.ddp"]))
        # nothing to assert, just make sure it runs


class TorchXRunArgsTest(unittest.TestCase):
    def test_torchx_run_args_from_json(self) -> None:
        # Test valid input with all required fields
        json_data = {
            "scheduler": "local",
            "scheduler_args": {"cluster": "test"},
            "component_name": "test_component",
        }
        result = torchx_run_args_from_json(json_data)

        self.assertIsInstance(result, TorchXRunArgs)
        self.assertEqual(result.scheduler, "local")
        self.assertEqual(result.scheduler_args, {"cluster": "test"})
        self.assertEqual(result.component_name, "test_component")
        # Check defaults are set
        self.assertEqual(result.dryrun, False)
        self.assertEqual(result.wait, False)
        self.assertEqual(result.log, False)
        self.assertEqual(result.workspace, f"{Path.cwd()}")
        self.assertEqual(result.parent_run_id, None)
        self.assertEqual(result.tee_logs, False)
        self.assertEqual(result.component_args, {})
        self.assertEqual(result.component_args_str, [])

        # Test valid input with optional fields provided
        json_data_with_optionals = {
            "scheduler": "k8s",
            "scheduler_args": {"namespace": "default"},
            "component_name": "my_component",
            "component_args": {"param": "test"},
            "component_args_str": ["--param test"],
            "dryrun": True,
            "wait": True,
            "log": True,
            "workspace": "/custom/path",
            "parent_run_id": "parent123",
            "tee_logs": True,
        }
        result2 = torchx_run_args_from_json(json_data_with_optionals)

        self.assertEqual(result2.scheduler, "k8s")
        self.assertEqual(result2.scheduler_args, {"namespace": "default"})
        self.assertEqual(result2.component_name, "my_component")
        self.assertEqual(result2.component_args, {"param": "test"})
        self.assertEqual(result2.component_args_str, ["--param test"])
        self.assertEqual(result2.dryrun, True)
        self.assertEqual(result2.wait, True)
        self.assertEqual(result2.log, True)
        self.assertEqual(result2.workspace, "/custom/path")
        self.assertEqual(result2.parent_run_id, "parent123")
        self.assertEqual(result2.tee_logs, True)

        # Test missing required field - scheduler
        json_missing_scheduler = {
            "scheduler_args": {"cluster": "test"},
            "component_name": "test_component",
        }
        with self.assertRaises(ValueError) as cm:
            torchx_run_args_from_json(json_missing_scheduler)
        self.assertEqual(
            "The following required fields are missing: scheduler", cm.exception.args[0]
        )

        # Test missing required field - component_name
        json_missing_component = {
            "scheduler": "local",
            "scheduler_args": {"cluster": "test"},
        }
        with self.assertRaises(ValueError) as cm:
            torchx_run_args_from_json(json_missing_component)
        self.assertEqual(
            "The following required fields are missing: component_name",
            cm.exception.args[0],
        )

        # Test missing required field - scheduler_args
        json_missing_scheduler_args = {
            "scheduler": "local",
            "component_name": "test_component",
        }
        with self.assertRaises(ValueError) as cm:
            torchx_run_args_from_json(json_missing_scheduler_args)
        self.assertEqual(
            "The following required fields are missing: scheduler_args",
            cm.exception.args[0],
        )

        # Test missing multiple required fields
        json_missing_multiple = {"dryrun": True, "wait": False}
        with self.assertRaises(ValueError) as cm:
            torchx_run_args_from_json(json_missing_multiple)
        error_msg = str(cm.exception)
        self.assertIn("The following required fields are missing:", error_msg)
        self.assertIn("scheduler", error_msg)
        self.assertIn("scheduler_args", error_msg)
        self.assertIn("component_name", error_msg)

        # Test unknown fields cause ValueError
        json_with_unknown = {
            "scheduler": "local",
            "scheduler_args": {"cluster": "test"},
            "component_name": "test_component",
            "component_args": {"arg1": "value1"},
            "unknown_field": "should_be_ignored",
            "another_unknown": 123,
        }

        with self.assertRaises(ValueError) as cm:
            torchx_run_args_from_json(json_with_unknown)
        self.assertIn(
            "The following fields are not part of the run command:",
            cm.exception.args[0],
        )
        self.assertIn("unknown_field", cm.exception.args[0])
        self.assertIn("another_unknown", cm.exception.args[0])

        # Test empty JSON
        with self.assertRaises(ValueError) as cm:
            torchx_run_args_from_json({})
        self.assertIn("The following required fields are missing:", str(cm.exception))

        # Test minimal valid input (only required fields)
        json_minimal = {
            "scheduler": "local",
            "scheduler_args": {},
            "component_name": "minimal_component",
        }
        result4 = torchx_run_args_from_json(json_minimal)

        self.assertEqual(result4.scheduler, "local")
        self.assertEqual(result4.scheduler_args, {})
        self.assertEqual(result4.component_name, "minimal_component")
        self.assertEqual(result4.component_args, {})
        self.assertEqual(result4.component_args_str, [])

    def test_torchx_run_args_from_argparse(self) -> None:
        # This test case isn't as important, since if the dataclass is being
        # init with argparse, the argparsing will have handled most of the missing
        # logic etc
        # Create a mock argparse.Namespace object
        args = argparse.Namespace()
        args.scheduler = "k8s"
        args.dryrun = True
        args.wait = False
        args.log = True
        args.workspace = "/custom/workspace"
        args.parent_run_id = "parent_123"
        args.tee_logs = False

        component_name = "test_component"
        component_args = ["--param1", "value1", "--param2", "value2"]
        scheduler_cfg: dict[str, CfgVal] = {
            "cluster": "test_cluster",
            "namespace": "default",
        }

        result = torchx_run_args_from_argparse(
            args=args,
            component_name=component_name,
            component_args=component_args,
            scheduler_cfg=scheduler_cfg,
        )

        self.assertIsInstance(result, TorchXRunArgs)
        self.assertEqual(result.component_name, "test_component")
        self.assertEqual(result.scheduler, "k8s")
        self.assertEqual(result.scheduler_args, {})
        self.assertEqual(
            result.scheduler_cfg, {"cluster": "test_cluster", "namespace": "default"}
        )
        self.assertEqual(result.dryrun, True)
        self.assertEqual(result.wait, False)
        self.assertEqual(result.log, True)
        self.assertEqual(result.workspace, "/custom/workspace")
        self.assertEqual(result.parent_run_id, "parent_123")
        self.assertEqual(result.tee_logs, False)
        self.assertEqual(result.component_args, {})
        self.assertEqual(
            result.component_args_str, ["--param1", "value1", "--param2", "value2"]
        )
