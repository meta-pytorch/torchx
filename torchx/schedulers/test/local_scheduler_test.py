#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from os.path import join
from typing import Callable, Generator
from unittest import mock
from unittest.mock import MagicMock, patch

from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.local_scheduler import (
    _join_PATH,
    create_scheduler,
    CWDImageProvider,
    ENV_CUDA_VISIBLE_DEVICES,
    LocalDirectoryImageProvider,
    LocalOpts,
    LocalScheduler,
    make_unique,
    PopenRequest,
)
from torchx.specs.api import (
    AppDef,
    AppDryRunInfo,
    AppState,
    is_terminal,
    macros,
    Resource,
    Role,
)
from torchx.util.types import none_throws

from .test_util import write_shell_script

LOCAL_SCHED_DEVICE_COUNT = (
    "torchx.schedulers.local_scheduler.LocalScheduler._cuda_device_count"
)

LOCAL_DIR_IMAGE_PROVIDER_FETCH = (
    "torchx.schedulers.local_scheduler.LocalDirectoryImageProvider.fetch"
)


def start_sleep_processes(
    test_dir: str,
    mp_queue: mp.Queue,
    num_replicas: int = 2,
) -> None:
    """
    Starts processes
    """

    print("starting start_sleep_processes")
    role = Role(
        name="sleep",
        image=test_dir,
        entrypoint="sleep.sh",
        args=["600"],  # seconds
        num_replicas=num_replicas,
    )

    app = AppDef(name="test_app", roles=[role])
    cfg = LocalOpts({"log_dir": test_dir})

    scheduler = LocalScheduler(
        session_name="test_session", image_provider_class=LocalDirectoryImageProvider
    )
    app_id = scheduler.submit(app, cfg)

    my_pid = os.getpid()
    mp_queue.put(my_pid)
    for app in scheduler._apps.values():
        for replicas in app.role_replicas.values():
            for replica in replicas:
                mp_queue.put(replica.proc.pid)

    elapsed = 0
    interval = 0.5
    while elapsed < 300:  # 5min timeout
        try:
            app_status = scheduler.describe(app_id)
            if none_throws(app_status).state == AppState.SUCCEEDED:
                raise RuntimeError("Child processes should not succeed in this test")
            time.sleep(interval)
            elapsed += interval
        finally:
            scheduler.close()
            break


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class LocalDirImageProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="LocalDirImageProviderTest")
        self.test_dir_name = os.path.basename(self.test_dir)
        self.maxDiff = None  # get full diff on assertion error

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_fetch_abs_path(self) -> None:
        provider = LocalDirectoryImageProvider(cfg={})
        self.assertEqual(self.test_dir, provider.fetch(self.test_dir))

    def test_fetch_relative_path_should_throw(self) -> None:
        provider = LocalDirectoryImageProvider(cfg={})
        with self.assertRaises(ValueError):
            provider.fetch(self.test_dir_name)

    def test_fetch_does_not_exist_should_throw(self) -> None:
        non_existent_dir = join(self.test_dir, "non_existent_dir")
        provider = LocalDirectoryImageProvider(cfg={})
        with self.assertRaises(ValueError):
            provider.fetch(non_existent_dir)

    def test_get_replica_param(self) -> None:
        provider = LocalDirectoryImageProvider(cfg={})
        role = Role(name="foo", image="/tmp", entrypoint="a", args=["b", "c"])
        self.assertEqual(
            ["a", "b", "c"],
            provider.get_replica_param(
                img_root=provider.fetch(role.image), role=role
            ).args,
        )


class CWDImageProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None  # get full diff on assertion error
        self.provider = CWDImageProvider(cfg={})

    def test_fetch(self) -> None:
        self.assertEqual(os.getcwd(), self.provider.fetch("/strawberry/toast"))

    def test_cwd(self) -> None:
        self.assertEqual(os.getcwd(), self.provider.get_cwd("/strawberry/toast"))

    def test_replica_param(self) -> None:
        role = Role(name="foo", image="ignored", entrypoint="a", args=["b", "c"])
        rp = self.provider.get_replica_param(
            img_root=self.provider.fetch(role.image), role=role
        )
        self.assertEqual(rp.args, ["a", "b", "c"])

    def test_get_entrypoint(self) -> None:
        role = Role(
            "role1",
            image="some/dir",
            entrypoint="entrypoint.sh",
        )
        self.assertEqual(self.provider.get_entrypoint("asdf", role), "entrypoint.sh")

    def test_create_scheduler(self) -> None:
        sched = create_scheduler("foo", cache_size=20, extra_paths=["foo"])
        self.assertEqual(sched.session_name, "foo")
        self.assertEqual(sched._image_provider_class, CWDImageProvider)
        self.assertEqual(sched._cache_size, 20)
        self.assertEqual(len(sched._extra_paths), 1)


LOCAL_SCHEDULER_MAKE_UNIQUE = "torchx.schedulers.local_scheduler.make_unique"


class LocalSchedulerTestUtil(abc.ABC):
    scheduler: LocalScheduler

    def wait(
        self,
        app_id: str,
        scheduler: LocalScheduler | None = None,
        timeout: float = 30,
        wait_for: Callable[[AppState], bool] = is_terminal,
    ) -> DescribeAppResponse | None:
        """
        Waits for the app to finish or raise TimeoutError upon timeout (in seconds).
        If no timeout is specified waits indefinitely.

        Returns:
            The last return value from ``describe()``
        """
        scheduler_ = scheduler or self.scheduler

        interval = 0.1
        expiry = time.time() + timeout
        while expiry > time.time():
            desc = scheduler_.describe(app_id)

            if desc is None:
                return None
            elif wait_for(desc.state):
                return desc

            time.sleep(interval)
        raise TimeoutError(f"timed out waiting for app: {app_id}")


class LocalDirectorySchedulerTest(unittest.TestCase, LocalSchedulerTestUtil):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")
        write_shell_script(self.test_dir, "touch.sh", ["touch $1"])
        write_shell_script(self.test_dir, "env.sh", ["env > $1"])
        write_shell_script(self.test_dir, "fail.sh", ["exit 1"])
        write_shell_script(self.test_dir, "sleep.sh", ["sleep $1"])
        write_shell_script(self.test_dir, "echo_stdout.sh", ["echo $1"])
        write_shell_script(self.test_dir, "echo_stderr.sh", ["echo $1 1>&2"])
        write_shell_script(
            self.test_dir,
            "echo_range.sh",
            ["for i in $(seq 0 $1); do echo $i 1>&2; sleep $2; done"],
        )
        write_shell_script(self.test_dir, "echo_env_foo.sh", ["echo $FOO 1>&2"])
        self.scheduler = LocalScheduler(
            session_name="test_session",
            image_provider_class=LocalDirectoryImageProvider,
        )
        write_shell_script(
            self.test_dir, "print_invalid_utf8.sh", ["printf '\\x02\\xc5\\xd8'"]
        )

    def tearDown(self) -> None:
        self.scheduler.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_submit(self) -> None:
        # make sure the macro substitution works
        # touch a file called {app_id}_{replica_id} in the img_root directory (self.test_dir)
        test_file_name = f"{macros.app_id}_{macros.replica_id}"
        num_replicas = 2
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="touch.sh",
            args=[join(f"{macros.img_root}", test_file_name)],
            num_replicas=num_replicas,
        )
        app = AppDef(name="test_app", roles=[role])
        expected_app_id = make_unique(app.name)
        cfg = {"log_dir": self.test_dir}
        with patch(LOCAL_SCHEDULER_MAKE_UNIQUE, return_value=expected_app_id):
            app_id = self.scheduler.submit(app, cfg)

        self.assertEqual(f"{expected_app_id}", app_id)
        app_response = self.wait(app_id)
        assert app_response is not None
        self.assertEqual(AppState.SUCCEEDED, app_response.state)

        for i in range(num_replicas):
            self.assertTrue(
                os.path.isfile(join(self.test_dir, f"{expected_app_id}_{i}"))
            )

        role = Role("role1", image=self.test_dir, entrypoint="fail.sh", num_replicas=2)
        app = AppDef(name="test_app", roles=[role])
        expected_app_id = make_unique(app.name)
        with patch(LOCAL_SCHEDULER_MAKE_UNIQUE, return_value=expected_app_id):
            app_id = self.scheduler.submit(app, cfg)

        desc = self.wait(app_id)
        assert desc is not None
        self.assertEqual(f"{expected_app_id}", app_id)
        self.assertEqual(AppState.FAILED, desc.state)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertFalse(self.scheduler._created_tmp_log_dir)

    def test_submit_cleanup(self) -> None:
        test_file_name = f"{macros.app_id}_{macros.replica_id}"
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="touch.sh",
            args=[join(f"{macros.img_root}", test_file_name)],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role])
        self.scheduler.submit(app, cfg={})
        self.scheduler.close()
        base_log_dir = self.scheduler._base_log_dir
        self.assertIsNotNone(base_log_dir)
        self.assertFalse(os.path.exists(base_log_dir))
        self.assertTrue(self.scheduler._created_tmp_log_dir)

    def test_macros_env(self) -> None:
        # make sure the macro substitution works
        # touch a file called {app_id}_{replica_id} in the img_root directory (self.test_dir)
        test_file_name = f"{macros.app_id}_{macros.replica_id}"
        num_replicas = 2
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="env.sh",
            args=[join(f"{macros.img_root}", test_file_name)],
            env={"FOO": join(macros.img_root, "config.yaml")},
            num_replicas=num_replicas,
        )
        app = AppDef(name="test_app", roles=[role])
        expected_app_id = make_unique(app.name)
        with patch(LOCAL_SCHEDULER_MAKE_UNIQUE, return_value=expected_app_id):
            cfg = {"log_dir": self.test_dir}
            app_id = self.scheduler.submit(app, cfg)

        self.assertEqual(f"{expected_app_id}", app_id)
        app_response = self.wait(app_id)
        assert app_response is not None
        self.assertEqual(AppState.SUCCEEDED, app_response.state)

        for i in range(num_replicas):
            file = join(self.test_dir, f"{expected_app_id}_{i}")
            self.assertTrue(os.path.isfile(file))

            # check environment variable is substituted
            with open(file, "r") as f:
                body = f.read()
            foo_env = join(self.test_dir, "config.yaml")
            self.assertIn(f"FOO={foo_env}", body)

    @mock.patch.dict(os.environ, {"FOO": "bar"})
    def test_submit_inherit_parent_envs(self) -> None:
        role = Role("echo_foo", image=self.test_dir, entrypoint="echo_env_foo.sh")
        app = AppDef(name="check_foo_env_var", roles=[role])
        app_id = self.scheduler.submit(app, {"log_dir": self.test_dir})
        for line in self.scheduler.log_iter(app_id, "echo_foo"):
            self.assertEqual("bar\n", line)

        desc = self.wait(app_id, self.scheduler)
        assert desc is not None
        self.assertEqual(AppState.SUCCEEDED, desc.state)

    @mock.patch("subprocess.Popen")
    @mock.patch.dict(
        os.environ, {"PATH": "/bin:/usr/bin", "TORCHELASTIC_ERROR_FILE": "ignored"}
    )
    def test_child_process_env_append_cwd(self, popen_mock: MagicMock) -> None:
        popen_request = self.scheduler._to_popen_request(
            app=AppDef(
                name="_",
                roles=[
                    Role(name="foo", image=self.test_dir, env={"PATH": "/home/bob"}),
                ],
            ),
            cfg={"prepend_cwd": False},
        )

        # the test scheduler is hooked up with LocalDirectoryImageProvider
        # which treats the role's image as CWD
        replica_params = popen_request.role_params["foo"][0]
        self.assertEqual(f"/home/bob:{self.test_dir}", replica_params.env["PATH"])

        # _popen actually adds in the os.environ[PATH]
        self.scheduler._popen(
            role_name="foo", replica_id=0, replica_params=replica_params
        )

        self.assertEqual(
            # for python 3.7 BC get call_args.kwargs by index
            f"/home/bob:{self.test_dir}:/bin:/usr/bin",
            popen_mock.call_args[1]["env"]["PATH"],
        )

    @mock.patch("subprocess.Popen")
    @mock.patch.dict(
        os.environ, {"PATH": "/bin:/usr/bin", "TORCHELASTIC_ERROR_FILE": "ignored"}
    )
    def test_child_process_env_prepend_cwd(self, popen_mock: MagicMock) -> None:
        popen_request = self.scheduler._to_popen_request(
            app=AppDef(
                name="_",
                roles=[
                    Role(name="foo", image=self.test_dir, env={"PATH": "/home/bob"}),
                ],
            ),
            cfg={"prepend_cwd": True},
        )

        # the test scheduler is hooked up with LocalDirectoryImageProvider
        # which treats the role's image as CWD
        replica_params = popen_request.role_params["foo"][0]
        self.assertEqual(f"{self.test_dir}:/home/bob", replica_params.env["PATH"])

        # _popen actually adds in the os.environ[PATH]
        self.scheduler._popen(
            role_name="foo", replica_id=0, replica_params=replica_params
        )

        self.assertEqual(
            # for python 3.7 BC get call_args.kwargs by index
            f"{self.test_dir}:/home/bob:/bin:/usr/bin",
            popen_mock.call_args[1]["env"]["PATH"],
        )

    @mock.patch("subprocess.Popen")
    @mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": "ignored"}, clear=True)
    def test_child_process_env_none(self, popen_mock: MagicMock) -> None:
        popen_request = self.scheduler._to_popen_request(
            app=AppDef(
                name="_",
                roles=[Role(name="foo", image=self.test_dir)],
            ),
            cfg={},
        )

        # the test scheduler is hooked up with LocalDirectoryImageProvider
        # which treats the role's image as CWD
        replica_params = popen_request.role_params["foo"][0]
        self.assertEqual(f"{self.test_dir}", replica_params.env["PATH"])

        # _popen actually adds in the os.environ[PATH]
        self.scheduler._popen(
            role_name="foo", replica_id=0, replica_params=replica_params
        )

        self.assertEqual(
            # for python 3.7 BC get call_args.kwargs by index
            f"{self.test_dir}",
            popen_mock.call_args[1]["env"]["PATH"],
        )

    @mock.patch.dict(os.environ, {"FOO": "bar"})
    def test_submit_override_parent_env(self) -> None:
        role = Role(
            "echo_foo",
            image=self.test_dir,
            entrypoint="echo_env_foo.sh",
            env={"FOO": "new_bar"},
        )
        app = AppDef(name="check_foo_env_var", roles=[role])
        app_id = self.scheduler.submit(app, {"log_dir": self.test_dir})
        for line in self.scheduler.log_iter(app_id, "echo_foo"):
            self.assertEqual("new_bar\n", line)

        desc = self.wait(app_id, self.scheduler)
        assert desc is not None
        self.assertEqual(AppState.SUCCEEDED, desc.state)

    def _assert_file_content(self, filename: str, expected: str) -> None:
        with open(filename, "r") as f:
            self.assertEqual(expected, f.read())

    def test_submit_with_log_dir_stdout(self) -> None:
        num_replicas = 2

        for std_stream in ["stdout", "stderr"]:
            with self.subTest(std_stream=std_stream):
                log_dir = join(self.test_dir, f"test_{std_stream}_log")
                cfg = LocalOpts({"log_dir": log_dir})

                role = Role(
                    "role1",
                    image=self.test_dir,
                    entrypoint=f"echo_{std_stream}.sh",
                    args=["hello_world"],
                    num_replicas=num_replicas,
                )
                app = AppDef(name="test_app", roles=[role])

                app_id = self.scheduler.submit(app, cfg)
                self.wait(app_id)

                success_file = join(
                    log_dir, self.scheduler.session_name, app_id, "SUCCESS"
                )
                with open(success_file, "r") as f:
                    sf_json = json.load(f)
                    self.assertEqual(app_id, sf_json["app_id"])
                    self.assertEqual(
                        join(log_dir, self.scheduler.session_name, app_id),
                        sf_json["log_dir"],
                    )
                    self.assertEqual(AppState.SUCCEEDED.name, sf_json["final_state"])

                    for replica_id in range(num_replicas):
                        replica_info = sf_json["roles"]["role1"][replica_id]
                        self._assert_file_content(
                            replica_info[std_stream], "hello_world\n"
                        )

    @patch(LOCAL_DIR_IMAGE_PROVIDER_FETCH, return_value="")
    def test_submit_dryrun_without_log_dir_cfg(self, _) -> None:
        master = Role(
            "master",
            image=self.test_dir,
            entrypoint="master.par",
            args=["arg1"],
            env={"ENV_VAR_1": "VAL1"},
        )
        trainer = Role(
            "trainer", image=self.test_dir, entrypoint="trainer.par", num_replicas=2
        )

        app = AppDef(name="test_app", roles=[master, trainer])
        info = self.scheduler.submit_dryrun(app, cfg={})
        # intentional print (to make sure it actually prints with no errors)
        print(info)

        request = info.request
        role_params = request.role_params
        role_log_dirs = request.role_log_dirs
        self.assertEqual(2, len(role_params))
        self.assertEqual(2, len(role_log_dirs))

        master_params = role_params["master"]
        trainer_params = role_params["trainer"]

        app_log_dir = request.log_dir

        self.assertEqual(1, len(master_params))
        self.assertEqual(2, len(trainer_params))

        for role in app.roles:
            replica_params = role_params[role.name]
            replica_log_dirs = role_log_dirs[role.name]

            for j in range(role.num_replicas):
                replica_param = replica_params[j]
                replica_log_dir = replica_log_dirs[j]

                # dryrun should NOT create any directories
                self.assertFalse(os.path.isdir(replica_log_dir))
                self.assertTrue(replica_log_dir.startswith(app_log_dir))
                self.assertEqual([role.entrypoint, *role.args], replica_param.args)
                self.assertEqual(
                    {
                        "TORCHX_RANK0_HOST": "localhost",
                        "TORCHELASTIC_ERROR_FILE": join(replica_log_dir, "error.json"),
                        "PET_LOG_DIR": join(app_log_dir, "torchelastic", role.name),
                        **role.env,
                    },
                    replica_param.env,
                )
                self.assertIsNotNone(replica_param.stdout)
                self.assertIsNotNone(replica_param.stderr)
                self.assertIsNotNone(replica_param.combined)

    @patch(LOCAL_DIR_IMAGE_PROVIDER_FETCH, return_value="")
    def test_submit_dryrun_with_log_dir_cfg(
        self, img_provider_fetch_mock: mock.Mock
    ) -> None:
        trainer = Role(
            "trainer", image=self.test_dir, entrypoint="trainer.par", num_replicas=2
        )

        app = AppDef(name="test_app", roles=[trainer])
        cfg = {"log_dir": self.test_dir}
        info = self.scheduler.submit_dryrun(app, cfg)
        # intentional print (to make sure it actually prints with no errors)
        print(info)

        request = info.request
        role_params = request.role_params
        role_log_dirs = request.role_log_dirs
        self.assertEqual(1, len(role_params))
        self.assertEqual(1, len(role_log_dirs))

        trainer_params = role_params["trainer"]

        app_log_dir = request.log_dir

        self.assertEqual(2, len(trainer_params))

        for role in app.roles:
            replica_params = role_params[role.name]
            replica_log_dirs = role_log_dirs[role.name]

            for j in range(role.num_replicas):
                replica_param = replica_params[j]
                replica_log_dir = replica_log_dirs[j]
                # dryrun should NOT create any directories
                self.assertFalse(os.path.isdir(replica_log_dir))
                self.assertTrue(replica_log_dir.startswith(app_log_dir))
                self.assertEqual([role.entrypoint, *role.args], replica_param.args)
                self.assertEqual(
                    {
                        "TORCHX_RANK0_HOST": "localhost",
                        "TORCHELASTIC_ERROR_FILE": join(replica_log_dir, "error.json"),
                        "PET_LOG_DIR": join(app_log_dir, "torchelastic", role.name),
                        **role.env,
                    },
                    replica_param.env,
                )
                stdout_path = join(replica_log_dir, "stdout.log")
                stderr_path = join(replica_log_dir, "stderr.log")
                self.assertEqual(stdout_path, replica_param.stdout)
                self.assertEqual(stderr_path, replica_param.stderr)

    def test_torchelastic_env_varsn(self) -> None:
        """
        Checks torchelastic (torch.distributed.run) env vars that are respected if already set
        in the role.env settings, and that they are auto set if not explicitly set in role.env
        """

        env = {"TORCHELASTIC_ERROR_FILE": "foobar.json", "PET_LOG_DIR": "/foo/bar"}
        trainer = Role(
            "trainer", image=self.test_dir, entrypoint="trainer_main.py", env=env
        )
        reader = Role("reader", image=self.test_dir, entrypoint="reader_main.py")
        app = AppDef(name="test_app", roles=[trainer, reader])
        cfg = {"log_dir": self.test_dir}
        info = self.scheduler.submit_dryrun(app, cfg)
        request = info.request

        app_log_dir = request.log_dir

        # only 1 replica per trainer and reader, so use 0 index
        trainer_replica_param = request.role_params["trainer"][0]
        reader_replica_param = request.role_params["reader"][0]
        reader_replica_log_dir = request.role_log_dirs["reader"][0]

        # trainer set torchelastic env vars in the role settings, must repect those
        self.assertEqual(
            "foobar.json", trainer_replica_param.env["TORCHELASTIC_ERROR_FILE"]
        )
        self.assertEqual("/foo/bar", trainer_replica_param.env["PET_LOG_DIR"])

        # reader didn't set torchelastic env vars in the role settings, these should be autoset
        self.assertEqual(
            join(reader_replica_log_dir, "error.json"),
            reader_replica_param.env["TORCHELASTIC_ERROR_FILE"],
        )
        self.assertEqual(
            join(app_log_dir, "torchelastic", "reader"),
            reader_replica_param.env["PET_LOG_DIR"],
        )

    def test_log_iterator(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="echo_range.sh",
            args=["10", "0.5"],
            num_replicas=1,
        )

        log_dir = join(self.test_dir, "log")
        cfg = LocalOpts({"log_dir": log_dir})
        app = AppDef(name="test_app", roles=[role])
        app_id = self.scheduler.submit(app, cfg)

        for i, line in enumerate(self.scheduler.log_iter(app_id, "role1", k=0)):
            self.assertEqual(str(i), line.strip())

        # since and until ignored
        for i, line in enumerate(
            self.scheduler.log_iter(
                app_id, "role1", k=0, since=datetime.now(), until=datetime.now()
            )
        ):
            self.assertEqual(str(i), line.strip())

        for i, line in enumerate(
            self.scheduler.log_iter(app_id, "role1", k=0, regex=r"[02468]")
        ):
            self.assertEqual(str(i * 2), line.strip())

    def test_log_iterator_no_log_dir(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="echo_range.sh",
            args=["10", "0.5"],
            num_replicas=1,
        )

        app = AppDef(name="test_app", roles=[role])

        app_id = self.scheduler.submit(app, cfg={})
        logs = list(self.scheduler.log_iter(app_id, "role1", k=0))
        self.assertEqual(len(logs), 11)

    def test_log_iterator_invalid_utf8(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="print_invalid_utf8.sh",
            num_replicas=1,
        )

        app = AppDef(name="test_app", roles=[role])

        app_id = self.scheduler.submit(app, cfg={})
        logs = list(self.scheduler.log_iter(app_id, "role1", k=0))
        self.assertEqual(logs, ["\x02\ufffd\ufffd"])

    def test_submit_multiple_roles(self) -> None:
        test_file1 = join(self.test_dir, "test_file_1")
        test_file2 = join(self.test_dir, "test_file_2")
        role1 = Role(
            "role1",
            image=self.test_dir,
            entrypoint="touch.sh",
            args=[test_file1],
            num_replicas=1,
        )
        role2 = Role(
            "role2",
            image=self.test_dir,
            entrypoint="touch.sh",
            args=[test_file2],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role1, role2])
        cfg = {"log_dir": self.test_dir}
        app_id = self.scheduler.submit(app, cfg)

        desc = self.wait(app_id)
        assert desc is not None
        self.assertEqual(AppState.SUCCEEDED, desc.state)
        self.assertTrue(os.path.isfile(test_file1))
        self.assertTrue(os.path.isfile(test_file2))

    def test_describe(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="sleep.sh",
            args=["2"],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role])
        cfg = {"log_dir": self.test_dir}
        self.assertIsNone(self.scheduler.describe("test_app_0"))
        app_id = self.scheduler.submit(app, cfg)
        desc = self.scheduler.describe(app_id)
        assert desc is not None
        self.assertEqual(AppState.RUNNING, desc.state)
        desc_resp = self.wait(app_id)
        assert desc_resp is not None
        self.assertEqual(AppState.SUCCEEDED, desc_resp.state)

    def test_cancel(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="sleep.sh",
            args=["10"],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role])
        cfg = {"log_dir": self.test_dir}
        app_id = self.scheduler.submit(app, cfg)
        desc = self.scheduler.describe(app_id)
        assert desc is not None
        self.assertEqual(AppState.RUNNING, desc.state)
        self.scheduler.cancel(app_id)
        desc = self.scheduler.describe(app_id)
        assert desc is not None
        self.assertEqual(AppState.CANCELLED, desc.state)

    def test_list(self) -> None:
        with self.assertRaisesRegex(Exception, "cannot be listed"):
            self.scheduler.list()

    def test_exists(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="sleep.sh",
            args=["10"],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role])
        cfg = {"log_dir": self.test_dir}
        app_id = self.scheduler.submit(app, cfg)

        self.assertTrue(self.scheduler.exists(app_id))
        self.scheduler.cancel(app_id)
        self.assertTrue(self.scheduler.exists(app_id))

    def test_invalid_cache_size(self) -> None:
        with self.assertRaises(ValueError):
            LocalScheduler(
                session_name="test_session",
                cache_size=0,
                image_provider_class=LocalDirectoryImageProvider,
            )

        with self.assertRaises(ValueError):
            LocalScheduler(
                session_name="test_session",
                cache_size=-1,
                image_provider_class=LocalDirectoryImageProvider,
            )

    def test_cache_full(self) -> None:
        scheduler = LocalScheduler(
            session_name="test_session",
            cache_size=1,
            image_provider_class=LocalDirectoryImageProvider,
        )
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="sleep.sh",
            args=["10"],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role])
        cfg = {"log_dir": self.test_dir}
        app_id = scheduler.submit(app, cfg)
        with self.assertRaises(IndexError):
            scheduler.submit(app, cfg)
        scheduler.close()

    def test_cache_evict(self) -> None:
        scheduler = LocalScheduler(
            session_name="test_session",
            cache_size=1,
            image_provider_class=LocalDirectoryImageProvider,
        )
        test_file1 = join(self.test_dir, "test_file_1")
        test_file2 = join(self.test_dir, "test_file_2")

        role1 = Role(
            "role1", image=self.test_dir, entrypoint="touch.sh", args=[test_file1]
        )
        role2 = Role(
            "role2", image=self.test_dir, entrypoint="touch.sh", args=[test_file2]
        )
        app1 = AppDef(name="touch_test_file1", roles=[role1])
        app2 = AppDef(name="touch_test_file2", roles=[role2])
        cfg = {"log_dir": self.test_dir}

        app_id1 = scheduler.submit(app1, cfg)
        resp1 = self.wait(app_id1, scheduler)
        assert resp1 is not None
        self.assertEqual(AppState.SUCCEEDED, resp1.state)

        app_id2 = scheduler.submit(app2, cfg)
        resp2 = self.wait(app_id2, scheduler)
        assert resp2 is not None
        self.assertEqual(AppState.SUCCEEDED, resp2.state)

        # app1 should've been evicted
        self.assertIsNone(scheduler.describe(app_id1))
        self.assertIsNone(self.wait(app_id1, scheduler))

        self.assertIsNotNone(scheduler.describe(app_id2))
        self.assertIsNotNone(self.wait(app_id2, scheduler))

    def test_close(self) -> None:
        # 2 apps each with 4 replicas == 8 total pids
        # make sure they all exist after submission
        # validate they do not exist after calling the close() method
        sleep_60sec = AppDef(
            name="sleep",
            roles=[
                Role(
                    name="sleep",
                    image=self.test_dir,
                    entrypoint="sleep.sh",
                    args=["60"],
                    num_replicas=4,
                )
            ],
        )

        self.scheduler.submit(sleep_60sec, cfg={})
        self.scheduler.submit(sleep_60sec, cfg={})

        pids = []
        for app_id, app in self.scheduler._apps.items():
            for role_name, replicas in app.role_replicas.items():
                for replica in replicas:
                    pid = replica.proc.pid
                    self.assertTrue(pid_exists(pid))
                    pids.append(pid)

        self.scheduler.close()

        for pid in pids:
            self.assertFalse(pid_exists(pid))

    def test_close_twice(self) -> None:
        sleep_60sec = AppDef(
            name="sleep",
            roles=[
                Role(
                    name="sleep",
                    image=self.test_dir,
                    entrypoint="sleep.sh",
                    args=["60"],
                    num_replicas=4,
                )
            ],
        )

        self.scheduler.submit(sleep_60sec, cfg={})
        self.scheduler.close()
        self.scheduler.close()
        # nothing to validate just make sure no errors are raised

    def test_get_gpu_device_count(self) -> None:
        @dataclass
        class ProcResult:
            stdout: str

        nvidia_smi_out = (
            "GPU 0: Tesla V100-SXM2-16GB (UUID: GPU-196a22c5-717b-66db-0acc-58cde6f3df85)\n"
            "GPU 1: Tesla V100-SXM2-16GB (UUID: GPU-45e9165d-4f7e-d954-7ff5-481bc2c0ec7b)\n"
            "GPU 2: Tesla V100-SXM2-16GB (UUID: GPU-26e22503-5fd5-8f55-d068-e1714fbb6fd6)\n"
            "GPU 3: Tesla V100-SXM2-16GB (UUID: GPU-ebfc20c7-5f1a-1bc9-0d98-601cbe21fc2d)\n"
        )

        stdout = nvidia_smi_out
        result = ProcResult(stdout)
        with patch("subprocess.run", return_value=result):
            gpu_count = self.scheduler._cuda_device_count()
            self.assertEqual(4, gpu_count)

    def test_get_gpu_device_count_error(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=2,
            cmd="",
            output="",
            stderr="",
        )
        with patch("subprocess.run", side_effect=error):
            gpu_count = self.scheduler._cuda_device_count()
            self.assertEqual(0, gpu_count)

    def role(self, name: str, replicas: int, gpu: int = 0) -> Role:
        return Role(
            name=name,
            image=self.test_dir,
            entrypoint="sleep.sh",
            args=["1"],
            num_replicas=replicas,
            resource=Resource(cpu=1, gpu=gpu, memMB=1024),
        )

    def dryrun(
        self, *roles: Role, auto_set_cuda_visible_devices: bool = True
    ) -> AppDryRunInfo[PopenRequest]:
        return self.scheduler.submit_dryrun(
            AppDef(name="_ignored_", roles=list(roles)),
            cfg={"auto_set_cuda_visible_devices": auto_set_cuda_visible_devices},
        )

    def assert_CUDA_VISIBLE_DEVICES(
        self, dryrun_info: AppDryRunInfo[PopenRequest], expected: dict[str, list[str]]
    ) -> None:
        for role_name, replica_params in dryrun_info.request.role_params.items():
            if role_name in expected:
                actual_cuda_visible_devices = [
                    rp.env[ENV_CUDA_VISIBLE_DEVICES] for rp in replica_params
                ]
                self.assertEqual(expected[role_name], actual_cuda_visible_devices)
            else:  # make sure we didn't set CUDA_VISIBLE_DEVICES for non-gpu roles
                for rp in replica_params:
                    self.assertFalse(ENV_CUDA_VISIBLE_DEVICES in rp.env)

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=0)
    def test_auto_set_CUDA_VISIBLE_DEVICES_no_gpus_on_host(self, _: MagicMock) -> None:
        self.assert_CUDA_VISIBLE_DEVICES(
            self.dryrun(self.role("a", replicas=2, gpu=0)),
            {},
        )
        self.assert_CUDA_VISIBLE_DEVICES(
            self.dryrun(self.role("a", replicas=2, gpu=1)),
            {},
        )

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=8)
    def test_auto_set_CUDA_VISIBLE_DEVICES_no_requested_gpus(
        self, _: MagicMock
    ) -> None:
        self.assert_CUDA_VISIBLE_DEVICES(
            self.dryrun(self.role("a", replicas=2, gpu=0)),
            {},
        )

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=2)
    def test_auto_set_CUDA_VISIBLE_DEVICES_more_gpus_then_available(
        self, _: MagicMock
    ) -> None:
        self.assert_CUDA_VISIBLE_DEVICES(
            self.dryrun(self.role("a", replicas=2, gpu=2)),
            {},
        )

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=2)
    def test_auto_set_CUDA_VISIBLE_DEVICES_false(self, _: MagicMock) -> None:
        self.assert_CUDA_VISIBLE_DEVICES(
            self.dryrun(
                self.role("a", replicas=2, gpu=1), auto_set_cuda_visible_devices=False
            ),
            {},
        )

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=8)
    def test_auto_set_CUDA_VISIBLE_DEVICES(self, _: MagicMock) -> None:
        for gpu in [1, 2, 3, 4, 5, 6, 7, 8]:
            expected = [
                [],
                ["0"],
                ["0,1"],
                ["0,1,2"],
                ["0,1,2,3"],
                ["0,1,2,3,4"],
                ["0,1,2,3,4,5"],
                ["0,1,2,3,4,5,6"],
                ["0,1,2,3,4,5,6,7"],
            ]
            self.assert_CUDA_VISIBLE_DEVICES(
                self.dryrun(self.role(name="a", replicas=1, gpu=gpu)),
                {"a": expected[gpu]},
            )

        for gpu in [1, 2, 3, 4]:
            expected = [
                [],
                ["0", "1"],
                ["0,1", "2,3"],
                ["0,1,2", "3,4,5"],
                ["0,1,2,3", "4,5,6,7"],
            ]
            self.assert_CUDA_VISIBLE_DEVICES(
                self.dryrun(self.role(name="a", replicas=2, gpu=gpu)),
                {"a": expected[gpu]},
            )

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=8)
    def test_auto_set_CUDA_VISIBLE_DEVICES_multi_role(self, _: MagicMock) -> None:
        self.assert_CUDA_VISIBLE_DEVICES(
            self.dryrun(
                self.role(name="a", replicas=2, gpu=1),
                self.role(name="b", replicas=1, gpu=0),
                self.role(name="c", replicas=3, gpu=2),
            ),
            {
                "a": ["0", "1"],
                "c": ["2,3", "4,5", "6,7"],
            },
        )

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=16)
    def test_get_cuda_devices_is_set(self, _: MagicMock) -> None:
        appdef = AppDef(
            name="role1",
            roles=[
                Role(
                    name="role1",
                    image=self.test_dir,
                    entrypoint="train",
                    resource=Resource(gpu=2, cpu=0, memMB=0),
                    num_replicas=2,
                ),
                Role(
                    name="role2",
                    image=self.test_dir,
                    entrypoint="train",
                    resource=Resource(gpu=3, cpu=0, memMB=0),
                    num_replicas=2,
                ),
            ],
        )
        popen_req = self.scheduler._to_popen_request(
            appdef, {"auto_set_cuda_visible_devices": True}
        )
        role1_params = popen_req.role_params["role1"]
        self.assertEqual(2, len(role1_params))
        self.assertEqual("0,1", role1_params[0].env[ENV_CUDA_VISIBLE_DEVICES])
        self.assertEqual("2,3", role1_params[1].env[ENV_CUDA_VISIBLE_DEVICES])
        role2_params = popen_req.role_params["role2"]
        self.assertEqual(2, len(role2_params))
        self.assertEqual("4,5,6", role2_params[0].env[ENV_CUDA_VISIBLE_DEVICES])
        self.assertEqual("7,8,9", role2_params[1].env[ENV_CUDA_VISIBLE_DEVICES])

    @patch(LOCAL_SCHED_DEVICE_COUNT, return_value=8)
    def test_get_cuda_devices_not_set(self, _: MagicMock) -> None:
        trainer1 = AppDef(
            name="trainer1",
            roles=[
                Role(
                    name="trainer1",
                    image=self.test_dir,
                    entrypoint="trainer1.sh",
                    resource=Resource(gpu=4, cpu=0, memMB=0),
                    num_replicas=4,
                )
            ],
        )

        popen_req = self.scheduler._to_popen_request(trainer1, {})
        role_params = popen_req.role_params["trainer1"]
        self.assertEqual(4, len(role_params))
        self.assertFalse(ENV_CUDA_VISIBLE_DEVICES in role_params[0].env)
        self.assertFalse(ENV_CUDA_VISIBLE_DEVICES in role_params[1].env)
        self.assertFalse(ENV_CUDA_VISIBLE_DEVICES in role_params[2].env)
        self.assertFalse(ENV_CUDA_VISIBLE_DEVICES in role_params[3].env)

    def test_no_orphan_process_function(self) -> None:
        self._test_orphan_workflow()

    def _test_orphan_workflow(self) -> None:
        mp_queue = mp.Queue()
        child_nproc = 2

        proc = mp.Process(
            target=start_sleep_processes, args=(self.test_dir, mp_queue, child_nproc)
        )
        proc.start()
        total_processes = child_nproc + 1
        pids = []
        for _ in range(total_processes):
            pids.append(mp_queue.get(timeout=5))
        parent_pid = pids[0]
        child_pids = pids[1:]

        os.kill(parent_pid, signal.SIGTERM)
        # Wait to give time for signal handlers to finish work
        time.sleep(5)
        for child_pid in child_pids:
            # Killing parent should kill all children, we expect that each call to
            # os.kill would raise OSError
            with self.assertRaises(OSError):
                os.kill(child_pid, 0)


class JoinPATHTest(unittest.TestCase):
    def test_join_PATH(self) -> None:
        self.assertEqual("", _join_PATH(None))
        self.assertEqual("", _join_PATH(""))
        self.assertEqual("", _join_PATH("", None))
        self.assertEqual("/usr/local/bin", _join_PATH("/usr/local/bin", ""))
        self.assertEqual("/usr/local/bin", _join_PATH("/usr/local/bin", None))
        self.assertEqual("/usr/local/bin", _join_PATH("", "/usr/local/bin"))
        self.assertEqual("/usr/local/bin", _join_PATH(None, "/usr/local/bin"))

        path = ":/usr/bin:/bin:"
        self.assertEqual(
            "/usr/local/bin:/usr/bin:/bin", _join_PATH("/usr/local/bin", path)
        )
        self.assertEqual(
            "/usr/bin:/bin:/usr/local/bin", _join_PATH(path, "/usr/local/bin")
        )


@contextmanager
def _temp_setenv(k: str, v: str) -> Generator[None, None, None]:
    old = os.environ[k]
    os.environ[k] = v
    try:
        yield
    finally:
        os.environ[k] = old
