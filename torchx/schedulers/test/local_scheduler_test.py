#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from datetime import datetime
from os.path import join
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock, call, patch

from pyre_extensions import none_throws
from torchx.components.base.binary_component import binary_component
from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.local_scheduler import (
    ReplicaParam,
    DockerImageProvider,
    LocalDirectoryImageProvider,
    LocalScheduler,
    make_unique,
)
from torchx.specs.api import AppDef, AppState, Role, RunConfig, is_terminal, macros

from .test_util import write_shell_script


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
    cfg = RunConfig({"log_dir": test_dir})

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
        cfg = RunConfig()
        provider = LocalDirectoryImageProvider(cfg)
        self.assertEqual(self.test_dir, provider.fetch(self.test_dir))

    def test_fetch_relative_path_should_throw(self) -> None:
        cfg = RunConfig()
        provider = LocalDirectoryImageProvider(cfg)
        with self.assertRaises(ValueError):
            provider.fetch(self.test_dir_name)

    def test_fetch_does_not_exist_should_throw(self) -> None:
        non_existent_dir = join(self.test_dir, "non_existent_dir")
        cfg = RunConfig()
        provider = LocalDirectoryImageProvider(cfg)
        with self.assertRaises(ValueError):
            provider.fetch(non_existent_dir)

    def test_get_command(self) -> None:
        cfg = RunConfig()
        provider = LocalDirectoryImageProvider(cfg)
        args = ["a", "b"]
        self.assertEqual(provider.get_command("image", args, {}), args)


class DockerImageProviderTest(unittest.TestCase):
    @patch("subprocess.run")
    def test_fetch(self, run: MagicMock) -> None:
        cfg = RunConfig()
        provider = DockerImageProvider(cfg)
        img = "pytorch/pytorch:latest"
        self.assertEqual(provider.fetch(img), "")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args, call(["docker", "pull", img], check=True))

    def test_get_command(self) -> None:
        cfg = RunConfig()
        provider = DockerImageProvider(cfg)
        cmd = provider.get_command(
            "pytorch/pytorch:latest",
            ["main.py", "--a", "b"],
            {
                "FOO": "bar",
                "FOO2": "bar3",
            },
        )
        self.assertEqual(
            cmd,
            [
                "docker",
                "run",
                "-i",
                "--rm",
                "--env",
                "FOO=bar",
                "--env",
                "FOO2=bar3",
                "pytorch/pytorch:latest",
                "main.py",
                "--a",
                "b",
            ],
        )


LOCAL_SCHEDULER_MAKE_UNIQUE = "torchx.schedulers.local_scheduler.make_unique"

ERR_FILE_ENV = "TORCHELASTIC_ERROR_FILE"


class LocalSchedulerTestUtil(abc.ABC):
    scheduler: LocalScheduler

    def wait(
        self,
        app_id: str,
        scheduler: Optional[LocalScheduler] = None,
        timeout: float = 30,
    ) -> Optional[DescribeAppResponse]:
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
            elif is_terminal(desc.state):
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

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

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
        with patch(LOCAL_SCHEDULER_MAKE_UNIQUE, return_value=expected_app_id):
            cfg = RunConfig({"log_dir": self.test_dir})
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
            cfg = RunConfig({"log_dir": self.test_dir})
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
        app_id = self.scheduler.submit(app, RunConfig({"log_dir": self.test_dir}))
        for line in self.scheduler.log_iter(app_id, "echo_foo"):
            self.assertEqual("bar", line)

        desc = self.wait(app_id, self.scheduler)
        assert desc is not None
        self.assertEqual(AppState.SUCCEEDED, desc.state)

    def test_child_process_env(self) -> None:
        dir_path = "/tmp/users"
        replica_params = ReplicaParam([], {}, None, None, dir_path)
        child_env = self.scheduler._get_child_process_env(replica_params)
        path = child_env["PATH"]
        self.assertTrue(dir_path in path)

    def test_child_process_env_none(self) -> None:
        replica_params = ReplicaParam([], {}, None, None, None)
        parent_proc_path = os.environ["PATH"]
        child_env = self.scheduler._get_child_process_env(replica_params)
        child_proc_path = child_env["PATH"]
        self.assertEqual(parent_proc_path, child_proc_path)

    @mock.patch.dict(os.environ, {"FOO": "bar"})
    def test_submit_override_parent_env(self) -> None:
        role = Role(
            "echo_foo",
            image=self.test_dir,
            entrypoint="echo_env_foo.sh",
            env={"FOO": "new_bar"},
        )
        app = AppDef(name="check_foo_env_var", roles=[role])
        app_id = self.scheduler.submit(app, RunConfig({"log_dir": self.test_dir}))
        for line in self.scheduler.log_iter(app_id, "echo_foo"):
            self.assertEqual("new_bar", line)

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
                cfg = RunConfig({"log_dir": log_dir})

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
    def test_submit_dryrun_without_log_dir_cfg(
        self, img_provider_fetch_mock: mock.Mock
    ) -> None:
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
        cfg = RunConfig()
        info = self.scheduler.submit_dryrun(app, cfg)
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
                        ERR_FILE_ENV: join(replica_log_dir, "error.json"),
                        **role.env,
                    },
                    replica_param.env,
                )
                self.assertIsNone(replica_param.stdout)
                self.assertIsNone(replica_param.stderr)

    @patch(LOCAL_DIR_IMAGE_PROVIDER_FETCH, return_value="")
    def test_submit_dryrun_with_log_dir_cfg(
        self, img_provider_fetch_mock: mock.Mock
    ) -> None:
        trainer = Role(
            "trainer", image=self.test_dir, entrypoint="trainer.par", num_replicas=2
        )

        app = AppDef(name="test_app", roles=[trainer])
        cfg = RunConfig({"log_dir": self.test_dir})
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
                        ERR_FILE_ENV: join(replica_log_dir, "error.json"),
                        **role.env,
                    },
                    replica_param.env,
                )
                stdout_path = join(replica_log_dir, "stdout.log")
                stderr_path = join(replica_log_dir, "stderr.log")
                self.assertEqual(stdout_path, replica_param.stdout)
                self.assertEqual(stderr_path, replica_param.stderr)

    def test_log_iterator(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="echo_range.sh",
            args=["10", "0.5"],
            num_replicas=1,
        )

        log_dir = join(self.test_dir, "log")
        cfg = RunConfig({"log_dir": log_dir})
        app = AppDef(name="test_app", roles=[role])
        app_id = self.scheduler.submit(app, cfg)

        for i, line in enumerate(self.scheduler.log_iter(app_id, "role1", k=0)):
            self.assertEqual(str(i), line)

        # since and until ignored
        for i, line in enumerate(
            self.scheduler.log_iter(
                app_id, "role1", k=0, since=datetime.now(), until=datetime.now()
            )
        ):
            self.assertEqual(str(i), line)

        for i, line in enumerate(
            self.scheduler.log_iter(app_id, "role1", k=0, regex=r"[02468]")
        ):
            self.assertEqual(str(i * 2), line)

    def test_log_iterator_no_log_dir(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="echo_range.sh",
            args=["10", "0.5"],
            num_replicas=1,
        )

        app = AppDef(name="test_app", roles=[role])

        with self.assertRaises(RuntimeError, msg="log_dir must be set to iterate logs"):
            app_id = self.scheduler.submit(app, RunConfig())
            self.scheduler.log_iter(app_id, "role1", k=0)

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
        cfg = RunConfig({"log_dir": self.test_dir})
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
        cfg = RunConfig({"log_dir": self.test_dir})
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
        cfg = RunConfig({"log_dir": self.test_dir})
        app_id = self.scheduler.submit(app, cfg)
        desc = self.scheduler.describe(app_id)
        assert desc is not None
        self.assertEqual(AppState.RUNNING, desc.state)
        self.scheduler.cancel(app_id)
        desc = self.scheduler.describe(app_id)
        assert desc is not None
        self.assertEqual(AppState.CANCELLED, desc.state)

    def test_exists(self) -> None:
        role = Role(
            "role1",
            image=self.test_dir,
            entrypoint="sleep.sh",
            args=["10"],
            num_replicas=1,
        )
        app = AppDef(name="test_app", roles=[role])
        cfg = RunConfig({"log_dir": self.test_dir})
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
        cfg = RunConfig({"log_dir": self.test_dir})
        scheduler.submit(app, cfg)
        with self.assertRaises(IndexError):
            scheduler.submit(app, cfg)

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
        cfg = RunConfig({"log_dir": self.test_dir})

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

        self.scheduler.submit(sleep_60sec, RunConfig())
        self.scheduler.submit(sleep_60sec, RunConfig())

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

        self.scheduler.submit(sleep_60sec, RunConfig())
        self.scheduler.close()
        self.scheduler.close()
        # nothing to validate just make sure no errors are raised

    def test_dryrun_base_image_should_throw(self) -> None:
        # base_image for vanilla local_scheduler shouldn't work
        app = AppDef(
            name="base_image_test_app",
            roles=[
                Role(
                    name="base_image_test_role",
                    image=self.test_dir,
                    base_image=self.test_dir,
                    entrypoint="touch.sh",
                    args=[join(self.test_dir, "testfile1")],
                ),
            ],
        )

        with self.assertRaises(NotImplementedError):
            self.scheduler.submit_dryrun(app, RunConfig())

    def test_no_orphan_process_function(self) -> None:
        self._test_orphan_workflow(signal.SIGTERM)

    def _test_orphan_workflow(self, signal_to_send: signal.Signals) -> None:
        mp_queue = mp.get_context("spawn").Queue()
        child_nproc = 2

        proc = mp.get_context("spawn").Process(
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


def _has_docker() -> bool:
    try:
        return subprocess.run(["docker", "version"]).returncode == 0
    except FileNotFoundError:
        return False


if _has_docker():

    class LocalDockerSchedulerTest(unittest.TestCase, LocalSchedulerTestUtil):
        def setUp(self) -> None:
            self.scheduler = LocalScheduler(
                session_name="test_session",
                image_provider_class=DockerImageProvider,
            )

        def _docker_app(self, entrypoint: str, *args: str) -> AppDef:
            return binary_component(
                name="test-app",
                image="busybox",
                entrypoint=entrypoint,
                args=list(args),
            )

        def test_docker_submit(self) -> None:
            app = self._docker_app("echo", "foo")
            cfg = RunConfig({"image_type": "docker"})
            app_id = self.scheduler.submit(app, cfg)

            desc = self.wait(app_id)
            assert desc is not None
            self.assertEqual(AppState.SUCCEEDED, desc.state)

        def test_docker_submit_error(self) -> None:
            app = self._docker_app("sh", "-c", "exit 1")
            cfg = RunConfig({"image_type": "docker"})
            app_id = self.scheduler.submit(app, cfg)

            desc = self.wait(app_id)
            assert desc is not None
            self.assertEqual(AppState.FAILED, desc.state)
