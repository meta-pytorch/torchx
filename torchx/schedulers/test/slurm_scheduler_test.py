# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import datetime
import importlib
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from typing import Generator
from unittest.mock import call, MagicMock, patch

import torchx
from torchx import specs
from torchx.schedulers.api import DescribeAppResponse, ListAppResponse, Stream
from torchx.schedulers.slurm_scheduler import (
    _get_job_dirs,
    _save_job_dir,
    create_scheduler,
    SlurmBatchRequest,
    SlurmOpts,
    SlurmReplicaRequest,
    SlurmScheduler,
)
from torchx.specs import AppState

DESCRIBE_SQUEUE = "torchx.schedulers.slurm_scheduler.SlurmScheduler._describe_squeue"
DESCRIBE_SACCT = "torchx.schedulers.slurm_scheduler.SlurmScheduler._describe_sacct"

CALLED_PROCESS_ERROR = subprocess.CalledProcessError(
    returncode=1,
    cmd="__ignored__",
    stderr="slurm_load_jobs error: Invalid job id specified",
)


@contextmanager
def tmp_cwd() -> Generator[None, None, None]:
    with tempfile.TemporaryDirectory() as path:
        cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(cwd)


def simple_role() -> specs.Role:
    return specs.Role(
        name="foo",
        image="/some/path",
        entrypoint="echo",
        args=["hello slurm", "test"],
        env={
            "FOO": "bar",
        },
        num_replicas=5,
        resource=specs.Resource(
            cpu=2,
            memMB=10,
            gpu=3,
        ),
    )


def simple_app() -> specs.AppDef:
    return specs.AppDef(
        name="foo",
        roles=[
            specs.Role(
                name="a",
                image="/some/path",
                entrypoint="echo",
                args=[specs.macros.replica_id, f"hello {specs.macros.app_id}"],
                num_replicas=2,
                max_retries=3,
            ),
            specs.Role(
                name="b",
                image="/some/path",
                entrypoint="echo",
                max_retries=2,
            ),
        ],
    )


def mem_app() -> specs.AppDef:
    return specs.AppDef(
        name="foo",
        roles=[
            simple_role(),
        ],
    )


class SlurmSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, SlurmScheduler)

    def test_replica_request(self) -> None:
        role = simple_role()
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-0", role, cfg={}, nomem=False
        ).materialize()
        self.assertEqual(
            sbatch,
            [
                "--job-name=role-0",
                "--requeue",
                "--ntasks-per-node=1",
                "--cpus-per-task=2",
                "--mem=10",
                "--gpus-per-task=3",
            ],
        )
        self.assertEqual(
            srun,
            [
                '--output=slurm-"$SLURM_JOB_ID"-role-0.out',
                '--error=slurm-"$SLURM_JOB_ID"-role-0.err',
                "--wait=60",
                "--kill-on-bad-exit=1",
                "--export=ALL,FOO=bar",
                "echo",
                "'hello slurm'",
                "test",
            ],
        )

    def test_replica_request_nomem(self) -> None:
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-name",
            simple_role(),
            cfg={},
            nomem=True,
        ).materialize()
        self.assertEqual(
            sbatch,
            [
                "--job-name=role-name",
                "--requeue",
                "--ntasks-per-node=1",
                "--cpus-per-task=2",
                "--gpus-per-task=3",
            ],
        )

    def test_replica_request_constraint(self) -> None:
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-name",
            simple_role(),
            cfg={"constraint": "orange"},
            nomem=False,
        ).materialize()
        self.assertIn(
            "--constraint=orange",
            sbatch,
        )

    def test_replica_request_app_id(self) -> None:
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=[f"hello {specs.macros.app_id}"],
        )
        _, srun = SlurmReplicaRequest.from_role(
            "role-name", role, cfg={}, nomem=False
        ).materialize()
        self.assertIn(
            "echo 'hello '\"$SLURM_JOB_ID\"''",
            " ".join(srun),
        )

    def test_replica_request_run_config(self) -> None:
        scheduler = create_scheduler("foo")
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=["hello"],
        )
        cfg = SlurmOpts(
            {
                "partition": "bubblegum",
                "time": "5:13",
            }
        )

        sbatch, _ = SlurmReplicaRequest.from_role(
            "role-name", role, cfg, nomem=False
        ).materialize()

        run_opts = scheduler.run_opts()

        for k, v in cfg.items():
            self.assertIsNotNone(run_opts.get(k))
            self.assertIn(
                f"--{k}={v}",
                sbatch,
            )

    def test_dryrun_multi_role(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(app, cfg={})
        req = info.request
        self.assertIsInstance(req, SlurmBatchRequest)
        self.assertEqual(req.cmd, ["sbatch", "--parsable"])
        self.assertEqual(
            set(req.replicas.keys()),
            {"a-0", "a-1", "b-0"},
        )

        script = req.materialize()
        print(script)
        self.assertEqual(
            script,
            f"""#!/bin/bash
#
# Generated by TorchX {torchx.__version__}
# Run with: sbatch --parsable
#
#SBATCH --job-name=a-0 --requeue --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --job-name=a-1 --requeue --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --job-name=b-0 --requeue --ntasks-per-node=1

set -evx

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES=2

set +e
srun --output=slurm-"$SLURM_JOB_ID"-a-0.out --error=slurm-"$SLURM_JOB_ID"-a-0.err --wait=60 --kill-on-bad-exit=1 echo 0 'hello '"$SLURM_JOB_ID"'' :\\
     --output=slurm-"$SLURM_JOB_ID"-a-1.out --error=slurm-"$SLURM_JOB_ID"-a-1.err --wait=60 --kill-on-bad-exit=1 echo 1 'hello '"$SLURM_JOB_ID"'' :\\
     --output=slurm-"$SLURM_JOB_ID"-b-0.out --error=slurm-"$SLURM_JOB_ID"-b-0.err --wait=60 --kill-on-bad-exit=1 echo
exitcode=$?
set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${{SLURM_RESTART_COUNT:-0}}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
""",
        )

    @patch(
        "torchx.schedulers.slurm_scheduler.SlurmScheduler._partition_memmb",
        return_value=2048,
    )
    @patch("subprocess.run")
    def test_run_multi_role(self, run: MagicMock, partition_memmb: MagicMock) -> None:
        run.return_value.stdout = b"1234"
        scheduler = create_scheduler("foo")
        app = specs.AppDef(
            name="foo",
            roles=[
                specs.Role(
                    name="a",
                    image="/some/path",
                    entrypoint="echo",
                    args=[specs.macros.replica_id, f"hello {specs.macros.app_id}"],
                    num_replicas=2,
                ),
                specs.Role(
                    name="b",
                    image="/some/path",
                    entrypoint="echo",
                ),
            ],
        )
        app_id = scheduler.submit(app, cfg={})
        self.assertEqual(app_id, "1234")

        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        self.assertEqual(kwargs, {"stdout": subprocess.PIPE, "check": True})
        (args,) = args
        self.assertEqual(
            args[:-1],
            [
                "sbatch",
                "--parsable",
            ],
        )

    @patch("torchx.schedulers.slurm_scheduler.SlurmScheduler.describe")
    @patch("subprocess.run")
    def test_cancel(self, run: MagicMock, describe: MagicMock) -> None:
        describe.return_value = DescribeAppResponse()

        scheduler = create_scheduler("foo")
        scheduler.cancel("1234")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args, call(["scancel", "1234"], check=True))

    @patch(DESCRIBE_SQUEUE, side_effect=CALLED_PROCESS_ERROR)
    @patch("subprocess.check_output")
    def test_describe_sacct_completed(
        self, check_output: MagicMock, _: MagicMock
    ) -> None:
        check_output.return_value = """
JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
1853+0|echo-0|compute||1|COMPLETED|0:0
1853+0.batch|batch|||1|COMPLETED|0:0
1853+0.0|echo|||1|COMPLETED|0:0
1853+1|echo-1|compute||1|COMPLETED|0:0
1853+1.0|echo|||1|COMPLETED|0:0
1853+2|echo-2|compute||1|COMPLETED|0:0
1853+2.0|echo|||1|COMPLETED|0:0
""".strip()

        scheduler = create_scheduler("foo")
        out = scheduler.describe(app_id="1853")

        self.assertEqual(check_output.call_count, 1)
        self.assertEqual(
            check_output.call_args,
            call(
                ["sacct", "--parsable2", "-j", "1853"],
                stderr=subprocess.PIPE,
                encoding="utf-8",
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "1853")
        self.assertEqual(out.msg, "COMPLETED")
        self.assertEqual(out.state, specs.AppState.SUCCEEDED)
        self.assertEqual(
            out.roles,
            [
                specs.Role(
                    name="echo",
                    image="",
                    num_replicas=3,
                )
            ],
        )

    @patch(DESCRIBE_SQUEUE, side_effect=CALLED_PROCESS_ERROR)
    @patch("subprocess.check_output")
    def test_describe_sacct_single_replica(
        self, check_output: MagicMock, _: MagicMock
    ) -> None:
        check_output.return_value = """
JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
1902|sh-0|compute||1|FAILED|2:0
1902.batch|batch|||1|FAILED|2:0
1902.0|sh|||1|FAILED|2:0
""".strip()

        scheduler = create_scheduler("foo")
        out = scheduler.describe(app_id="1902")

        self.assertEqual(check_output.call_count, 1)
        self.assertEqual(
            check_output.call_args,
            call(
                ["sacct", "--parsable2", "-j", "1902"],
                stderr=subprocess.PIPE,
                encoding="utf-8",
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "1902")
        self.assertEqual(out.msg, "FAILED")
        self.assertEqual(out.state, specs.AppState.FAILED)
        self.assertEqual(
            out.roles,
            [
                specs.Role(
                    name="sh",
                    image="",
                    num_replicas=1,
                )
            ],
        )

    @patch(DESCRIBE_SQUEUE, side_effect=CALLED_PROCESS_ERROR)
    @patch("subprocess.check_output")
    def test_describe_sacct_running(
        self, check_output: MagicMock, _: MagicMock
    ) -> None:
        check_output.return_value = """
JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
54|echo-echo-0|compute||1|RUNNING|0:0
""".strip()

        scheduler = create_scheduler("foo")
        out = scheduler.describe("54")

        self.assertEqual(check_output.call_count, 1)
        self.assertEqual(
            check_output.call_args,
            call(
                ["sacct", "--parsable2", "-j", "54"],
                stderr=subprocess.PIPE,
                encoding="utf-8",
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "54")
        self.assertEqual(out.msg, "RUNNING")
        self.assertEqual(out.state, specs.AppState.RUNNING)

    def test_describe_squeue(self) -> None:
        with importlib.resources.path(
            __package__, "slurm-squeue-output.json"
        ) as path, open(path) as fp:
            mock_output = fp.read()

        with patch("subprocess.check_output", return_value=mock_output):
            scheduler = create_scheduler("__ignored__")
            desc = scheduler.describe(app_id="204")

            self.assertIsNotNone(desc)
            self.assertEqual(desc.app_id, "204")
            self.assertEqual(desc.state, AppState.RUNNING)

            self.assertListEqual(
                desc.roles,
                [
                    specs.Role(
                        name="trainer",
                        num_replicas=2,
                        image="/home/foo",
                        entrypoint="/tmp/tmpa4u7gedr/torchx-sbatch.sh",
                        resource=specs.Resource(cpu=1, memMB=16, gpu=-1),
                    ),
                    specs.Role(
                        name="generator",
                        num_replicas=2,
                        image="/home/foo",
                        entrypoint="/tmp/tmpa4u7gedr/torchx-sbatch.sh",
                        resource=specs.Resource(cpu=1, memMB=16, gpu=-1),
                    ),
                ],
            )
            self.assertListEqual(
                desc.roles_statuses,
                [
                    specs.RoleStatus(
                        role="trainer",
                        replicas=[
                            specs.ReplicaStatus(
                                id=0,
                                role="trainer",
                                state=AppState.RUNNING,
                                hostname="slurm-compute-node-234",
                            ),
                            specs.ReplicaStatus(
                                id=1,
                                role="trainer",
                                state=AppState.RUNNING,
                                hostname="slurm-compute-node-231",
                            ),
                        ],
                    ),
                    specs.RoleStatus(
                        role="generator",
                        replicas=[
                            specs.ReplicaStatus(
                                id=0,
                                role="generator",
                                state=AppState.RUNNING,
                                hostname="slurm-compute-node-235",
                            ),
                            specs.ReplicaStatus(
                                id=1,
                                role="generator",
                                state=AppState.RUNNING,
                                hostname="slurm-compute-node-233",
                            ),
                        ],
                    ),
                ],
            )

    @patch("subprocess.run")
    def test_list_sacct(self, run: MagicMock) -> None:
        run.return_value.stdout = b"""{\n   "meta": {\n   },\n   "errors": [\n   ],\n   "jobs": [
\n     {\n       "account": null,\n       "job_id": 123,\n       "name": "main-0",
\n       "state": {\n   "current": "COMPLETED",\n   "reason": "None"},
\n       "working_directory": "\\/home\\/ec2-user\\/tests\\/runner-1234\\/job"\n     },
\n     {\n       "account": null,\n       "job_id": 124,\n       "name": "main-0",
\n       "state": {\n   "current": "CANCELLED",\n   "reason": "None"},
\n       "working_directory": "\\/home\\/ec2-user\\/tests\\/runner-1234\\/job"\n     }\n   ]\n }"""
        scheduler = create_scheduler("foo")
        expected_apps = [
            ListAppResponse(app_id="123", state=AppState.SUCCEEDED),
            ListAppResponse(app_id="124", state=AppState.CANCELLED),
        ]
        apps = scheduler.list()
        self.assertIsNotNone(apps)
        self.assertEqual(apps, expected_apps)

    @patch("subprocess.run")
    def test_list_squeue(self, run: MagicMock) -> None:
        run.return_value.stdout = b"""{
  "jobs": [
    {
        "job_id": 1234,
        "name": "foo",
        "job_state": ["FAILED"]
    },
    {
        "job_id": 1235,
        "name": "foo",
        "job_state": ["FAILED"],
        "het_job_id": {
            "set": true,
            "infinite": false,
            "number": 0
        }
    },
    {
        "job_id": 1236,
        "name": "foo-0",
        "job_state": ["RUNNING"],
        "het_job_id": {
            "set": true,
            "infinite": false,
            "number": 1236
        }
    },
    {
        "job_id": 1237,
        "name": "foo-1",
        "job_state": ["RUNNING"],
        "het_job_id": {
            "set": true,
            "infinite": false,
            "number": 1236
        }
    }
  ]
}"""
        scheduler = create_scheduler("foo")
        expected_apps = [
            ListAppResponse(app_id="1234", state=AppState.FAILED, name="foo"),
            ListAppResponse(app_id="1235", state=AppState.FAILED, name="foo"),
            ListAppResponse(app_id="1236", state=AppState.RUNNING, name="foo-0"),
        ]
        apps = scheduler._list_squeue()
        self.assertIsNotNone(apps)
        self.assertEqual(apps, expected_apps)

    @patch(DESCRIBE_SQUEUE, return_value=None)
    @patch(DESCRIBE_SACCT, return_value=None)
    def test_log_iter(self, _1: MagicMock, _2: MagicMock) -> None:
        scheduler = create_scheduler("foo")

        for job_dir in ["", "dir"]:
            with tmp_cwd():
                if job_dir:
                    os.mkdir(job_dir)
                    _save_job_dir("54", job_dir)

                with open(os.path.join(job_dir, "slurm-54-echo-1.out"), "wt") as f:
                    f.write("hello\nworld\n")

                logs = list(
                    scheduler.log_iter(
                        "54",
                        "echo",
                        1,
                        streams=Stream.STDOUT,
                        since=datetime.datetime.now(),
                    )
                )
                self.assertEqual(logs, ["hello\n", "world\n"])

                with open(os.path.join(job_dir, "slurm-54-echo-1.err"), "wt") as f:
                    f.write("foo\nbar\n")

                logs = list(
                    scheduler.log_iter(
                        "54",
                        "echo",
                        1,
                        streams=Stream.STDERR,
                    )
                )

                self.assertEqual(logs, ["foo\n", "bar\n"])

                # no stream specified should default to STDERR
                logs = list(
                    scheduler.log_iter(
                        "54",
                        "echo",
                        1,
                    )
                )
                self.assertEqual(logs, ["foo\n", "bar\n"])

        with self.assertRaises(ValueError):
            scheduler.log_iter("54", "echo", 1, streams=Stream.COMBINED)

    @patch("subprocess.run")
    def test_dryrun_nomem(self, run: MagicMock) -> None:
        run.return_value.returncode = 0

        scheduler = create_scheduler("foo")
        app = mem_app()

        run.return_value.stdout = b"PARTITION,MEMORY\nfoo*,5000"
        info = scheduler.submit_dryrun(app, cfg={})
        self.assertIn("mem", info.request.replicas["foo-0"].sbatch_opts)

        run.return_value.stdout = b"PARTITION,MEMORY\nfoo*,1"
        info = scheduler.submit_dryrun(app, cfg={})
        self.assertNotIn("mem", info.request.replicas["foo-0"].sbatch_opts)

        run.return_value.stdout = b""
        info = scheduler.submit_dryrun(app, cfg={})
        self.assertIn("mem", info.request.replicas["foo-0"].sbatch_opts)

    def test_dryrun_comment(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(
            app,
            cfg={
                "comment": "banana foo bar",
            },
        )
        self.assertIn(
            "--comment=banana foo bar",
            info.request.cmd,
        )

    def test_dryrun_mail(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(
            app,
            cfg={
                "mail-user": "foo@bar.com",
                "mail-type": "END",
            },
        )
        self.assertIn(
            "--mail-user=foo@bar.com",
            info.request.cmd,
        )
        self.assertIn(
            "--mail-type=END",
            info.request.cmd,
        )

    @patch(
        "torchx.schedulers.slurm_scheduler.SlurmScheduler._partition_memmb",
        return_value=2048,
    )
    @patch("subprocess.run")
    def test_run_workspace_job_dir(
        self, run: MagicMock, partition_memmb: MagicMock
    ) -> None:
        with tmp_cwd():
            run.return_value.stdout = b"1234"
            scheduler = create_scheduler("foo")
            scheduler.submit(
                simple_app(),
                cfg={
                    "job_dir": "dir",
                },
                workspace=".",
            )
            self.assertIn(("1234", "dir"), _get_job_dirs().items())

        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        (args,) = args
        self.assertEqual(
            args,
            [
                "sbatch",
                "--parsable",
                "--chdir=dir",
                "dir/torchx-sbatch.sh",
            ],
        )

    @patch("subprocess.run")
    def test_partition_memmb(self, run: MagicMock) -> None:
        scheduler = create_scheduler("foo")

        ret = run.return_value
        ret.returncode = 0
        ret.stdout = b"""
PARTITION,MEMORY
scavenge,500000+
compute*,1
"""
        self.assertEqual(scheduler._partition_memmb(None), 1)
        self.assertEqual(scheduler._partition_memmb("compute"), 1)
        self.assertEqual(scheduler._partition_memmb("nonexistant"), None)
        self.assertEqual(scheduler._partition_memmb("scavenge"), 500000)

        ret.stdout = b"""
PARTITION,MEMORY
"""
        self.assertEqual(scheduler._partition_memmb(None), None)

    def _run_req(
        self, req: SlurmBatchRequest, srun_exit: int, scontrol_exit: int
    ) -> int:
        os.environ["SRUN"] = str(srun_exit)
        os.environ["SCONTROL"] = str(scontrol_exit)
        script = req.materialize()
        with tmp_cwd():
            with open("sbatch.sh", "w") as f:
                f.write(script)
            with open("test.sh", "w") as f:
                f.write(
                    """#!/bin/bash
set -evx

srun () {
  return $SRUN
}

scontrol () {
  return $SCONTROL
}

source sbatch.sh
                """
                )
            return os.WEXITSTATUS(os.system("bash test.sh"))

    def test_sbatch(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(app, cfg={})
        self.assertEqual(self._run_req(info.request, srun_exit=0, scontrol_exit=1), 0)

    def test_sbatch_requeue(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(app, cfg={})
        os.environ["SLURM_RESTART_COUNT"] = ""
        self.assertEqual(
            self._run_req(info.request, srun_exit=1, scontrol_exit=123), 123
        )
        os.environ["SLURM_RESTART_COUNT"] = "1"
        self.assertEqual(
            self._run_req(info.request, srun_exit=1, scontrol_exit=123), 123
        )
        os.environ["SLURM_RESTART_COUNT"] = "3"
        self.assertEqual(self._run_req(info.request, srun_exit=1, scontrol_exit=123), 1)
