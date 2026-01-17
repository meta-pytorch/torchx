# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock, patch

import torchx
from torchx.schedulers.api import Stream
from torchx.schedulers.lsf_scheduler import (
    bjobs_msg_to_describe,
    bjobs_msg_to_list,
    bjobs_msg_to_log_file,
    cleanup_str,
    create_scheduler,
    find_rank0_host_from_bhosts_stdout,
    get_bsub,
    get_docker_command,
    get_job_state,
    get_submit_script,
    LsfBsub,
    LsfOpts,
    LsfScheduler,
)
from torchx.specs import (
    AppDef,
    AppState,
    BindMount,
    DeviceMount,
    macros,
    Resource,
    Role,
    VolumeMount,
)
from torchx.util import shlex


def simple_role() -> Role:
    return Role(
        name="foo",
        image="/some/path",
        entrypoint="echo",
        args=["hello", "$HOSTNAME"],
        env={},
        mounts=[],
        num_replicas=1,
    )


def simple_app() -> AppDef:
    return AppDef(
        name="foo",
        roles=[
            Role(
                name="a",
                image="/some/path",
                entrypoint="echo",
                args=[macros.replica_id, f"hello {macros.app_id}"],
                num_replicas=1,
                max_retries=3,
            ),
        ],
    )


def simple_opts() -> LsfOpts:
    return LsfOpts(
        {
            "lsf_queue": "queue",
            "jobdir": "/path/to/job",
            "container_workdir": "/path/to/container",
            "host_network": True,
            "shm_size": "10G",
        }
    )


class LsfSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.make_unique_patch = patch("torchx.schedulers.lsf_scheduler.make_unique")
        make_unique = self.make_unique_patch.start()
        make_unique.return_value = "app-name-42"

    def tearDown(self) -> None:
        self.make_unique_patch.stop()

    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, LsfScheduler)

    def test_get_job_state_DONE(self) -> None:
        self.assertEqual(get_job_state("DONE", "0"), AppState.SUCCEEDED)

    def test_get_job_state_EXIT(self) -> None:
        self.assertEqual(get_job_state("EXIT", "1"), AppState.FAILED)

    def test_get_job_state_SIGINT(self) -> None:
        self.assertEqual(get_job_state("EXIT", "130"), AppState.CANCELLED)

    def test_get_job_state_PEND(self) -> None:
        self.assertEqual(get_job_state("PEND", "0"), AppState.PENDING)

    def test_get_job_state_RUN(self) -> None:
        self.assertEqual(get_job_state("RUN", "0"), AppState.RUNNING)

    def test_get_job_state_PSUSP(self) -> None:
        self.assertEqual(get_job_state("PSUSP", "0"), AppState.PENDING)

    def test_get_job_state_USUSP(self) -> None:
        self.assertEqual(get_job_state("USUSP", "0"), AppState.PENDING)

    def test_get_job_state_SSUSP(self) -> None:
        self.assertEqual(get_job_state("SSUSP", "0"), AppState.PENDING)

    def test_get_job_state_UNKNOWN(self) -> None:
        self.assertEqual(get_job_state("UNKONWN", "0"), AppState.UNKNOWN)

    def test_get_docker_command(self) -> None:
        role = simple_role()
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_bind_mount_ro(self) -> None:
        role = simple_role()
        role.mounts = [
            BindMount(src_path="/bind/src", dst_path="/bind/dst", read_only=True)
        ]
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo -v /bind/src:/bind/dst:ro --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_bind_mount_rw(self) -> None:
        role = simple_role()
        role.mounts = [
            BindMount(src_path="/bind/src", dst_path="/bind/dst", read_only=False)
        ]
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo -v /bind/src:/bind/dst:rw --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_volume_mount_ro(self) -> None:
        role = simple_role()
        role.mounts = [VolumeMount(src="srcvol", dst_path="/vol/dst", read_only=True)]
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo --mount type=volume,src=srcvol,dst=/vol/dst,ro "
            "--entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_volume_mount_rw(self) -> None:
        role = simple_role()
        role.mounts = [VolumeMount(src="srcvol", dst_path="/vol/dst", read_only=False)]
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo --mount type=volume,src=srcvol,dst=/vol/dst "
            "--entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_device_mount(self) -> None:
        role = simple_role()
        role.mounts = [
            DeviceMount(src_path="/dev/fuse", dst_path="/dev/fuse", permissions="rwm")
        ]
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo --device=/dev/fuse:/dev/fuse:rwm --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_container_workdir(self) -> None:
        role = simple_role()
        self.assertEqual(
            get_docker_command(
                "foo", role, cfg={"container_workdir": "/tmp/container"}
            ),
            "docker run --name=foo -w /tmp/container --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_host_network(self) -> None:
        role = simple_role()
        self.assertEqual(
            get_docker_command("foo", role, cfg={"host_network": True}),
            "docker run --name=foo --net=host --ipc=host --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_port(self) -> None:
        role = simple_role()
        role.port_map = {"http": 80}
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo -p 80 --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_shm_size(self) -> None:
        role = simple_role()
        self.assertEqual(
            get_docker_command("foo", role, cfg={"shm_size": "10G"}),
            "docker run --name=foo --shm-size=10G --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_envs(self) -> None:
        role = simple_role()
        role.env = {"FOO": "bar"}
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            "docker run --name=foo -e FOO=bar --entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_resource(self) -> None:
        role = simple_role()
        role.resource = Resource(cpu=1, memMB=1, gpu=1)
        self.assertEqual(
            get_docker_command("foo", role, cfg={}),
            f"docker run --name=foo --cpus=1 --memory={1024 * 1024} --gpus all --entrypoint echo "
            "--rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_docker_command_full(self) -> None:
        role = simple_role()
        role.mounts = [
            BindMount(src_path="/bind/src", dst_path="/bind/dst", read_only=True)
        ]
        role.mounts += [
            BindMount(src_path="/bind/src", dst_path="/bind/dst", read_only=False)
        ]
        role.mounts += [
            DeviceMount(src_path="/dev/fuse", dst_path="/dev/fuse", permissions="rwm")
        ]
        role.port_map = {"http": 80}
        role.env = {"FOO": "bar"}
        role.resource = Resource(cpu=1, memMB=1, gpu=1)
        self.assertEqual(
            get_docker_command(
                "foo",
                role,
                cfg={"container_workdir": "/tmp/container", "shm_size": "10G"},
            ),
            "docker run --name=foo -v /bind/src:/bind/dst:ro -v /bind/src:/bind/dst:rw --device=/dev/fuse:/dev/fuse:rwm "
            f"-w /tmp/container -p 80 --shm-size=10G -e FOO=bar --cpus=1 --memory={1024 * 1024} --gpus all "
            "--entrypoint echo --rm /some/path hello '\\$HOSTNAME'",
        )

    def test_get_bsub(self) -> None:
        app_id = "appid"
        job_name = "job_name"
        job_host = "job_host"
        role = simple_role()
        self.assertEqual(
            get_bsub(
                app_id, job_name, role, cfg={}, head_job_name="", head_job_host=job_host
            ),
            f'bsub -P {app_id} -J {job_name} -m {job_host} -R "span[hosts=1]]" '
            f"<< EOF\n{get_docker_command(job_name, role, cfg={})}\nEOF",
        )

    def test_get_bsub_head_job(self) -> None:
        app_id = "appid"
        job_name = "job_name"
        head_job_name = "head_job"
        job_host = "job_host"
        role = simple_role()
        self.assertEqual(
            get_bsub(
                app_id,
                job_name,
                role,
                cfg={},
                head_job_name=head_job_name,
                head_job_host=job_host,
            ),
            f'bsub -P {app_id} -J {job_name} -w "started({head_job_name})" -R "select[hname!=\'{job_host}\']" '
            f'-R "span[hosts=1]]" << EOF\n{get_docker_command(job_name, role, cfg={})}\nEOF',
        )

    def test_get_bsub_jobdir(self) -> None:
        app_id = "appid"
        job_name = "job_name"
        job_host = "job_host"
        jobdir = "/data"
        role = simple_role()
        self.assertEqual(
            get_bsub(
                app_id,
                job_name,
                role,
                cfg={"jobdir": jobdir},
                head_job_name="",
                head_job_host=job_host,
            ),
            f"bsub -P {app_id} -J {job_name} -m {job_host} "
            f"-cwd {jobdir} -outdir {jobdir} -oo {jobdir}/{job_name}.submit.out -eo {jobdir}/{job_name}.submit.err "
            f'-R "span[hosts=1]]" << EOF\n{get_docker_command(job_name, role, cfg={})} > {jobdir}/{job_name}.out '
            f"2> {jobdir}/{job_name}.err\nEOF",
        )

    def test_get_bsub_queue(self) -> None:
        app_id = "appid"
        job_name = "job_name"
        job_host = "job_host"
        role = simple_role()
        self.assertEqual(
            get_bsub(
                app_id,
                job_name,
                role,
                cfg={"lsf_queue": "queue"},
                head_job_name="",
                head_job_host=job_host,
            ),
            f'bsub -P {app_id} -J {job_name} -m {job_host} -q queue -R "span[hosts=1]]" '
            f"<< EOF\n{get_docker_command(job_name, role, cfg={})}\nEOF",
        )

    def test_get_bsub_resource(self) -> None:
        app_id = "appid"
        job_name = "job_name"
        job_host = "job_host"
        role = simple_role()
        role.resource = Resource(cpu=1, memMB=1, gpu=1)
        self.assertEqual(
            get_bsub(
                app_id, job_name, role, cfg={}, head_job_name="", head_job_host=job_host
            ),
            f"bsub -P {app_id} -J {job_name} -m {job_host} "
            f'-n 1 -R "span[hosts=1] rusage[mem=1]" -gpu "num=1:mode=shared:j_exclusive=yes" << EOF\n'
            f"{get_docker_command(job_name, role, cfg={})}\nEOF",
        )

    def test_get_bsub_full(self) -> None:
        app_id = "appid"
        job_name = "job_name"
        head_job_name = "head_job"
        job_host = "job_host"
        jobdir = "/data"
        role = simple_role()
        role.resource = Resource(cpu=1, memMB=1, gpu=1)
        self.maxDiff = None
        self.assertEqual(
            get_bsub(
                app_id,
                job_name,
                role,
                cfg={"jobdir": jobdir, "lsf_queue": "queue"},
                head_job_name=head_job_name,
                head_job_host=job_host,
            ),
            f'bsub -P {app_id} -J {job_name} -w "started({head_job_name})" -R "select[hname!=\'{job_host}\']" '
            f"-cwd {jobdir} -outdir {jobdir} -oo {jobdir}/{job_name}.submit.out -eo {jobdir}/{job_name}.submit.err -q queue "
            '-n 1 -R "span[hosts=1] rusage[mem=1]" -gpu "num=1:mode=shared:j_exclusive=yes" << EOF\n'
            f"{get_docker_command(job_name, role, cfg={})} > {jobdir}/{job_name}.out 2> {jobdir}/{job_name}.err\nEOF",
        )

    def test_cleanup_str(self) -> None:
        self.assertEqual(
            cleanup_str("-aBc*%Def"),
            "abcdef",
        )

    def test_find_rank0_host_from_bhosts_stdout(self) -> None:
        role = simple_role()
        role.resource = Resource(cpu=1, memMB=1, gpu=1)
        bhosts_stdout = "icgen2host-10-240-0-21 0 0\nicgen2host-10-240-0-22 16 2\nicgen2host-10-240-0-23 16 2\n"
        self.assertEqual(
            find_rank0_host_from_bhosts_stdout(bhosts_stdout, role),
            "icgen2host-10-240-0-22",
        )

    def test_find_rank0_host_from_bhosts_stdout_too_big_request(self) -> None:
        role = simple_role()
        role.resource = Resource(cpu=10000, memMB=1, gpu=1)
        bhosts_stdout = "icgen2host-10-240-0-21 0 0\nicgen2host-10-240-0-22 16 2\nicgen2host-10-240-0-23 16 2\n"
        with self.assertRaises(ValueError):
            find_rank0_host_from_bhosts_stdout(bhosts_stdout, role)

    def test_get_submit_script(self) -> None:
        app_id = "appid"
        app = simple_app()
        cmd = ["/bin/bash"]
        rank0_host = "icgen2host-10-240-0-22"
        head_job_name = ""
        values = macros.Values(
            img_root="", app_id=app_id, replica_id="0", rank0_env="TORCHX_RANK0_HOST"
        )
        replica_role = values.apply(app.roles[0])
        replica_role.env["TORCHX_RANK0_HOST"] = rank0_host
        job_name = "appid-a-0"
        self.maxDiff = None
        self.assertEqual(
            get_submit_script(
                app_id, cmd=cmd, app=app, cfg=LsfOpts({}), rank0_host=rank0_host
            ),
            f"""#!/bin/bash
#
# Generated by TorchX {torchx.__version__}
# Run with: {shlex.join(cmd)}
#
{get_bsub(app_id, job_name, replica_role, LsfOpts({}), head_job_name, rank0_host)}
""",
        )

    def test_bjobs_msg_to_describe(self) -> None:
        # bjobs -noheader -a -P dist_app-c6v2phgkc2j2tc -o  "proj name stat exit_code"
        appid = "dist_app-c6v2phgkc2j2tc"
        msg = """dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-0 DONE -
dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-1 DONE -
"""
        describe = bjobs_msg_to_describe(appid, msg)
        self.assertIsNot(describe, None)
        if describe:
            self.assertEqual(describe.app_id, appid)
            self.assertEqual(describe.state, AppState.SUCCEEDED)
            self.assertEqual(describe.roles[0].num_replicas, 2)

    def test_bjobs_msg_to_describe_no_msg(self) -> None:
        # bjobs -noheader -a -P dist_app-c6v2phgkc2j2tc -o  "proj name stat exit_code"
        appid = "dist_app-c6v2phgkc2j2tc"
        msg = ""
        describe = bjobs_msg_to_describe(appid, msg)
        self.assertEqual(describe, None)

    def test_bjobs_msg_to_describe_fail(self) -> None:
        # bjobs -noheader -a -P dist_app-vdkcfm1p7lxcx -o  "proj name stat exit_code"
        appid = "dist_app-vdkcfm1p7lxcx"
        msg = """dist_app-vdkcfm1p7lxcx dist_app-vdkcfm1p7lxcx-dist_app-0 EXIT 1
dist_app-vdkcfm1p7lxcx dist_app-vdkcfm1p7lxcx-dist_app-1 EXIT 1
"""
        describe = bjobs_msg_to_describe(appid, msg)
        self.assertIsNot(describe, None)
        if describe:
            self.assertEqual(describe.app_id, appid)
            self.assertEqual(describe.state, AppState.FAILED)
            self.assertEqual(describe.roles[0].num_replicas, 2)

    def test_bjobs_msg_to_log_file_out(self) -> None:
        # bjobs -noheader -a -P dist_app-c6v2phgkc2j2tc -o "proj name output_dir"
        msg = """dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-0 /mnt/data/torchx
dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-1 /mnt/data/torchx"""
        log_file = bjobs_msg_to_log_file(
            "dist_app-c6v2phgkc2j2tc", "dist_app", k=0, streams=Stream.STDOUT, msg=msg
        )
        self.assertEqual(
            log_file, "/mnt/data/torchx/dist_app-c6v2phgkc2j2tc-dist_app-0.out"
        )

    def test_bjobs_msg_to_log_file_err(self) -> None:
        # bjobs -noheader -a -P dist_app-c6v2phgkc2j2tc -o "proj name output_dir"
        msg = """dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-0 /mnt/data/torchx
dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-1 /mnt/data/torchx
"""
        log_file = bjobs_msg_to_log_file(
            "dist_app-c6v2phgkc2j2tc", "dist_app", k=0, streams=Stream.STDERR, msg=msg
        )
        self.assertEqual(
            log_file, "/mnt/data/torchx/dist_app-c6v2phgkc2j2tc-dist_app-0.err"
        )

    def test_bjobs_msg_to_log_file_combined(self) -> None:
        # bjobs -noheader -a -P dist_app-c6v2phgkc2j2tc -o "proj name output_dir"
        msg = """dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-0 /mnt/data/torchx
dist_app-c6v2phgkc2j2tc dist_app-c6v2phgkc2j2tc-dist_app-1 /mnt/data/torchx
"""
        with self.assertRaises(ValueError):
            bjobs_msg_to_log_file(
                "dist_app-c6v2phgkc2j2tc",
                "dist_app",
                k=0,
                streams=Stream.COMBINED,
                msg=msg,
            )

    def test_bjobs_msg_to_log_file_no_jobdir(self) -> None:
        # bjobs -noheader -a -P dist_app-mnhnfk1gvhcqq -o "proj name output_dir"
        msg = """dist_app-mnhnfk1gvhcqq dist_app-mnhnfk1gvhcqq-dist_app-0 -
dist_app-mnhnfk1gvhcqq dist_app-mnhnfk1gvhcqq-dist_app-1 -
"""
        with self.assertRaises(ValueError):
            bjobs_msg_to_log_file(
                "dist_app-mnhnfk1gvhcqq",
                "dist_app",
                k=0,
                streams=Stream.STDERR,
                msg=msg,
            )

    def test_bjobs_msg_to_list(self) -> None:
        # bjobs -noheader -a -o "proj stat exit_code"
        msg = """dist_app-c6v2phgkc2j2tc DONE -
dist_app-c6v2phgkc2j2tc DONE -
dist_app-vdkcfm1p7lxcx EXIT 1
dist_app-vdkcfm1p7lxcx EXIT 1
"""
        listApps = bjobs_msg_to_list(msg)
        self.assertEqual(len(listApps), 2)
        self.assertEqual(listApps[0].app_id, "dist_app-c6v2phgkc2j2tc")
        self.assertEqual(listApps[0].state, AppState.SUCCEEDED)
        self.assertEqual(listApps[1].app_id, "dist_app-vdkcfm1p7lxcx")
        self.assertEqual(listApps[1].state, AppState.FAILED)

    def test_submit_dryrun(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(app, cfg={})
        req = info.request
        self.assertIsInstance(req, LsfBsub)

    def test_validate(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        scheduler._validate(app, "lsf", cfg={})

    @patch("subprocess.run")
    def test_schedule(self, run: MagicMock) -> None:
        run.return_value.stdout = b"icgen2host-10-240-0-23 16 2\n"
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(app, cfg={})
        out = scheduler.schedule(info)
        self.assertEqual(out, "app-name-42")
        self.assertEqual(run.call_count, 2)

    @patch("subprocess.run")
    def test_cancel(self, run: MagicMock) -> None:
        scheduler = create_scheduler("foo")
        app_id = "1234"
        self.assertTrue(scheduler.exists(app_id))
        self.assertEqual(run.call_count, 1)
        scheduler.cancel(app_id)
        self.assertEqual(run.call_count, 4)

    @patch("subprocess.run")
    def test_list(self, run: MagicMock) -> None:
        scheduler = create_scheduler("foo")
        self.assertEqual(scheduler.list(), [])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args[0][0],
            ["bjobs", "-noheader", "-a", "-o", "proj stat exit_code"],
        )

    @patch("torchx.schedulers.lsf_scheduler.LogIterator")
    @patch("subprocess.run")
    def test_log_iter(self, run: MagicMock, LogIterator: MagicMock) -> None:
        run.return_value.stdout = b"app-id app-id-role-0 /some/path\n"

        scheduler = create_scheduler("foo")
        out = scheduler.log_iter("app-id", "role", regex=".*")
        self.assertEqual(list(out), [])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args[0][0],
            ["bjobs", "-noheader", "-a", "-P", "app-id", "-o", "proj name output_dir"],
        )
