#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This contains the TorchX LSF scheduler which can be used to run TorchX
components on a LSF cluster.

This scheduler is in prototype stage and may change without notice. If you run
into any issues or have feedback please submit an issue.

Prerequisites
==============

You'll need either an existing LSF cluster to run your jobs or for individuals
you can install LSF Community Edition.

See the LSF documentation for more details:
https://www.ibm.com/docs/en/cloud-private/3.2.x?topic=paks-spectrum-lsf-community-edition
"""
import os.path
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, TypedDict

import torchx
from torchx.schedulers.api import (
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    split_lines_iterator,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.schedulers.local_scheduler import LogIterator
from torchx.specs import (
    AppDef,
    AppDryRunInfo,
    AppState,
    BindMount,
    DeviceMount,
    macros,
    NONE,
    ReplicaStatus,
    Role,
    RoleStatus,
    runopts,
    VolumeMount,
)
from torchx.util import shlex

JOB_STATE: Dict[str, AppState] = {
    "DONE": AppState.SUCCEEDED,
    "EXIT": AppState.FAILED,
    "PEND": AppState.PENDING,
    "RUN": AppState.RUNNING,
    "PSUSP": AppState.PENDING,
    "USUSP": AppState.PENDING,
    "SSUSP": AppState.PENDING,
}


def get_job_state(state_str: str, exit_code: str) -> AppState:
    state = AppState.UNKNOWN
    if state_str in JOB_STATE.keys():
        state = JOB_STATE[state_str]
    if state == AppState.FAILED and exit_code == "130":  # likely SIGINT
        state = AppState.CANCELLED
    return state


class LsfOpts(TypedDict, total=False):
    lsf_queue: Optional[str]
    jobdir: Optional[str]
    container_workdir: Optional[str]
    host_network: Optional[bool]
    shm_size: Optional[str]


def get_docker_command(job_name: str, role: Role, cfg: LsfOpts) -> str:
    cmds = ["docker", "run", f"--name={job_name}"]
    for mount in role.mounts:
        if isinstance(mount, BindMount):
            rw = "rw"
            if mount.read_only:
                rw = "ro"
            cmds += ["-v", f"{mount.src_path}:{mount.dst_path}:{rw}"]
        elif isinstance(mount, VolumeMount):
            ro = ""
            if mount.read_only:
                ro = ",ro"
            cmds += [
                "--mount",
                f"type=volume,src={mount.src},dst={mount.dst_path}{ro}",
            ]
        elif isinstance(mount, DeviceMount):
            cmds += [f"--device={mount.src_path}:{mount.dst_path}:{mount.permissions}"]
    container_workdir = cfg.get("container_workdir")
    if container_workdir:
        cmds += ["-w", container_workdir]
    host_network = cfg.get("host_network")
    if host_network:
        cmds += ["--net=host", "--ipc=host"]
    else:
        for name, port in role.port_map.items():
            cmds += ["-p", str(port)]
    shm_size = cfg.get("shm_size")
    if shm_size:
        cmds += [f"--shm-size={shm_size}"]
    for key, value in dict(role.env).items():
        cmds += ["-e", f"{key}={value}"]

    resource = role.resource
    if resource != NONE:
        if resource.cpu > 0:
            cmds += [f"--cpus={str(resource.cpu)}"]
        if resource.memMB > 0:
            cmds += [f"--memory={str(resource.memMB * 1024 * 1024)}"]
        if resource.gpu > 0:
            cmds += ["--gpus", "all"]
    cmds += ["--entrypoint", role.entrypoint, "--rm", role.image] + [
        arg.replace("$", "\\$") for arg in role.args
    ]
    return shlex.join(cmds)


def get_command(job_name: str, role: Role, cfg: LsfOpts) -> str:
    return get_docker_command(job_name, role, cfg)  # TODO: add get_singularity_command


def get_bsub(
    app_id: str,
    job_name: str,
    role: Role,
    cfg: LsfOpts,
    head_job_name: str,
    head_job_host: str,
) -> str:
    bsub_args = ["bsub", "-P", app_id, "-J", job_name]
    if head_job_name != "":
        bsub_args += [
            "-w",
            f'"started({head_job_name})"',
            "-R",
            f"\"select[hname!='{head_job_host}']\"",
        ]
    else:
        bsub_args += ["-m", head_job_host]
    jobdir = cfg.get("jobdir")
    if jobdir:
        bsub_args += [
            "-cwd",
            jobdir,
            "-outdir",
            jobdir,
            "-oo",
            f"{jobdir}/{job_name}.submit.out",
            "-eo",
            f"{jobdir}/{job_name}.submit.err",
        ]
    queue = cfg.get("lsf_queue")
    if queue:
        bsub_args += ["-q", queue]
    resource = role.resource

    if resource is not None:
        if resource.cpu > 0:
            bsub_args += ["-n", str(resource.cpu)]
        if resource.memMB > 0:
            bsub_args += ["-R", f'"span[hosts=1] rusage[mem={str(resource.memMB)}]"']
        else:
            bsub_args += ["-R", '"span[hosts=1]]"']
        if resource.gpu > 0:
            bsub_args += [
                "-gpu",
                f'"num={str(resource.gpu)}:mode=shared:j_exclusive=yes"',
            ]
    bsub_line = " ".join(bsub_args)
    if jobdir:
        return f"{bsub_line} << EOF\n{get_command(job_name, role, cfg)} > {jobdir}/{job_name}.out 2> {jobdir}/{job_name}.err\nEOF"
    else:
        return f"{bsub_line} << EOF\n{get_command(job_name, role, cfg)}\nEOF"


def cleanup_str(data: str) -> str:
    """
    Invokes ``lower`` on thes string and removes all
    characters that do not satisfy ``[a-z0-9]`` pattern.
    This method is mostly used to make sure kubernetes scheduler gets
    the job name that does not violate its validation.
    """
    if data.startswith("-"):
        data = data[1:]
    pattern = r"[a-z0-9\-_]"
    return "".join(re.findall(pattern, data.lower()))


def find_rank0_host_from_bhosts_stdout(msg: str, role: Role) -> str:
    resource = role.resource
    cpu = 1
    gpu = 0

    if resource != NONE:
        if resource.cpu > 0:
            cpu = resource.cpu
        if resource.gpu > 0:
            gpu = resource.gpu

    for line in msg.split("\n"):
        split = line.split(" ")
        if len(split) >= 3 and cpu <= int(split[1]) and gpu <= int(split[2]):
            return split[0]
    raise ValueError(
        f"cannot find a host with {cpu} CPUs, and {gpu} GPUs. Try again with enough available resource."
    )


def find_rank0_host(role: Role) -> str:
    p = subprocess.run(
        ["bhosts", "-noheader", "-o", "hname max ng"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return find_rank0_host_from_bhosts_stdout(p.stdout.decode("utf-8"), role)


def get_submit_script(
    app_id: str, cmd: List[str], app: AppDef, cfg: LsfOpts, rank0_host: str
) -> str:
    bsubs = []
    head_job_name = ""
    for role_idx, role in enumerate(app.roles):
        for replica_id in range(role.num_replicas):
            values = macros.Values(
                img_root="",
                app_id=app_id,
                replica_id=str(replica_id),
                rank0_env="TORCHX_RANK0_HOST",
            )
            replica_role = values.apply(role)
            name = cleanup_str(f"{role.name}-{replica_id}")
            replica_role.env["TORCHX_RANK0_HOST"] = rank0_host
            job_name = app_id + "-" + name
            bsubs.append(
                get_bsub(app_id, job_name, replica_role, cfg, head_job_name, rank0_host)
            )
            if role_idx == 0 and replica_id == 0:
                head_job_name = job_name
    script = f"""#!/bin/bash
#
# Generated by TorchX {torchx.__version__}
# Run with: {shlex.join(cmd)}
#
"""
    return script + "\n".join(bsubs) + "\n"


def bjobs_msg_to_describe(app_id: str, msg: str) -> Optional[DescribeAppResponse]:
    if msg == "":
        return None
    roles = {}
    roles_statuses = {}
    app_state = AppState.RUNNING
    success_count = 0
    total_count = 0
    for line in msg.split("\n"):
        split = line.split(" ")
        if len(split) < 2:
            continue
        proj = split[0]
        role, _, idx = split[1][len(proj) + 1 :].rpartition("-")
        idx = int(idx)
        if role not in roles:
            roles[role] = Role(name=role, num_replicas=0, image="")
            roles_statuses[role] = RoleStatus(role, [])
        roles[role].num_replicas += 1
        state = get_job_state(split[2], split[3])
        roles_statuses[role].replicas.append(
            ReplicaStatus(id=idx, role=role, state=state, hostname="")
        )
        if (
            state == AppState.FAILED
            or state == AppState.CANCELLED
            or state == AppState.PENDING
            or state == AppState.UNKNOWN
        ):
            app_state = state
        elif state == AppState.SUCCEEDED:
            success_count += 1
        total_count += 1
    if success_count == total_count:
        app_state = AppState.SUCCEEDED
    # set roles, roles_statuses, app_state
    return DescribeAppResponse(
        app_id=app_id,
        roles=list(roles.values()),
        roles_statuses=list(roles_statuses.values()),
        state=app_state,
        msg=msg,
    )


def bjobs_msg_to_log_file(
    app_id: str,
    role_name: str,
    k: int = 0,
    streams: Optional[Stream] = None,
    msg: str = "",
) -> str:
    if streams == Stream.COMBINED:
        raise ValueError(
            "LsfScheduler does not support COMBINED log stream."
            " Use `stdout` or `stderr`"
        )

    extension = "err" if streams == Stream.STDERR else "out"

    lines = msg.split("\n")
    jobs = {}
    log_file = ""
    for line in lines:
        if line != "":
            split = line.split(" ")
            if split[2] == "-":
                continue
            proj = split[0]
            role, _, idx = split[1][len(proj) + 1 :].rpartition("-")
            if app_id == proj and role == role_name and idx == str(k):
                log_file = split[2] + f"/{split[1]}.{extension}"
    if log_file == "":
        raise ValueError(
            f"cannot find log directory for {app_id}. Note: need to specify -cfg jobdir to use this functionality."
        )
    return log_file


def bjobs_msg_to_list(msg: str) -> List[ListAppResponse]:
    ret = []
    lines = msg.split("\n")
    apps = {}
    for line in lines:
        if line != "":
            split = line.split(" ")
            state = get_job_state(split[1], split[2])
            if split[0] not in apps.keys():
                apps[split[0]] = []
            apps[split[0]].append(state)
    for app_id, states in apps.items():
        success_count = 0
        app_state = AppState.RUNNING
        for state in states:
            if (
                state == AppState.FAILED
                or state == AppState.CANCELLED
                or state == AppState.PENDING
                or state == AppState.UNKNOWN
            ):
                app_state = state
                break
            elif state == AppState.SUCCEEDED:
                success_count += 1
        if success_count == len(states):
            app_state = AppState.SUCCEEDED
        ret.append(ListAppResponse(app_id=app_id, state=app_state))
    return ret


@dataclass
class LsfBsub:
    jobdir: Optional[str]
    app_id: str
    app: AppDef
    cfg: LsfOpts
    cmd: List[str]

    def materialize(self, rank0_host: str = "RANK0_HOST") -> str:
        return get_submit_script(self.app_id, self.cmd, self.app, self.cfg, rank0_host)

    def __repr__(self) -> str:
        return f"""{' '.join(self.cmd + ['$BSUB_SCRIPT'])}
#------------
# BSUB_SCRIPT
#------------
{self.materialize()}"""


class LsfScheduler(Scheduler[LsfOpts, AppDef, AppDryRunInfo]):
    """
    **Example: hello_world**

    .. code-block:: bash

        $ torchx run -s lsf -cfg jobdir=/mnt/data/torchx utils.echo --image alpine:latest --msg hello_world --num_replicas 3
        ...

    **Example: Gloo**

    .. code-block:: bash

        $ cp dist_app.py /mnt/data/dist/
        $ torchx run -s lsf -cfg "jobdir=/mnt/data/torchx,host_network=True" dist.ddp -j 2x2 --gpu 2 \
          --script /data/dist_app.py --mount "type=bind,src=/mnt/data/dist,dst=/data"
        ...

    **Config Options**

    .. runopts::
        class: torchx.schedulers.lsf_scheduler.create_scheduler

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
          cancel: true
          logs: true
          distributed: true
          describe: |
              LsfScheduler will return job.
          mounts: true
          workspaces: false
          elasticity: false

    **TOFIX**

    - On host_network=False, tasks cannot reoslve static names such as /etc/hosts (containers cannot reach it without host network)
    - Image downloads should be separated jobs

    """

    def __init__(self, session_name: str) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("lsf", session_name)

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "lsf_queue",
            type_=str,
            default=None,
            help="queue name to submit jobs",
        )
        opts.add(
            "jobdir",
            type_=str,
            default=None,
            help="The directory to place the job code and outputs. The directory must not exist and will be created.",
        )
        opts.add(
            "container_workdir",
            type_=str,
            default=None,
            help="working directory in container jobs",
        )
        opts.add(
            "host_network",
            type_=bool,
            default=False,
            help="True if using the host network for jobs",
        )
        opts.add(
            "shm_size",
            type_=str,
            default="64m",
            help="size of shared memory (/dev/shm) for jobs",
        )

        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[LsfBsub]) -> str:
        req = dryrun_info.request
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(req.jobdir or tempdir, f"{req.app_id}.sh")
            req.cmd += [path]
            with open(path, "w") as f:
                f.write(req.materialize(find_rank0_host(req.app.roles[0])))
            subprocess.run(req.cmd, stdout=subprocess.PIPE, check=True)
        return req.app_id

    def _validate(self, app: AppDef, scheduler: str, cfg: LsfOpts) -> None:
        # Skip validation step for lsf
        pass

    def _cancel_existing(self, app_id: str) -> None:
        p = subprocess.run(
            ["bjobs", "-noheader", "-a", "-P", app_id, "-o", "id"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        msg = p.stdout.decode("utf-8")
        if msg != "":
            subprocess.run(
                ["bkill"] + msg.strip().split("\n"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )

    def _submit_dryrun(self, app: AppDef, cfg: LsfOpts) -> AppDryRunInfo[LsfBsub]:
        jobdir = cfg.get("jobdir")
        assert jobdir is None or isinstance(jobdir, str), "jobdir must be str"

        app_id = cleanup_str(make_unique(app.name))
        return AppDryRunInfo(
            LsfBsub(app_id=app_id, cmd=["/bin/bash"], jobdir=jobdir, app=app, cfg=cfg),
            repr,
        )

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        p = subprocess.run(
            [
                "bjobs",
                "-noheader",
                "-a",
                "-P",
                app_id,
                "-o",
                "proj name stat exit_code",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return bjobs_msg_to_describe(app_id=app_id, msg=p.stdout.decode("utf-8"))

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        p = subprocess.run(
            ["bjobs", "-noheader", "-a", "-P", app_id, "-o", "proj name output_dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        log_file = bjobs_msg_to_log_file(
            app_id=app_id,
            role_name=role_name,
            k=k,
            streams=streams,
            msg=p.stdout.decode("utf-8"),
        )
        iterator = split_lines_iterator(
            LogIterator(app_id, log_file, self, should_tail=should_tail)
        )
        if regex:
            iterator = filter_regex(regex, iterator)
        return iterator

    def list(self) -> List[ListAppResponse]:
        p = subprocess.run(
            ["bjobs", "-noheader", "-a", "-o", "proj stat exit_code"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return bjobs_msg_to_list(p.stdout.decode("utf-8"))


def create_scheduler(session_name: str, **kwargs: Any) -> LsfScheduler:
    return LsfScheduler(
        session_name=session_name,
    )
