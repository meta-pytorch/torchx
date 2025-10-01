# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import fnmatch
import logging
import os.path
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING, TypedDict, Union

import torchx
import yaml
from torchx.schedulers.api import (
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    split_lines,
    Stream,
)
from torchx.schedulers.devices import get_device_mounts
from torchx.schedulers.ids import make_unique
from torchx.specs.api import (
    AppDef,
    AppDryRunInfo,
    AppState,
    BindMount,
    DeviceMount,
    is_terminal,
    macros,
    ReplicaStatus,
    Role,
    RoleStatus,
    runopts,
    VolumeMount,
)
from torchx.workspace.docker_workspace import DockerWorkspaceMixin


if TYPE_CHECKING:
    from docker import DockerClient
    from docker.models.containers import Container

log: logging.Logger = logging.getLogger(__name__)

CONTAINER_STATE: Dict[str, AppState] = {
    "created": AppState.SUBMITTED,
    "restarting": AppState.PENDING,
    "running": AppState.RUNNING,
    "paused": AppState.PENDING,
    "removing": AppState.PENDING,
    "dead": AppState.FAILED,
}


@dataclass
class DockerContainer:
    image: str
    command: List[str]
    kwargs: Dict[str, object]


@dataclass
class DockerJob:
    app_id: str
    containers: List[DockerContainer]

    def __str__(self) -> str:
        return yaml.dump(self.containers)

    def __repr__(self) -> str:
        return str(self)


LABEL_VERSION: str = DockerWorkspaceMixin.LABEL_VERSION
LABEL_APP_ID: str = "torchx.pytorch.org/app-id"
LABEL_ROLE_NAME: str = "torchx.pytorch.org/role-name"
LABEL_REPLICA_ID: str = "torchx.pytorch.org/replica-id"

ENV_TORCHX_IMAGE: str = "TORCHX_IMAGE"

NETWORK = "torchx"


def has_docker() -> bool:
    try:
        import docker

        docker.from_env()
        return True
    except (ImportError, docker.errors.DockerException):
        return False


def ensure_network(client: Optional["DockerClient"] = None) -> None:
    """
    This creates the torchx docker network. Multi-process safe.
    """
    import filelock
    from docker.errors import APIError

    if client is None:
        import docker

        client = docker.from_env()

    lock_path = os.path.join(tempfile.gettempdir(), "torchx_docker_network_lock")

    # Docker networks.create check_duplicate has a race condition so we need
    # to do client side locking to ensure only one network is created.
    with filelock.FileLock(lock_path, timeout=10):
        try:
            client.networks.create(name=NETWORK, driver="bridge", check_duplicate=True)
        except APIError as e:
            if "already exists" not in str(e):
                raise


class DockerOpts(TypedDict, total=False):
    copy_env: Optional[List[str]]
    env: Optional[Dict[str, str]]
    privileged: bool


class DockerScheduler(
    DockerWorkspaceMixin, Scheduler[DockerOpts, AppDef, AppDryRunInfo[DockerJob]]
):
    """
    DockerScheduler is a TorchX scheduling interface to Docker.

    This is exposed via the scheduler `local_docker`.

    This scheduler runs the provided app via the local docker runtime using the
    specified images in the AppDef. Docker must be installed and running.  This
    provides the closest environment to schedulers that natively use Docker such
    as Kubernetes.

    .. note:: docker doesn't provide gang scheduling mechanisms. If one replica
              in a job fails, only that replica will be restarted.


    **Config Options**

    .. runopts::
        class: torchx.schedulers.docker_scheduler.create_scheduler

    **Mounts**

    This class supports bind mounting directories and named volumes.

    * bind mount: ``type=bind,src=<host path>,dst=<container path>[,readonly]``
    * named volume: ``type=volume,src=<name>,dst=<container path>[,readonly]``
    * devices: ``type=device,src=<name>[,dst=<container path>][,permissions=rwm]``

    See :py:func:`torchx.specs.parse_mounts` for more info.

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. DockerScheduler will return job and replica
                status but does not provide the complete original AppSpec.
            workspaces: true
            mounts: true
            elasticity: false
    """

    def __init__(self, session_name: str) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("docker", session_name)

    def schedule(self, dryrun_info: AppDryRunInfo[DockerJob]) -> str:
        client = self._docker_client

        req = dryrun_info.request

        images = set()
        for container in req.containers:
            images.add(container.image)
        for image in images:
            if image.startswith("sha256:"):
                continue
            log.info(f"Pulling container image: {image} (this may take a while)")
            try:
                client.images.pull(image)
            except Exception as e:
                log.warning(f"failed to pull image {image}, falling back to local: {e}")

        ensure_network(self._docker_client)

        for container in req.containers:
            client.containers.run(
                container.image,
                container.command,
                detach=True,
                **container.kwargs,
            )

        return req.app_id

    def _submit_dryrun(self, app: AppDef, cfg: DockerOpts) -> AppDryRunInfo[DockerJob]:
        from docker.types import DeviceRequest, Mount

        default_env = {}
        copy_env = cfg.get("copy_env")
        if copy_env:
            assert isinstance(
                copy_env, list
            ), f"copy_env must be a list, got {copy_env}"
            keys = set()
            for pattern in copy_env:
                keys |= set(fnmatch.filter(os.environ.keys(), pattern))
            for k in keys:
                default_env[k] = os.environ[k]

        env = cfg.get("env")
        if env:
            default_env.update(env)

        app_id = make_unique(app.name)
        req = DockerJob(app_id=app_id, containers=[])
        # trim app_id and role name in case name is longer than 64 letters
        rank0_name = f"{app_id[-30:]}-{app.roles[0].name[:30]}-0"
        for role in app.roles:
            mounts = []
            devices = []
            role.mounts += get_device_mounts(role.resource.devices)
            for mount in role.mounts:
                if isinstance(mount, BindMount):
                    mounts.append(
                        Mount(
                            target=mount.dst_path,
                            source=mount.src_path,
                            read_only=mount.read_only,
                            type="bind",
                        )
                    )
                elif isinstance(mount, VolumeMount):
                    mounts.append(
                        Mount(
                            target=mount.dst_path,
                            source=mount.src,
                            read_only=mount.read_only,
                            type="volume",
                        )
                    )
                elif isinstance(mount, DeviceMount):
                    devices.append(
                        f"{mount.src_path}:{mount.dst_path}:{mount.permissions}"
                    )
                else:
                    raise TypeError(f"unknown mount type {mount}")

            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root="",
                    app_id=app_id,
                    replica_id=str(replica_id),
                    rank0_env="TORCHX_RANK0_HOST",
                )
                replica_role = values.apply(role)
                # trim app_id and role name in case name is longer than 64 letters. Assume replica_id is less than 10_000. Remove invalid prefixes (https://github.com/moby/moby/blob/master/daemon/names/names.go#L6).
                name = re.sub(
                    r"^[^a-zA-Z0-9]+",
                    "",
                    f"{app_id[-30:]}-{role.name[:30]}-{replica_id}",
                )
                env = default_env.copy()
                if replica_role.env:
                    env.update(replica_role.env)

                # configure distributed host envs
                env["TORCHX_RANK0_HOST"] = rank0_name
                env[ENV_TORCHX_IMAGE] = replica_role.image

                c = DockerContainer(
                    image=replica_role.image,
                    command=[replica_role.entrypoint] + replica_role.args,
                    kwargs={
                        "name": name,
                        "environment": env,
                        "labels": {
                            LABEL_VERSION: torchx.__version__,
                            LABEL_APP_ID: app_id,
                            LABEL_ROLE_NAME: role.name,
                            LABEL_REPLICA_ID: str(replica_id),
                        },
                        "hostname": name,
                        "privileged": cfg.get("privileged", False),
                        "network": NETWORK,
                        "mounts": mounts,
                        "devices": devices,
                    },
                )
                if replica_role.max_retries > 0:
                    c.kwargs["restart_policy"] = {
                        "Name": "on-failure",
                        "MaximumRetryCount": replica_role.max_retries,
                    }
                resource = replica_role.resource
                if resource.memMB >= 0:
                    # To support PyTorch dataloaders we need to set /dev/shm to
                    # larger than the 64M default.
                    c.kwargs["mem_limit"] = c.kwargs["shm_size"] = (
                        f"{int(resource.memMB)}m"
                    )
                if resource.cpu >= 0:
                    c.kwargs["nano_cpus"] = int(resource.cpu * 1e9)
                if resource.gpu > 0:
                    # `compute` means a CUDA or OpenCL capable device.
                    # For more info:
                    # * https://github.com/docker/docker-py/blob/master/docker/types/containers.py
                    # * https://github.com/NVIDIA/nvidia-container-runtime
                    c.kwargs["device_requests"] = [
                        DeviceRequest(
                            count=resource.gpu,
                            capabilities=[["compute", "utility"]],
                        )
                    ]
                req.containers.append(c)

        return AppDryRunInfo(req, repr)

    def _validate(self, app: AppDef, scheduler: str, cfg: DockerOpts) -> None:
        # Skip validation step
        pass

    def _get_container(self, app_id: str, role: str, replica_id: int) -> "Container":
        client = self._docker_client
        containers = client.containers.list(
            all=True,
            filters={
                "label": [
                    f"{LABEL_APP_ID}={app_id}",
                    f"{LABEL_ROLE_NAME}={role}",
                    f"{LABEL_REPLICA_ID}={replica_id}",
                ]
            },
        )
        if len(containers) == 0:
            raise RuntimeError(
                f"failed to find container for {app_id}/{role}/{replica_id}"
            )
        elif len(containers) > 1:
            raise RuntimeError(
                f"found multiple containers for {app_id}/{role}/{replica_id}: {containers}"
            )
        return containers[0]

    def _get_containers(self, app_id: str) -> List["Container"]:
        client = self._docker_client
        return client.containers.list(
            all=True, filters={"label": f"{LABEL_APP_ID}={app_id}"}
        )

    def _cancel_existing(self, app_id: str) -> None:
        containers = self._get_containers(app_id)
        for container in containers:
            container.stop()

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "copy_env",
            type_=List[str],
            default=None,
            help="list of glob patterns of environment variables to copy if not set in AppDef. Ex: FOO_*",
        )
        opts.add(
            "env",
            type_=Dict[str, str],
            default=None,
            help="""environment variables to be passed to the run. The separator sign can be eiher comma or semicolon
            (e.g. ENV1:v1,ENV2:v2,ENV3:v3 or ENV1:V1;ENV2:V2). Environment variables from env will be applied on top
            of the ones from copy_env""",
        )
        opts.add(
            "privileged",
            type_=bool,
            default=False,
            help="If true runs the container with elevated permissions."
            " Equivalent to running with `docker run --privileged`.",
        )
        return opts

    def _get_app_state(self, container: "Container") -> AppState:
        if container.status == "exited":
            # docker doesn't have success/failed states -- we have to call
            # `wait()` to get the exit code to determine that
            status = container.wait(timeout=10)
            if status["StatusCode"] == 0:
                state = AppState.SUCCEEDED
            else:
                state = AppState.FAILED
        else:
            state = CONTAINER_STATE[container.status]
        return state

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        roles = {}
        roles_statuses = {}

        states = []

        containers = self._get_containers(app_id)
        for container in containers:
            role = container.labels[LABEL_ROLE_NAME]
            replica_id = container.labels[LABEL_REPLICA_ID]

            if role not in roles:
                roles[role] = Role(
                    name=role,
                    num_replicas=0,
                    image=container.image,
                )
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1

            state = self._get_app_state(container)

            roles_statuses[role].replicas.append(
                ReplicaStatus(
                    id=int(replica_id),
                    role=role,
                    state=state,
                    hostname=container.name,
                )
            )
            states.append(state)

        state = AppState.UNKNOWN
        if all(is_terminal(state) for state in states):
            if all(state == AppState.SUCCEEDED for state in states):
                state = AppState.SUCCEEDED
            else:
                state = AppState.FAILED
        else:
            state = next(state for state in states if not is_terminal(state))

        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=state,
        )

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
        c = self._get_container(app_id, role_name, k)

        logs = c.logs(
            since=since,
            until=until,
            stream=should_tail,
            stderr=streams != Stream.STDOUT,
            stdout=streams != Stream.STDERR,
        )

        if isinstance(logs, (bytes, str)):
            logs = _to_str(logs)

            if len(logs) == 0:
                logs = []
            else:
                logs = split_lines(logs)

        logs = map(_to_str, logs)

        if regex:
            return filter_regex(regex, logs)
        else:
            return logs

    def list(self) -> List[ListAppResponse]:
        unique_apps = {
            ListAppResponse(
                app_id=cntr.labels[LABEL_APP_ID], state=self._get_app_state(cntr)
            )
            for cntr in self._docker_client.containers.list(
                all=True, filters={"label": f"{LABEL_APP_ID}"}
            )
        }
        return list(unique_apps)


def _to_str(a: Union[str, bytes]) -> str:
    if isinstance(a, bytes):
        a = a.decode("utf-8")
    return a


def create_scheduler(session_name: str, **kwargs: Any) -> DockerScheduler:
    return DockerScheduler(
        session_name=session_name,
    )
