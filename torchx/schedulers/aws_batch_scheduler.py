#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""

This contains the TorchX AWS Batch scheduler which can be used to run TorchX
components directly on AWS Batch.

This scheduler is in prototype stage and may change without notice.

Prerequisites
==============

You'll need to create an AWS Batch queue configured for multi-node parallel jobs.

See
https://docs.aws.amazon.com/batch/latest/userguide/Batch_GetStarted.html
for how to setup a job queue and compute environment. It needs to be backed by
EC2 for multi-node parallel jobs.

See
https://docs.aws.amazon.com/batch/latest/userguide/multi-node-parallel-jobs.html
for more information on distributed jobs.

If you want to use workspaces and container patching you'll also need to
configure a docker registry to store the patched containers with your changes
such as AWS ECR.

See
https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-create-repository
for how to create a image repository.
"""
import getpass
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    TypeVar,
)

import torchx
import yaml
from torchx.schedulers.api import (
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    Stream,
)

from torchx.schedulers.devices import get_device_mounts
from torchx.schedulers.ids import make_unique
from torchx.specs.api import (
    AppDef,
    AppDryRunInfo,
    AppState,
    BindMount,
    CfgVal,
    DeviceMount,
    is_terminal,
    macros,
    MISSING,
    Resource,
    Role,
    runopts,
    VolumeMount,
)
from torchx.specs.named_resources_aws import instance_type_from_resource
from torchx.util.types import none_throws
from torchx.workspace.docker_workspace import DockerWorkspaceMixin

ENV_TORCHX_ROLE_IDX = "TORCHX_ROLE_IDX"

ENV_TORCHX_ROLE_NAME = "TORCHX_ROLE_NAME"

ENV_TORCHX_IMAGE = "TORCHX_IMAGE"

DEFAULT_ROLE_NAME = "node"

TAG_TORCHX_VER = "torchx.pytorch.org/version"
TAG_TORCHX_APPNAME = "torchx.pytorch.org/app-name"
TAG_TORCHX_USER = "torchx.pytorch.org/user"


def parse_ulimits(ulimits_list: list[str]) -> List[Dict[str, Any]]:
    """
    Parse ulimit string in format: name:softLimit:hardLimit
    Multiple ulimits separated by commas.
    """
    if not ulimits_list:
        return []

    ulimits = []
    for ulimit_str in ulimits_list:
        if not ulimit_str.strip():
            continue

        parts = ulimit_str.strip().split(":")
        if len(parts) != 3:
            raise ValueError(
                f"ulimit must be in format name:softLimit:hardLimit, got: {ulimit_str}"
            )

        name, soft_limit, hard_limit = parts
        ulimits.append(
            {
                "name": name,
                "softLimit": int(soft_limit) if soft_limit != "-1" else -1,
                "hardLimit": int(hard_limit) if hard_limit != "-1" else -1,
            }
        )

    return ulimits


if TYPE_CHECKING:
    from docker import DockerClient

JOB_STATE: Dict[str, AppState] = {
    "SUBMITTED": AppState.PENDING,
    "PENDING": AppState.PENDING,
    "RUNNABLE": AppState.PENDING,
    "STARTING": AppState.PENDING,
    "RUNNING": AppState.RUNNING,
    "SUCCEEDED": AppState.SUCCEEDED,
    "FAILED": AppState.FAILED,
}


def to_millis_since_epoch(ts: datetime) -> int:
    # datetime's timestamp returns seconds since epoch
    return int(round(ts.timestamp() * 1000))


def to_datetime(ms_since_epoch: int) -> datetime:
    return datetime.fromtimestamp(ms_since_epoch / 1000)


class ResourceType(Enum):
    VCPU = auto()
    GPU = auto()
    MEMORY = auto()

    @staticmethod
    def from_str(resource_type: str) -> "ResourceType":
        for rt in ResourceType:
            if rt.name == resource_type.upper():
                return rt
        raise ValueError(
            f"No ResourceType found for `{resource_type}`. Valid types: {[r.name for r in ResourceType]}"
        )


def resource_requirements_from_resource(resource: Resource) -> List[Dict[str, str]]:
    cpu = resource.cpu if resource.cpu > 0 else 1
    gpu = resource.gpu
    memMB = resource.memMB
    assert (
        memMB > 0
    ), f"AWSBatchScheduler requires memMB to be set to a positive value, got {memMB}"

    resource_requirements = [
        {"type": ResourceType.VCPU.name, "value": str(cpu)},
        {"type": ResourceType.MEMORY.name, "value": str(memMB)},
    ]
    if gpu > 0:
        resource_requirements.append({"type": ResourceType.GPU.name, "value": str(gpu)})
    return resource_requirements


def resource_from_resource_requirements(
    resource_requirements: List[Dict[str, str]]
) -> Resource:
    resrc_req = {
        ResourceType.from_str(r["type"]): int(r["value"]) for r in resource_requirements
    }
    return Resource(
        cpu=resrc_req[ResourceType.VCPU],
        gpu=resrc_req.get(ResourceType.GPU, 0),
        memMB=resrc_req[ResourceType.MEMORY],
        # TODO kiukchung@ map back capabilities and devices
        # might be better to tag the named resource and finding the resource
        # this requires the named resource to be part of the AppDef spec
        # but today we lose the named resource str at the component level
    )


def _role_to_node_properties(
    role: Role,
    start_idx: int,
    privileged: bool = False,
    job_role_arn: Optional[str] = None,
    execution_role_arn: Optional[str] = None,
    ulimits: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    role.mounts += get_device_mounts(role.resource.devices)

    mount_points = []
    volumes = []
    devices = []
    for i, mount in enumerate(role.mounts):
        name = f"mount_{i}"
        if isinstance(mount, BindMount):
            volumes.append(
                {
                    "name": name,
                    "host": {
                        "sourcePath": mount.src_path,
                    },
                }
            )
            mount_points.append(
                {
                    "containerPath": mount.dst_path,
                    "readOnly": mount.read_only,
                    "sourceVolume": name,
                }
            )
        elif isinstance(mount, VolumeMount):
            volumes.append(
                {
                    "name": name,
                    "efsVolumeConfiguration": {
                        "fileSystemId": mount.src,
                    },
                }
            )
            mount_points.append(
                {
                    "containerPath": mount.dst_path,
                    "readOnly": mount.read_only,
                    "sourceVolume": name,
                }
            )
        elif isinstance(mount, DeviceMount):
            perm_map = {
                "r": "READ",
                "w": "WRITE",
                "m": "MKNOD",
            }
            devices.append(
                {
                    "hostPath": mount.src_path,
                    "containerPath": mount.dst_path,
                    "permissions": [perm_map[p] for p in mount.permissions],
                },
            )
        else:
            raise TypeError(f"unknown mount type {mount}")

    container = {
        "command": [role.entrypoint] + role.args,
        "image": role.image,
        "environment": [{"name": k, "value": v} for k, v in role.env.items()],
        "privileged": privileged,
        "resourceRequirements": resource_requirements_from_resource(role.resource),
        **({"ulimits": ulimits} if ulimits else {}),
        "linuxParameters": {
            # To support PyTorch dataloaders we need to set /dev/shm to larger
            # than the 64M default.
            "sharedMemorySize": role.resource.memMB,
            "devices": devices,
        },
        "logConfiguration": {
            "logDriver": "awslogs",
        },
        "mountPoints": mount_points,
        "volumes": volumes,
    }
    if job_role_arn:
        container["jobRoleArn"] = job_role_arn
    if execution_role_arn:
        container["executionRoleArn"] = execution_role_arn
    if role.num_replicas > 0:
        instance_type = instance_type_from_resource(role.resource)
        if instance_type is not None:
            container["instanceType"] = instance_type

    return {
        "targetNodes": f"{start_idx}:{start_idx + role.num_replicas - 1}",
        "container": container,
    }


def _job_ui_url(job_arn: str) -> Optional[str]:
    match = re.match(
        "arn:aws:batch:([a-z-0-9]+):[0-9]+:job/([a-z-0-9]+)",
        job_arn,
    )
    if match is None:
        return None
    region = match.group(1)
    job_id = match.group(2)
    return f"https://{region}.console.aws.amazon.com/batch/home?region={region}#jobs/mnp-job/{job_id}"


def _parse_num_replicas(target_nodes: str, num_nodes: int) -> int:
    """
    Parses the number of replicas for a role given the target_nodes string
    and total num_nodes. See docstring for ``_parse_start_and_end_idx()``
    for details on the format of ``target_nodes`` string.
    """

    start_idx, end_idx = _parse_start_and_end_idx(target_nodes, num_nodes)
    return end_idx - start_idx + 1


def _parse_start_and_end_idx(target_nodes: str, num_nodes: int) -> Tuple[int, int]:
    """
    Takes the ``target_nodes`` str (as required by AWS Batch NodeRangeProperties)
    and parses out the start and end indices (aka global rank) of the replicas in the node group.
    The ``target_nodes`` string is of the form:

    #. ``[start_node_index]:[end_node_index]`` (e.g. ``0:5``)
    #. --or-- ``:[end_node_index]`` (e.g. ``:5``)
    #. --or-- ``[start_node_index]:`` (e.g. ``0:``)
    #. --or-- ``[node_index]`` (e.g. ``0`` - single node multi-node-parallel job)

    See: https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html
    """

    indices = target_nodes.split(":")
    if len(indices) == 1:
        return int(indices[0]), int(indices[0])
    else:
        start_idx = indices[0]
        end_idx = indices[1]
        return int(start_idx or "0"), int(end_idx or str(num_nodes - 1))


@dataclass
class BatchJob:
    name: str
    queue: str
    share_id: Optional[str]
    job_def: Dict[str, object]
    images_to_push: Dict[str, Tuple[str, str]]

    def __str__(self) -> str:
        return yaml.dump(asdict(self))

    def __repr__(self) -> str:
        return str(self)


T = TypeVar("T")


def _thread_local_cache(f: Callable[[], T]) -> Callable[[], T]:
    local: threading.local = threading.local()
    key: str = "value"

    def wrapper() -> T:
        if key in local.__dict__:
            return local.__dict__[key]

        v = f()
        local.__dict__[key] = v
        return v

    return wrapper


@_thread_local_cache
def _local_session() -> "boto3.session.Session":
    import boto3.session

    return boto3.session.Session()


class AWSBatchOpts(TypedDict, total=False):
    queue: str
    user: str
    image_repo: Optional[str]
    privileged: bool
    share_id: Optional[str]
    priority: int
    job_role_arn: Optional[str]
    execution_role_arn: Optional[str]
    ulimits: Optional[list[str]]


class AWSBatchScheduler(
    DockerWorkspaceMixin, Scheduler[AWSBatchOpts, AppDef, AppDryRunInfo[BatchJob]]
):
    """
    AWSBatchScheduler is a TorchX scheduling interface to AWS Batch.

    .. code-block:: bash

        $ pip install torchx[kubernetes]
        $ torchx run --scheduler aws_batch --scheduler_args queue=torchx utils.echo --image alpine:latest --msg hello
        aws_batch://torchx_user/1234
        $ torchx status aws_batch://torchx_user/1234
        ...

    Authentication is loaded from the environment using the ``boto3`` credential
    handling.

    **Config Options**

    .. runopts::
        class: torchx.schedulers.aws_batch_scheduler.create_scheduler

    **Mounts**

    This class supports bind mounting host directories, efs volumes and host
    devices.

    * bind mount: ``type=bind,src=<host path>,dst=<container path>[,readonly]``
    * efs volume: ``type=volume,src=<efs id>,dst=<container path>[,readonly]``
    * devices: ``type=device,src=/dev/infiniband/uverbs0,[dst=<container path>][,perm=rwm]``

    See :py:func:`torchx.specs.parse_mounts` for more info.

    For other filesystems such as FSx you can mount them onto the host and bind
    mount them into your job: https://repost.aws/knowledge-center/batch-fsx-lustre-file-system-mount

    For Elastic Fabric Adapter (EFA) you'll need to use a device mount to mount
    them into the container: https://docs.aws.amazon.com/batch/latest/userguide/efa.html

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. AWSBatchScheduler will return job and replica
                status but does not provide the complete original AppSpec.
            workspaces: true
            mounts: true
            elasticity: false
    """

    def __init__(
        self,
        session_name: str,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        client: Optional[Any] = None,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        log_client: Optional[Any] = None,
        docker_client: Optional["DockerClient"] = None,
    ) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("aws_batch", session_name, docker_client=docker_client)

        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__client = client
        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__log_client = log_client

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _client(self) -> Any:
        if self.__client:
            return self.__client
        return _local_session().client("batch")

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _log_client(self) -> Any:
        if self.__log_client:
            return self.__log_client
        return _local_session().client("logs")

    def schedule(self, dryrun_info: AppDryRunInfo[BatchJob]) -> str:
        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"

        images_to_push = dryrun_info.request.images_to_push
        self.push_images(images_to_push)

        req = dryrun_info.request
        self._client.register_job_definition(**req.job_def)

        batch_job_req = {
            **{
                "jobName": req.name,
                "jobQueue": req.queue,
                "jobDefinition": req.name,
                "tags": req.job_def["tags"],
            },
            **({"shareIdentifier": req.share_id} if req.share_id is not None else {}),
        }
        self._client.submit_job(**batch_job_req)

        return f"{req.queue}:{req.name}"

    def _submit_dryrun(self, app: AppDef, cfg: AWSBatchOpts) -> AppDryRunInfo[BatchJob]:
        queue = cfg.get("queue")
        if not isinstance(queue, str):
            raise TypeError(f"config value 'queue' must be a string, got {queue}")

        share_id = cfg.get("share_id")
        priority = cfg["priority"]

        name_suffix = f"-{share_id}" if share_id is not None else ""
        name = make_unique(f"{app.name}{name_suffix}")

        assert len(app.roles) <= 5, (
            "AWS Batch only supports <= 5 roles (NodeGroups)."
            " See: https://docs.aws.amazon.com/batch/latest/userguide/multi-node-parallel-jobs.html#mnp-node-groups"
        )

        # map any local images to the remote image
        images_to_push = self.dryrun_push_images(app, cast(Mapping[str, CfgVal], cfg))

        nodes = []
        node_idx = 0
        for role_idx, role in enumerate(app.roles):
            values = macros.Values(
                img_root="",
                app_id=name,
                # this only resolves for role.args
                # if the entrypoint is run with sh or bash
                # but won't actually work for macros in env vars
                replica_id="$AWS_BATCH_JOB_NODE_INDEX",
                rank0_env="AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS",
            )
            role = values.apply(role)
            role.env[ENV_TORCHX_ROLE_IDX] = str(role_idx)
            role.env[ENV_TORCHX_ROLE_NAME] = str(role.name)
            role.env[ENV_TORCHX_IMAGE] = role.image

            nodes.append(
                _role_to_node_properties(
                    role,
                    start_idx=node_idx,
                    privileged=cfg["privileged"],
                    job_role_arn=cfg.get("job_role_arn"),
                    execution_role_arn=cfg.get("execution_role_arn"),
                    ulimits=parse_ulimits(cfg.get("ulimits") or []),
                )
            )
            node_idx += role.num_replicas

        job_def = {
            **{
                "jobDefinitionName": name,
                "type": "multinode",
                "nodeProperties": {
                    "numNodes": node_idx,
                    "mainNode": 0,
                    "nodeRangeProperties": nodes,
                },
                "retryStrategy": {
                    "attempts": max(max(role.max_retries for role in app.roles), 1),
                    "evaluateOnExit": [
                        {"onExitCode": "0", "action": "EXIT"},
                    ],
                },
                "tags": {
                    TAG_TORCHX_VER: torchx.__version__,
                    TAG_TORCHX_APPNAME: app.name,
                    TAG_TORCHX_USER: cfg.get("user"),
                    **app.metadata,
                },
            },
            **({"schedulingPriority": priority} if share_id is not None else {}),
        }

        req = BatchJob(
            name=name,
            queue=queue,
            share_id=share_id,
            job_def=job_def,
            images_to_push=images_to_push,
        )
        return AppDryRunInfo(req, repr)

    def _cancel_existing(self, app_id: str) -> None:
        job_id = self._get_job_id(app_id)
        self._client.terminate_job(
            jobId=job_id,
            reason="killed via torchx CLI",
        )

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add("queue", type_=str, help="queue to schedule job in", required=True)
        opts.add(
            "user",
            type_=str,
            default=getpass.getuser(),
            help="The username to tag the job with. `getpass.getuser()` if not specified.",
        )
        opts.add(
            "privileged",
            type_=bool,
            default=False,
            help="If true runs the container with elevated permissions."
            " Equivalent to running with `docker run --privileged`.",
        )
        opts.add(
            "share_id",
            type_=str,
            help="The share identifier for the job. "
            "This must be set if and only if the job queue has a scheduling policy.",
        )
        opts.add(
            "priority",
            type_=int,
            default=0,
            help="The scheduling priority for the job within the context of share_id. "
            "Higher number (between 0 and 9999) means higher priority. "
            "This will only take effect if the job queue has a scheduling policy.",
        )
        opts.add(
            "job_role_arn",
            type_=str,
            help="The Amazon Resource Name (ARN) of the IAM role that the container can assume for AWS permissions.",
        )
        opts.add(
            "execution_role_arn",
            type_=str,
            help="The Amazon Resource Name (ARN) of the IAM role that the ECS agent can assume for AWS permissions.",
        )
        opts.add(
            "ulimits",
            type_=List[str],
            help="Ulimit settings in format: name:softLimit:hardLimit (multiple separated by commas)",
        )
        return opts

    def _get_job_id(self, app_id: str) -> Optional[str]:
        queue, name = app_id.split(":")

        for resp in self._client.get_paginator("list_jobs").paginate(
            jobQueue=queue,
            filters=[{"name": "JOB_NAME", "values": [name]}],
        ):
            job_summary_list = resp["jobSummaryList"]
            if job_summary_list:
                return job_summary_list[0]["jobArn"]
        return None

    def _get_job(
        self, app_id: str, rank: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        job_id = self._get_job_id(app_id)
        if not job_id:
            return None
        if rank is not None:
            job_id += f"#{rank}"
        jobs = self._client.describe_jobs(jobs=[job_id])["jobs"]
        if len(jobs) == 0:
            return None
        return jobs[0]

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        job = self._get_job(app_id)
        if job is None:
            return None

        # each AppDef.role maps to a batch NodeGroup
        roles = []
        node_properties = job["nodeProperties"]
        num_nodes = node_properties["numNodes"]
        for node_group in node_properties["nodeRangeProperties"]:
            container = node_group["container"]
            env = {opt["name"]: opt["value"] for opt in container["environment"]}

            command = container["command"]
            roles.append(
                Role(
                    name=env.get(ENV_TORCHX_ROLE_NAME, DEFAULT_ROLE_NAME),
                    num_replicas=_parse_num_replicas(
                        node_group["targetNodes"], num_nodes
                    ),
                    image=container["image"],
                    entrypoint=command[0] if command else MISSING,
                    args=command[1:],
                    env=env,
                    resource=resource_from_resource_requirements(
                        container["resourceRequirements"]
                    ),
                )
            )

        return DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job["status"]],
            roles=roles,
            # TODO: role statuses
            ui_url=_job_ui_url(job["jobArn"]),
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
        if streams not in (None, Stream.COMBINED):
            raise ValueError("AWSBatchScheduler only supports COMBINED log stream")

        job = self._get_job(app_id)
        if job is None:
            return []
        node_properties = job["nodeProperties"]
        nodes = node_properties["nodeRangeProperties"]

        global_idx = -1
        # finds the global idx of the node that matches the role's k'th replica
        for i, node in enumerate(nodes):
            container = node["container"]
            env = {opt["name"]: opt["value"] for opt in container["environment"]}
            node_role = env.get(ENV_TORCHX_ROLE_NAME, DEFAULT_ROLE_NAME)
            start_idx, _ = _parse_start_and_end_idx(
                node["targetNodes"],
                node_properties["numNodes"],
            )

            # k with the replica idx within the role
            # so add k to the start index of the node group to get the global idx
            global_idx = start_idx + k

            if role_name == node_role:
                break

        assert global_idx != -1, (
            f"Role `{role_name}`'s replica `{k}` not found in job `{job['jobName']}.\n"
            f"Inspect the job by running `aws batch describe-jobs --jobs {job['jobId']}`"
        )

        job = self._get_job(app_id, rank=global_idx)
        if not job:
            return []

        if "status" in job and job["status"] == "RUNNING":
            stream_name = job["container"]["logStreamName"]
        else:
            attempts = job["attempts"]
            if len(attempts) == 0:
                return []

            attempt = attempts[-1]
            container = attempt["container"]
            stream_name = container["logStreamName"]

        iterator = self._stream_events(
            app_id,
            stream_name,
            since=since,
            until=until,
            should_tail=should_tail,
        )
        if regex:
            return filter_regex(regex, iterator)
        else:
            return iterator

    def list(self) -> List[ListAppResponse]:
        # TODO: get queue name input instead of iterating over all queues?
        all_apps = []
        for resp in self._client.get_paginator("describe_job_queues").paginate():
            queue_names = [queue["jobQueueName"] for queue in resp["jobQueues"]]
            for qn in queue_names:
                all_apps.extend(self._list_by_queue(qn))
        return all_apps

    def _list_by_queue(self, queue_name: str) -> List[ListAppResponse]:
        # By default, only running jobs are listed by batch/boto client's list_jobs API
        # When 'filters' parameter is specified, jobs with all statuses are listed
        # So use AFTER_CREATED_AT filter to list jobs in all statuses
        # milli_seconds_after_epoch can later be used to list jobs by timeframe
        MS_AFTER_EPOCH = "1"
        EVERY_STATUS = {"name": "AFTER_CREATED_AT", "values": [MS_AFTER_EPOCH]}

        jobs = []
        for resp in self._client.get_paginator("list_jobs").paginate(
            jobQueue=queue_name,
            filters=[EVERY_STATUS],
            # describe-jobs API can take up to 100 jobIds
            PaginationConfig={"MaxItems": 100},
        ):
            # torchx.pytorch.org/version tag is used to filter torchx jobs
            # list_jobs() API only returns a job summary which does not include the job's tag
            # so we need to call the describe_jobs API.
            # Ideally batch lets us pass tags as a filter to list_jobs API
            # but this is currently not supported
            job_ids = [js["jobId"] for js in resp["jobSummaryList"]]
            for jobdesc in self._get_torchx_submitted_jobs(job_ids):
                jobs.append(
                    ListAppResponse(
                        app_id=f"{queue_name}:{jobdesc['jobName']}",
                        state=JOB_STATE[jobdesc["status"]],
                    )
                )

        return jobs

    def _get_torchx_submitted_jobs(self, job_ids: List[str]) -> List[Dict[str, Any]]:
        if not job_ids:
            return []

        return [
            jobdesc
            for jobdesc in self._client.describe_jobs(jobs=job_ids)["jobs"]
            if TAG_TORCHX_VER in jobdesc["tags"]
        ]

    def _stream_events(
        self,
        app_id: str,
        stream_name: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
    ) -> Iterable[str]:
        next_token = None
        last_event_timestamp: int = 0  # in millis since epoch

        while True:
            args = {}
            if next_token is not None:
                args["nextToken"] = next_token
            if until is not None:
                args["endTime"] = to_millis_since_epoch(until)
            if since is not None:
                args["startTime"] = to_millis_since_epoch(since)
            try:
                response = self._log_client.get_log_events(
                    logGroupName="/aws/batch/job",
                    logStreamName=stream_name,
                    limit=10000,
                    startFromHead=True,
                    **args,
                )
            # pyre-fixme[66]: Exception handler type annotation `unknown` must
            #  extend BaseException.
            except self._log_client.exceptions.ResourceNotFoundException:
                return []  # noqa: B901
            if response["nextForwardToken"] == next_token:
                if (
                    not until or last_event_timestamp < to_millis_since_epoch(until)
                ) and should_tail:
                    if not is_terminal(none_throws(self.describe(app_id)).state):
                        since = to_datetime(last_event_timestamp)
                        continue
                break

            next_token = response["nextForwardToken"]

            for event in response["events"]:
                last_event_timestamp = event["timestamp"]
                yield event["message"] + "\n"


def create_scheduler(
    session_name: str,
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    client: Optional[Any] = None,
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    log_client: Optional[Any] = None,
    docker_client: Optional["DockerClient"] = None,
    **kwargs: object,
) -> AWSBatchScheduler:
    return AWSBatchScheduler(
        session_name=session_name,
        client=client,
        log_client=log_client,
        docker_client=docker_client,
    )
