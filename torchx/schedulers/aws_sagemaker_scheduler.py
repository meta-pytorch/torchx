#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import getpass
import os
import re
import threading
from collections import OrderedDict as OrdDict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    TypeVar,
)

import boto3
import yaml

from sagemaker.pytorch import PyTorch
from torchx.components.structured_arg import StructuredNameArgument
from torchx.schedulers.api import (
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.specs.api import AppDef, AppDryRunInfo, AppState, CfgVal, runopts
from torchx.workspace.docker_workspace import DockerWorkspaceMixin


if TYPE_CHECKING:
    from docker import DockerClient  # pragma: no cover

JOB_STATE: Dict[str, AppState] = {
    "InProgress": AppState.RUNNING,
    "Completed": AppState.SUCCEEDED,
    "Failed": AppState.FAILED,
    "Stopping": AppState.CANCELLED,
    "Stopped": AppState.CANCELLED,
}


class AWSSageMakerOpts(TypedDict, total=False):
    """
    Opts where we can get from .torchxconfig or user command args
    """

    role: str
    instance_count: int
    instance_type: str
    keep_alive_period_in_seconds: Optional[int]
    volume_size: Optional[int]
    volume_kms_key: Optional[str]
    max_run: Optional[int]
    input_mode: Optional[str]
    output_path: Optional[str]
    output_kms_key: Optional[str]
    base_job_name: Optional[str]
    tags: Optional[Dict[str, str]]
    subnets: Optional[List[str]]
    security_group_ids: Optional[List[str]]
    model_uri: Optional[str]
    model_channel_name: Optional[str]
    metric_definitions: Optional[Dict[str, str]]
    encrypt_inter_container_traffic: Optional[bool]
    use_spot_instances: Optional[bool]
    max_wait: Optional[int]
    checkpoint_s3_uri: Optional[str]
    checkpoint_local_path: Optional[str]
    debugger_hook_config: Optional[bool]
    enable_sagemaker_metrics: Optional[bool]
    enable_network_isolation: Optional[bool]
    disable_profiler: Optional[bool]
    environment: Optional[Dict[str, str]]
    max_retry_attempts: Optional[int]
    source_dir: Optional[str]
    git_config: Optional[Dict[str, str]]
    hyperparameters: Optional[Dict[str, str]]
    container_log_level: Optional[int]
    code_location: Optional[str]
    dependencies: Optional[List[str]]
    training_repository_access_mode: Optional[str]
    training_repository_credentials_provider_arn: Optional[str]
    disable_output_compression: Optional[bool]
    enable_infra_check: Optional[bool]


@dataclass
class AWSSageMakerJob:
    """
    Jobs defined the key values that is requried to schedule a job. This will be the value
    of `request` in the AppDryRunInfo object.

    - job_name: defines the job name shown in SageMaker
    - job_def: defines the job description that will be used to schedule the job on SageMaker
    - images_to_push: used by torchx to push to image_repo
    """

    job_name: str
    job_def: Dict[str, Any]
    images_to_push: Dict[str, Tuple[str, str]]

    def __str__(self) -> str:
        return yaml.dump(asdict(self))

    def __repr__(self) -> str:
        return str(self)


T = TypeVar("T")


def _thread_local_cache(f: Callable[[], T]) -> Callable[[], T]:
    # decorator function for keeping object in cache
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
def _local_session() -> boto3.session.Session:
    return boto3.session.Session()


def _merge_ordered(
    src: Optional[Dict[str, str]], extra: Dict[str, str]
) -> OrderedDict[str, str]:
    merged = OrdDict(src or {})
    merged.update(extra)
    return merged


class AWSSageMakerScheduler(
    DockerWorkspaceMixin,
    Scheduler[AWSSageMakerOpts, AppDef, AppDryRunInfo[AWSSageMakerJob]],
):
    """
    AWSSageMakerScheduler is a TorchX scheduling interface to AWS SageMaker.

    .. code-block:: bash

        $ torchx run -s aws_sagemaker utils.echo --image alpine:latest --msg hello
        aws_batch://torchx_user/1234
        $ torchx status aws_batch://torchx_user/1234
        ...

    Authentication is loaded from the environment using the ``boto3`` credential
    handling.

    **Config Options**

    .. runopts::
        class: torchx.schedulers.aws_sagemaker_scheduler.create_scheduler

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: false
            distributed: true
            describe: |
                Partial support. SageMakerScheduler will return job and replica
                status but does not provide the complete original AppSpec.
            workspaces: true
            mounts: false
            elasticity: false
    """

    def __init__(
        self,
        session_name: str,
        client: Optional[Any] = None,  # pyre-ignore[2]
        docker_client: Optional["DockerClient"] = None,
    ) -> None:
        super().__init__("aws_sagemaker", session_name, docker_client=docker_client)
        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__client = client

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _client(self) -> Any:
        if self.__client:
            return self.__client
        return _local_session().client("sagemaker")

    def schedule(self, dryrun_info: AppDryRunInfo[AWSSageMakerJob]) -> str:
        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"

        images_to_push = dryrun_info.request.images_to_push
        self.push_images(images_to_push)

        req = dryrun_info.request
        pt_estimator = PyTorch(**req.job_def)
        pt_estimator.fit(wait=False, job_name=req.job_name)

        return req.job_name

    def _submit_dryrun(
        self, app: AppDef, cfg: AWSSageMakerOpts
    ) -> AppDryRunInfo[AWSSageMakerJob]:
        role = app.roles[0]
        entrypoint, hyperparameters = self._parse_args(role.args)

        # map any local images to the remote image
        images_to_push = self.dryrun_push_images(app, cast(Mapping[str, CfgVal], cfg))
        structured_name_kwargs = {}
        if entrypoint.startswith("-m"):
            structured_name_kwargs["m"] = entrypoint.replace("-m", "").strip()
        else:
            structured_name_kwargs["script"] = entrypoint
        structured_name = StructuredNameArgument.parse_from(
            app.name, **structured_name_kwargs
        )
        job_name = make_unique(structured_name.run_name)

        role.env["TORCHX_JOB_ID"] = job_name

        # see https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.EstimatorBase
        job_def = {
            "entry_point": entrypoint,
            "image_uri": role.image,
            "distribution": {"torch_distributed": {"enabled": True}},
        }

        cfg["environment"] = _merge_ordered(cfg.get("environment"), role.env)
        # hyperparameters are used for both script/module entrypoint args and the values from .torchxconfig
        # order matters, adding script args last to handle wildcard parameters
        cfg["hyperparameters"] = _merge_ordered(
            cfg.get("hyperparameters"), hyperparameters
        )
        # tags are used for AppDef metadata and the values from .torchxconfig
        cfg["tags"] = [  # pyre-ignore[54]
            *(cfg.get("tags") or []),
            *({"Key": k, "Value": v} for k, v in app.metadata.items()),
        ]
        # following the principle of least astonishment defaulting source_dir to current working directory
        cfg["source_dir"] = cfg.get("source_dir") or os.getcwd()

        for key in cfg:
            if key in job_def:
                raise ValueError(
                    f"{key} is controlled by aws_sagemaker_scheduler and is set to {job_def[key]}"
                )
            value = cfg.get(key)  # type: ignore
            if value is not None:
                job_def[key] = value  # type: ignore

        req = AWSSageMakerJob(
            job_name=job_name,
            job_def=job_def,
            images_to_push=images_to_push,
        )
        return AppDryRunInfo(req, repr)

    def _parse_args(self, args: List[str]) -> Tuple[str, Dict[str, str]]:
        if len(args) < 1:
            raise ValueError("Not enough args to resolve entrypoint")
        offset = 1
        if args[0] == "-m":
            if len(args) < 2:
                raise ValueError("Missing module name")
            offset += 1
        entrypoint = " ".join(args[:offset])
        hyperparameters = OrdDict()  # the order matters, e.g. for wildcard params
        while offset < len(args):
            arg = args[offset]
            sp_pos = arg.find("=")
            if sp_pos < 0:
                if offset + 1 >= len(args):
                    raise ValueError(
                        "SageMaker currently only supports named arguments"
                    )
                key = arg
                offset += 1
                value = args[offset]
            else:
                key = arg[:sp_pos]
                value = arg[sp_pos + 1 :]
            if not key.startswith("--"):
                raise ValueError("SageMaker only supports arguments that start with --")
            offset += 1
            hyperparameters[key[2:]] = value
        return entrypoint, hyperparameters

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "role",
            type_=str,
            help="an AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints use this role to access training data and model artifacts. After the endpoint is created, the inference code might use the IAM role, if it needs to access an AWS resource.",
            required=True,
        )
        opts.add(
            "instance_count",
            type_=int,
            default=1,
            help="number of Amazon EC2 instances to use for training. Required if instance_groups is not set.",
        )
        opts.add(
            "instance_type",
            type_=str,
            help="type of EC2 instance to use for training, for example, 'ml.c4.xlarge'",
            required=True,
        )
        opts.add(
            "user",
            type_=str,
            default=getpass.getuser(),
            help="the username to tag the job with. `getpass.getuser()` if not specified.",
        )
        opts.add(
            "keep_alive_period_in_seconds",
            type_=int,
            default=None,
            help="the duration of time in seconds to retain configured resources in a warm pool for subsequent training jobs.",
        )
        opts.add(
            "volume_size",
            type_=int,
            default=None,
            help="size in GB of the storage volume to use for storing input and output data during training (default: 30).",
        )
        opts.add(
            "volume_kms_key",
            type_=str,
            default=None,
            help="KMS key ID for encrypting EBS volume attached to the training instance.",
        )
        opts.add(
            "max_run",
            type_=int,
            default=None,
            help="timeout in seconds for training (default: 24 * 60 * 60).",
        )
        opts.add(
            "input_mode",
            type_=str,
            default=None,
            help="the input mode that the algorithm supports (default: ‘File’).",
        )
        opts.add(
            "output_path",
            type_=str,
            default=None,
            help="S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket. If the bucket with the specific name does not exist, the estimator creates the bucket during the fit() method execution.",
        )
        opts.add(
            "output_kms_key",
            type_=str,
            default=None,
            help="KMS key ID for encrypting the training output (default: Your IAM role’s KMS key for Amazon S3).",
        )
        opts.add(
            "base_job_name",
            type_=str,
            default=None,
            help="prefix for training job name when the fit() method launches. If not specified, the estimator generates a default job name based on the training image name and current timestamp.",
        )
        opts.add(
            "tags",
            type_=List[Dict[str, str]],
            default=None,
            help="list of tags for labeling a training job.",
        )
        opts.add(
            "subnets",
            type_=List[str],
            default=None,
            help="list of subnet ids. If not specified training job will be created without VPC config.",
        )
        opts.add(
            "security_group_ids",
            type_=List[str],
            default=None,
            help="list of security group ids. If not specified training job will be created without VPC config.",
        )
        opts.add(
            "model_uri",
            type_=str,
            default=None,
            help="URI where a pre-trained model is stored, either locally or in S3.",
        )
        opts.add(
            "model_channel_name",
            type_=str,
            default=None,
            help="name of the channel where ‘model_uri’ will be downloaded (default: ‘model’).",
        )
        opts.add(
            "metric_definitions",
            type_=List[Dict[str, str]],
            default=None,
            help="list of dictionaries that defines the metric(s) used to evaluate the training jobs. Each dictionary contains two keys: ‘Name’ for the name of the metric, and ‘Regex’ for the regular expression used to extract the metric from the logs.",
        )
        opts.add(
            "encrypt_inter_container_traffic",
            type_=bool,
            default=None,
            help="specifies whether traffic between training containers is encrypted for the training job (default: False).",
        )
        opts.add(
            "use_spot_instances",
            type_=bool,
            default=None,
            help="specifies whether to use SageMaker Managed Spot instances for training. If enabled then the max_wait arg should also be set.",
        )
        opts.add(
            "max_wait",
            type_=int,
            default=None,
            help="timeout in seconds waiting for spot training job.",
        )
        opts.add(
            "checkpoint_s3_uri",
            type_=str,
            default=None,
            help="S3 URI in which to persist checkpoints that the algorithm persists (if any) during training.",
        )
        opts.add(
            "checkpoint_local_path",
            type_=str,
            default=None,
            help="local path that the algorithm writes its checkpoints to.",
        )
        opts.add(
            "debugger_hook_config",
            type_=bool,
            default=None,
            help="configuration for how debugging information is emitted with SageMaker Debugger. If not specified, a default one is created using the estimator’s output_path, unless the region does not support SageMaker Debugger. To disable SageMaker Debugger, set this parameter to False.",
        )
        opts.add(
            "enable_sagemaker_metrics",
            type_=bool,
            default=None,
            help="enable SageMaker Metrics Time Series.",
        )
        opts.add(
            "enable_network_isolation",
            type_=bool,
            default=None,
            help="specifies whether container will run in network isolation mode (default: False).",
        )
        opts.add(
            "disable_profiler",
            type_=bool,
            default=None,
            help="specifies whether Debugger monitoring and profiling will be disabled (default: False).",
        )
        opts.add(
            "environment",
            type_=Dict[str, str],
            default=None,
            help="environment variables to be set for use during training job",
        )
        opts.add(
            "max_retry_attempts",
            type_=int,
            default=None,
            help="number of times to move a job to the STARTING status. You can specify between 1 and 30 attempts.",
        )
        opts.add(
            "source_dir",
            type_=str,
            default=None,
            help="absolute, relative, or S3 URI Path to a directory with any other training source code dependencies aside from the entry point file (default: current working directory)",
        )
        opts.add(
            "git_config",
            type_=Dict[str, str],
            default=None,
            help="git configurations used for cloning files, including repo, branch, commit, 2FA_enabled, username, password, and token.",
        )
        opts.add(
            "hyperparameters",
            type_=Dict[str, str],
            default=None,
            help="dictionary containing the hyperparameters to initialize this estimator with.",
        )
        opts.add(
            "container_log_level",
            type_=int,
            default=None,
            help="log level to use within the container (default: logging.INFO).",
        )
        opts.add(
            "code_location",
            type_=str,
            default=None,
            help="S3 prefix URI where custom code is uploaded.",
        )
        opts.add(
            "dependencies",
            type_=List[str],
            default=None,
            help="list of absolute or relative paths to directories with any additional libraries that should be exported to the container.",
        )
        opts.add(
            "training_repository_access_mode",
            type_=str,
            default=None,
            help="specifies how SageMaker accesses the Docker image that contains the training algorithm.",
        )
        opts.add(
            "training_repository_credentials_provider_arn",
            type_=str,
            default=None,
            help="Amazon Resource Name (ARN) of an AWS Lambda function that provides credentials to authenticate to the private Docker registry where your training image is hosted.",
        )
        opts.add(
            "disable_output_compression",
            type_=bool,
            default=None,
            help="when set to true, Model is uploaded to Amazon S3 without compression after training finishes.",
        )
        opts.add(
            "enable_infra_check",
            type_=bool,
            default=None,
            help="specifies whether it is running Sagemaker built-in infra check jobs.",
        )
        return opts

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        job = self._get_job(app_id)
        if job is None:
            return None

        return DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job["TrainingJobStatus"]],
            ui_url=self._job_ui_url(job["TrainingJobArn"]),
        )

    def list(self) -> List[ListAppResponse]:
        raise NotImplementedError()

    def _cancel_existing(self, app_id: str) -> None:
        self._client.stop_training_job(TrainingJobName=app_id)

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
        raise NotImplementedError()

    def _get_job(self, app_id: str) -> Optional[Dict[str, Any]]:
        job = self._client.describe_training_job(TrainingJobName=app_id)
        return job

    def _job_ui_url(self, job_arn: str) -> Optional[str]:
        match = re.match(
            "arn:aws:sagemaker:(?P<region>[a-z-0-9]+):[0-9]+:training-job/(?P<job_id>[a-z-0-9]+)",
            job_arn,
        )
        if match is None:
            return None
        region = match.group("region")
        job_id = match.group("job_id")
        return f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#jobs/{job_id}"


def create_scheduler(session_name: str, **kwargs: object) -> AWSSageMakerScheduler:
    return AWSSageMakerScheduler(session_name=session_name)
