#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import getpass
import os
import re
import threading
from collections import OrderedDict as OrdDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, OrderedDict, TYPE_CHECKING, TypeVar

import boto3
import yaml
from sagemaker.pytorch import PyTorch
from torchx.components.structured_arg import StructuredNameArgument
from torchx.schedulers.api import (
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    Stream,
    StructuredOpts,
)
from torchx.schedulers.ids import make_unique
from torchx.specs.api import AppDef, AppDryRunInfo, AppState, CfgVal, runopts
from torchx.workspace.docker_workspace import DockerWorkspaceMixin


if TYPE_CHECKING:
    from docker import DockerClient  # pragma: no cover

JOB_STATE: dict[str, AppState] = {
    "InProgress": AppState.RUNNING,
    "Completed": AppState.SUCCEEDED,
    "Failed": AppState.FAILED,
    "Stopping": AppState.CANCELLED,
    "Stopped": AppState.CANCELLED,
}


@dataclass
class Opts(StructuredOpts):
    """Typed configuration options for AWSSageMakerScheduler."""

    role: str
    """An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs that create Amazon SageMaker endpoints use this role to access training data and model artifacts. After the endpoint is created, the inference code might use the IAM role, if it needs to access an AWS resource."""

    instance_type: str
    """Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'."""

    instance_count: int = 1
    """Number of Amazon EC2 instances to use for training. Required if instance_groups is not set."""

    user: str = getpass.getuser()
    """The username to tag the job with. `getpass.getuser()` if not specified."""

    keep_alive_period_in_seconds: int | None = None
    """The duration of time in seconds to retain configured resources in a warm pool for subsequent training jobs."""

    volume_size: int | None = None
    """Size in GB of the storage volume to use for storing input and output data during training (default: 30)."""

    volume_kms_key: str | None = None
    """KMS key ID for encrypting EBS volume attached to the training instance."""

    max_run: int | None = None
    """Timeout in seconds for training (default: 24 * 60 * 60)."""

    input_mode: str | None = None
    """The input mode that the algorithm supports (default: 'File')."""

    output_path: str | None = None
    """S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket. If the bucket with the specific name does not exist, the estimator creates the bucket during the fit() method execution."""

    output_kms_key: str | None = None
    """KMS key ID for encrypting the training output (default: Your IAM role's KMS key for Amazon S3)."""

    base_job_name: str | None = None
    """Prefix for training job name when the fit() method launches. If not specified, the estimator generates a default job name based on the training image name and current timestamp."""

    tags: dict[str, str] = field(default_factory=dict)
    """Dictionary of tags for labeling a training job (e.g., key1:val1,key2:val2)."""

    subnets: list[str] | None = None
    """List of subnet ids. If not specified training job will be created without VPC config."""

    security_group_ids: list[str] | None = None
    """List of security group ids. If not specified training job will be created without VPC config."""

    model_uri: str | None = None
    """URI where a pre-trained model is stored, either locally or in S3."""

    model_channel_name: str | None = None
    """Name of the channel where 'model_uri' will be downloaded (default: 'model')."""

    metric_definitions: dict[str, str] | None = None
    """Dictionary that defines the metric(s) used to evaluate the training jobs. Each key is the metric name and the value is the regular expression used to extract the metric from the logs (e.g., metric_name:regex_pattern,other_metric:other_regex)."""

    encrypt_inter_container_traffic: bool | None = None
    """Specifies whether traffic between training containers is encrypted for the training job (default: False)."""

    use_spot_instances: bool | None = None
    """Specifies whether to use SageMaker Managed Spot instances for training. If enabled then the max_wait arg should also be set."""

    max_wait: int | None = None
    """Timeout in seconds waiting for spot training job."""

    checkpoint_s3_uri: str | None = None
    """S3 URI in which to persist checkpoints that the algorithm persists (if any) during training."""

    checkpoint_local_path: str | None = None
    """Local path that the algorithm writes its checkpoints to."""

    debugger_hook_config: bool | None = None
    """Configuration for how debugging information is emitted with SageMaker Debugger. If not specified, a default one is created using the estimator's output_path, unless the region does not support SageMaker Debugger. To disable SageMaker Debugger, set this parameter to False."""

    enable_sagemaker_metrics: bool | None = None
    """Enable SageMaker Metrics Time Series."""

    enable_network_isolation: bool | None = None
    """Specifies whether container will run in network isolation mode (default: False)."""

    disable_profiler: bool | None = None
    """Specifies whether Debugger monitoring and profiling will be disabled (default: False)."""

    environment: dict[str, str] | None = None
    """Environment variables to be set for use during training job."""

    max_retry_attempts: int | None = None
    """Number of times to move a job to the STARTING status. You can specify between 1 and 30 attempts."""

    source_dir: str | None = None
    """Absolute, relative, or S3 URI Path to a directory with any other training source code dependencies aside from the entry point file (default: current working directory)."""

    git_config: dict[str, str] | None = None
    """Git configurations used for cloning files, including repo, branch, commit, 2FA_enabled, username, password, and token."""

    hyperparameters: dict[str, str] | None = None
    """Dictionary containing the hyperparameters to initialize this estimator with."""

    container_log_level: int | None = None
    """Log level to use within the container (default: logging.INFO)."""

    code_location: str | None = None
    """S3 prefix URI where custom code is uploaded."""

    dependencies: list[str] | None = None
    """List of absolute or relative paths to directories with any additional libraries that should be exported to the container."""

    training_repository_access_mode: str | None = None
    """Specifies how SageMaker accesses the Docker image that contains the training algorithm."""

    training_repository_credentials_provider_arn: str | None = None
    """Amazon Resource Name (ARN) of an AWS Lambda function that provides credentials to authenticate to the private Docker registry where your training image is hosted."""

    disable_output_compression: bool | None = None
    """When set to true, Model is uploaded to Amazon S3 without compression after training finishes."""

    enable_infra_check: bool | None = None
    """Specifies whether it is running Sagemaker built-in infra check jobs."""


AWSSageMakerOpts = Opts


@dataclass
class AWSSageMakerJob:
    """
    Jobs defined the key values that is required to schedule a job. This will be the value
    of `request` in the AppDryRunInfo object.

    - job_name: defines the job name shown in SageMaker
    - job_def: defines the job description that will be used to schedule the job on SageMaker
    - images_to_push: used by torchx to push to image_repo
    """

    job_name: str
    job_def: dict[str, Any]
    images_to_push: dict[str, tuple[str, str]]

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
    src: dict[str, str] | None, extra: dict[str, str]
) -> OrderedDict[str, str]:
    merged = OrdDict(src or {})
    merged.update(extra)
    return merged


class AWSSageMakerScheduler(
    DockerWorkspaceMixin,
    Scheduler[Opts],
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
        client: Any | None = None,  # pyre-ignore[2]
        docker_client: "DockerClient | None" = None,
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

    def _submit_dryrun(self, app: AppDef, cfg: Opts) -> AppDryRunInfo[AWSSageMakerJob]:
        role = app.roles[0]
        entrypoint, hyperparameters = self._parse_args(role.args)

        opts = Opts.from_cfg(cfg)

        # map any local images to the remote image
        images_to_push = self.dryrun_push_images(app, opts)
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
        job_def: dict[str, object] = {
            "entry_point": entrypoint,
            "image_uri": role.image,
            "distribution": {"torch_distributed": {"enabled": True}},
        }

        # Merge environment and hyperparameters with role values
        merged_env = _merge_ordered(opts.environment, role.env)
        # hyperparameters are used for both script/module entrypoint args and the values from .torchxconfig
        # order matters, adding script args last to handle wildcard parameters
        merged_hp = _merge_ordered(opts.hyperparameters, hyperparameters)
        # following the principle of least astonishment defaulting source_dir to current working directory
        resolved_source_dir = opts.source_dir or os.getcwd()

        for key in opts:
            if key == "tags":
                # tags are used for AppDef metadata and the values from .torchxconfig
                # Convert dict[str, str] to SageMaker's list[dict[str, str]] format
                job_def["tags"] = [
                    *({"Key": k, "Value": v} for k, v in (opts.tags or {}).items()),
                    *({"Key": k, "Value": v} for k, v in app.metadata.items()),
                ]
            elif key == "environment":
                job_def["environment"] = merged_env
            elif key == "hyperparameters":
                job_def["hyperparameters"] = merged_hp
            elif key == "source_dir":
                job_def["source_dir"] = resolved_source_dir
            else:
                if key in job_def:
                    raise ValueError(
                        f"{key} is controlled by aws_sagemaker_scheduler and is set to {job_def[key]}"
                    )
                value = getattr(opts, key)
                if value is not None:
                    job_def[key] = value

        req = AWSSageMakerJob(
            job_name=job_name,
            job_def=job_def,
            images_to_push=images_to_push,
        )
        return AppDryRunInfo(req, repr)

    def _parse_args(self, args: list[str]) -> tuple[str, dict[str, str]]:
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
        return Opts.as_runopts()

    def describe(self, app_id: str) -> DescribeAppResponse | None:
        job = self._get_job(app_id)
        if job is None:
            return None

        return DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job["TrainingJobStatus"]],
            ui_url=self._job_ui_url(job["TrainingJobArn"]),
        )

    def list(self, cfg: Mapping[str, CfgVal] | None = None) -> list[ListAppResponse]:
        raise NotImplementedError()

    def _cancel_existing(self, app_id: str) -> None:
        self._client.stop_training_job(TrainingJobName=app_id)

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        should_tail: bool = False,
        streams: Stream | None = None,
    ) -> Iterable[str]:
        raise NotImplementedError()

    def _get_job(self, app_id: str) -> dict[str, Any] | None:
        job = self._client.describe_training_job(TrainingJobName=app_id)
        return job

    def _job_ui_url(self, job_arn: str) -> str | None:
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
