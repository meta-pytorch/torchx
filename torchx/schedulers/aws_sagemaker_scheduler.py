#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
import re
import threading
from collections import OrderedDict as OrdDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, OrderedDict, TYPE_CHECKING, TypeVar

import boto3
import yaml
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

    # pyre-fixme[21]: Could not find module `sagemaker.train`.
    from sagemaker.train import ModelTrainer  # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)

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
    """S3 location for saving the training result (model artifacts and output files). If not specified, results are stored to a default bucket."""

    output_kms_key: str | None = None
    """KMS key ID for encrypting the training output (default: Your IAM role's KMS key for Amazon S3)."""

    base_job_name: str | None = None
    """Prefix for training job name when the train() method launches. If not specified, the trainer generates a default job name based on the training image name and current timestamp."""

    tags: dict[str, str] = field(default_factory=dict)
    """Dictionary of tags for labeling a training job (e.g., key1:val1,key2:val2)."""

    subnets: list[str] | None = None
    """List of subnet ids. If not specified training job will be created without VPC config."""

    security_group_ids: list[str] | None = None
    """List of security group ids. If not specified training job will be created without VPC config."""

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

    enable_network_isolation: bool | None = None
    """Specifies whether container will run in network isolation mode (default: False)."""

    environment: dict[str, str] | None = None
    """Environment variables to be set for use during training job."""

    max_retry_attempts: int | None = None
    """Number of times to move a job to the STARTING status. You can specify between 1 and 30 attempts."""

    source_dir: str | None = None
    """Absolute, relative, or S3 URI Path to a directory with any other training source code dependencies aside from the entry point file (default: current working directory)."""

    hyperparameters: dict[str, str] | None = None
    """Dictionary containing the hyperparameters to initialize this estimator with."""

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


# pyre-fixme[11]: Annotation `ModelTrainer` is not defined as a type.
def _build_model_trainer(job_def: dict[str, Any]) -> "ModelTrainer":
    """Construct a ModelTrainer from a serializable job_def dict."""
    # Lazy imports: sagemaker v3 modules are unavailable in fbcode third-party
    # (which bundles sagemaker v2). This scheduler is OSS-only, so we defer
    # the imports to runtime when actually building a trainer.
    # pyre-fixme[21]: Could not find module `sagemaker.core.shapes.shapes`.
    from sagemaker.core.shapes.shapes import (
        CheckpointConfig,
        InfraCheckConfig,
        MetricDefinition,
        OutputDataConfig,
        RetryStrategy,
        StoppingCondition,
        Tag,
        TrainingImageConfig,
        TrainingRepositoryAuthConfig,
    )

    # pyre-fixme[21]: Could not find module `sagemaker.core.training.configs`.
    from sagemaker.core.training.configs import Compute, Networking, SourceCode
    from sagemaker.train import ModelTrainer

    # pyre-fixme[21]: Could not find module `sagemaker.train.distributed`.
    from sagemaker.train.distributed import Torchrun

    kwargs: dict[str, Any] = {}

    # Required fields
    kwargs["training_image"] = job_def["training_image"]
    kwargs["role"] = job_def.get("role")

    # SourceCode
    source_code_def = job_def.get("source_code")
    if source_code_def:
        kwargs["source_code"] = SourceCode(
            entry_script=source_code_def.get("entry_script"),
            source_dir=source_code_def.get("source_dir"),
        )

    # Compute
    compute_def = job_def.get("compute")
    if compute_def:
        kwargs["compute"] = Compute(
            instance_type=compute_def.get("instance_type"),
            instance_count=compute_def.get("instance_count"),
            volume_size_in_gb=compute_def.get("volume_size_in_gb"),
            volume_kms_key_id=compute_def.get("volume_kms_key_id"),
            keep_alive_period_in_seconds=compute_def.get(
                "keep_alive_period_in_seconds"
            ),
            enable_managed_spot_training=compute_def.get(
                "enable_managed_spot_training"
            ),
        )

    # Distributed
    if job_def.get("distributed"):
        kwargs["distributed"] = Torchrun()

    # Networking
    networking_def = job_def.get("networking")
    if networking_def:
        kwargs["networking"] = Networking(
            subnets=networking_def.get("subnets"),
            security_group_ids=networking_def.get("security_group_ids"),
            enable_network_isolation=networking_def.get("enable_network_isolation"),
            enable_inter_container_traffic_encryption=networking_def.get(
                "enable_inter_container_traffic_encryption"
            ),
        )

    # StoppingCondition
    stopping_def = job_def.get("stopping_condition")
    if stopping_def:
        kwargs["stopping_condition"] = StoppingCondition(
            max_runtime_in_seconds=stopping_def.get("max_runtime_in_seconds"),
            max_wait_time_in_seconds=stopping_def.get("max_wait_time_in_seconds"),
        )

    # OutputDataConfig
    output_def = job_def.get("output_data_config")
    if output_def:
        kwargs["output_data_config"] = OutputDataConfig(
            s3_output_path=output_def["s3_output_path"],
            kms_key_id=output_def.get("kms_key_id"),
            compression_type=output_def.get("compression_type"),
        )

    # CheckpointConfig
    checkpoint_def = job_def.get("checkpoint_config")
    if checkpoint_def:
        kwargs["checkpoint_config"] = CheckpointConfig(
            s3_uri=checkpoint_def["s3_uri"],
            local_path=checkpoint_def.get("local_path"),
        )

    # TrainingImageConfig
    image_config_def = job_def.get("training_image_config")
    if image_config_def:
        auth_config = None
        auth_arn = image_config_def.get("training_repository_auth_config_arn")
        if auth_arn:
            auth_config = TrainingRepositoryAuthConfig(
                training_repository_credentials_provider_arn=auth_arn,
            )
        kwargs["training_image_config"] = TrainingImageConfig(
            training_repository_access_mode=image_config_def[
                "training_repository_access_mode"
            ],
            training_repository_auth_config=auth_config,
        )

    # Simple pass-through fields
    if job_def.get("training_input_mode") is not None:
        kwargs["training_input_mode"] = job_def["training_input_mode"]
    if job_def.get("environment") is not None:
        kwargs["environment"] = job_def["environment"]
    if job_def.get("hyperparameters") is not None:
        kwargs["hyperparameters"] = job_def["hyperparameters"]
    if job_def.get("base_job_name") is not None:
        kwargs["base_job_name"] = job_def["base_job_name"]

    # Tags
    tags_def = job_def.get("tags")
    if tags_def:
        kwargs["tags"] = [Tag(key=t["key"], value=t["value"]) for t in tags_def]

    trainer = ModelTrainer(**kwargs)

    # Post-construction configuration via builder methods
    metric_defs = job_def.get("metric_definitions")
    if metric_defs:
        trainer = trainer.with_metric_definitions(
            [MetricDefinition(name=name, regex=regex) for name, regex in metric_defs]
        )

    retry_attempts = job_def.get("max_retry_attempts")
    if retry_attempts is not None:
        trainer = trainer.with_retry_strategy(
            RetryStrategy(maximum_retry_attempts=retry_attempts)
        )

    enable_infra_check = job_def.get("enable_infra_check")
    if enable_infra_check is not None:
        trainer = trainer.with_infra_check_config(
            InfraCheckConfig(enable_infra_check=enable_infra_check)
        )

    return trainer


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
        trainer = _build_model_trainer(req.job_def)
        trainer.train(wait=False)

        # ModelTrainer generates the actual job name from base_job_name + timestamp.
        # Read back the real name so describe()/cancel() work correctly.
        if trainer._latest_training_job is not None:
            return trainer._latest_training_job.get_name()
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

        # Merge environment and hyperparameters with role values
        env_val = cfg.get("environment")
        assert env_val is None or isinstance(
            env_val, dict
        ), f"expected environment to be a dict or None, got {type(env_val)}"
        merged_env = _merge_ordered(env_val, role.env)

        hp_val = cfg.get("hyperparameters")
        assert hp_val is None or isinstance(
            hp_val, dict
        ), f"expected hyperparameters to be a dict or None, got {type(hp_val)}"
        merged_hp = _merge_ordered(hp_val, hyperparameters)

        source_dir_val = cfg.get("source_dir")
        assert source_dir_val is None or isinstance(
            source_dir_val, str
        ), f"expected source_dir to be a str or None, got {type(source_dir_val)}"
        resolved_source_dir = source_dir_val or os.getcwd()

        # Build structured job_def for ModelTrainer
        job_def: dict[str, object] = {
            "training_image": role.image,
            "source_code": {
                "entry_script": entrypoint,
                "source_dir": resolved_source_dir,
            },
            # Torchrun distribution is always enabled, matching the v2 behavior
            # (distribution={"torch_distributed": {"enabled": True}}). Even for
            # single-instance jobs, the entrypoint is launched via torchrun.
            "distributed": True,
            "environment": dict(merged_env),
            "hyperparameters": dict(merged_hp),
        }

        # Compute config
        compute: dict[str, object] = {
            "instance_type": cfg.get("instance_type"),
            "instance_count": cfg.get("instance_count"),
        }
        volume_size = cfg.get("volume_size")
        if volume_size is not None:
            compute["volume_size_in_gb"] = volume_size
        volume_kms_key = cfg.get("volume_kms_key")
        if volume_kms_key is not None:
            compute["volume_kms_key_id"] = volume_kms_key
        keep_alive = cfg.get("keep_alive_period_in_seconds")
        if keep_alive is not None:
            compute["keep_alive_period_in_seconds"] = keep_alive
        use_spot = cfg.get("use_spot_instances")
        if use_spot is not None:
            compute["enable_managed_spot_training"] = use_spot
        job_def["compute"] = compute

        # Role
        role_arn = cfg.get("role")
        if role_arn is not None:
            job_def["role"] = role_arn

        # Networking config
        networking: dict[str, object] = {}
        subnets = cfg.get("subnets")
        if subnets is not None:
            networking["subnets"] = subnets
        sg_ids = cfg.get("security_group_ids")
        if sg_ids is not None:
            networking["security_group_ids"] = sg_ids
        net_isolation = cfg.get("enable_network_isolation")
        if net_isolation is not None:
            networking["enable_network_isolation"] = net_isolation
        encrypt_traffic = cfg.get("encrypt_inter_container_traffic")
        if encrypt_traffic is not None:
            networking["enable_inter_container_traffic_encryption"] = encrypt_traffic
        if networking:
            job_def["networking"] = networking

        # Stopping condition
        stopping: dict[str, object] = {}
        max_run = cfg.get("max_run")
        if max_run is not None:
            stopping["max_runtime_in_seconds"] = max_run
        max_wait = cfg.get("max_wait")
        if max_wait is not None:
            stopping["max_wait_time_in_seconds"] = max_wait
        if stopping:
            job_def["stopping_condition"] = stopping

        # Output data config
        output_data: dict[str, object] = {}
        output_path = cfg.get("output_path")
        if output_path is not None:
            output_data["s3_output_path"] = output_path
        output_kms = cfg.get("output_kms_key")
        if output_kms is not None:
            output_data["kms_key_id"] = output_kms
        if cfg.get("disable_output_compression"):
            output_data["compression_type"] = "NONE"
        if output_data:
            if "s3_output_path" not in output_data:
                # s3_output_path is required by OutputDataConfig; skip config
                # and warn instead of passing an empty string
                logger.warning(
                    "`output_kms_key` or `disable_output_compression` set without"
                    " `output_path`; ignoring output data config"
                )
            else:
                job_def["output_data_config"] = output_data

        # Checkpoint config
        ckpt_s3 = cfg.get("checkpoint_s3_uri")
        if ckpt_s3 is not None:
            checkpoint: dict[str, object] = {"s3_uri": ckpt_s3}
            ckpt_local = cfg.get("checkpoint_local_path")
            if ckpt_local is not None:
                checkpoint["local_path"] = ckpt_local
            job_def["checkpoint_config"] = checkpoint

        # Training image config
        repo_access_mode = cfg.get("training_repository_access_mode")
        if repo_access_mode is not None:
            image_config: dict[str, object] = {
                "training_repository_access_mode": repo_access_mode,
            }
            repo_creds_arn = cfg.get("training_repository_credentials_provider_arn")
            if repo_creds_arn is not None:
                image_config["training_repository_auth_config_arn"] = repo_creds_arn
            job_def["training_image_config"] = image_config

        # Training input mode
        input_mode = cfg.get("input_mode")
        if input_mode is not None:
            job_def["training_input_mode"] = input_mode

        # Base job name: use cfg override or TorchX-generated name
        base_job_name = cfg.get("base_job_name") or job_name
        job_def["base_job_name"] = base_job_name

        # Tags: convert dict[str, str] to list of {key, value} dicts
        tags_val = cfg.get("tags")
        assert tags_val is None or isinstance(
            tags_val, dict
        ), f"expected tags to be a dict or None, got {type(tags_val)}"
        tag_list = [
            *({"key": k, "value": v} for k, v in (tags_val or {}).items()),
            *({"key": k, "value": v} for k, v in app.metadata.items()),
        ]
        if tag_list:
            job_def["tags"] = tag_list

        # Metric definitions: stored as list of [name, regex] pairs
        metric_defs = cfg.get("metric_definitions")
        if metric_defs is not None:
            assert isinstance(
                metric_defs, dict
            ), f"expected metric_definitions to be a dict or None, got {type(metric_defs)}"
            job_def["metric_definitions"] = [[k, v] for k, v in metric_defs.items()]

        # Retry strategy
        retry_attempts = cfg.get("max_retry_attempts")
        if retry_attempts is not None:
            job_def["max_retry_attempts"] = retry_attempts

        # Infra check
        infra_check = cfg.get("enable_infra_check")
        if infra_check is not None:
            job_def["enable_infra_check"] = infra_check

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
