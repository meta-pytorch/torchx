#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""

This contains the TorchX Kubernetes scheduler which can be used to run TorchX
components on a Kubernetes cluster.

Prerequisites
==============

The TorchX Kubernetes scheduler depends on Volcano. If you're trying to do an
upgrade you'll need to completely remove all non-Job Volcano resources and recreate.

Install Volcano:

.. code:: bash

    kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.6.0/installer/volcano-development.yaml

See the
`Volcano Quickstart <https://github.com/volcano-sh/volcano>`_
for more information.

Pod Overlay
===========

You can overlay arbitrary Kubernetes Pod fields on generated pods by setting
the ``kubernetes`` metadata on your role. The value can be:

- A dict with the overlay structure
- A resource URI pointing to a YAML file (e.g. ``file://``, ``s3://``, ``gs://``)

Merge semantics:
- **dict**: recursive merge (upsert)
- **list**: append by default, replace if tuple (Python) or ``!!python/tuple`` tag (YAML)
- **primitives**: replace

.. code:: python

    from torchx.specs import Role

    # Dict overlay - lists append, tuples replace
    role = Role(
        name="trainer",
        image="my-image:latest",
        entrypoint="train.py",
        metadata={
            "kubernetes": {
                "spec": {
                    "nodeSelector": {"gpu": "true"},
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],  # appends
                    "volumes": ({"name": "my-volume", "emptyDir": {}},)  # replaces
                }
            }
        }
    )

    # File URI overlay
    role = Role(
        name="trainer",
        image="my-image:latest",
        entrypoint="train.py",
        metadata={
            "kubernetes": "file:///path/to/pod_overlay.yaml"
        }
    )

CLI usage with builtin components:

.. code:: bash

    $ torchx run --scheduler kubernetes dist.ddp \\
        --metadata kubernetes=file:///path/to/pod_overlay.yaml \\
        --script train.py

Example ``pod_overlay.yaml``:

.. code:: yaml

    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: p4d.24xlarge
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      volumes: !!python/tuple
        - name: my-volume
          emptyDir: {}

The overlay is deep-merged with the generated pod, preserving existing fields
and adding or overriding specified ones.
"""

import json
import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    Union,
)

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
from torchx.schedulers.ids import make_unique
from torchx.specs.api import (
    AppDef,
    AppDryRunInfo,
    AppState,
    BindMount,
    CfgVal,
    DeviceMount,
    macros,
    ReplicaState,
    ReplicaStatus,
    RetryPolicy,
    Role,
    RoleStatus,
    runopts,
    VolumeMount,
)
from torchx.util.strings import normalize_str
from torchx.workspace.docker_workspace import DockerWorkspaceMixin


if TYPE_CHECKING:
    from docker import DockerClient
    from kubernetes.client import ApiClient, CustomObjectsApi
    from kubernetes.client.models import (  # noqa: F401 imported but unused
        V1Container,
        V1Pod,
    )
    from kubernetes.client.rest import ApiException

logger: logging.Logger = logging.getLogger(__name__)

# Kubernetes reserves a small amount of resources per host for the system. For
# TorchX we always assume the entire host is being requested so we adjust the
# requested numbers account for the node reserved resources.
#
# https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/
RESERVED_MILLICPU = 100
RESERVED_MEMMB = 1024


def _apply_pod_overlay(pod: "V1Pod", overlay: Dict[str, Any]) -> None:
    """Apply overlay dict to V1Pod object, merging nested fields.

    Merge semantics:
    - dict: upsert (recursive merge)
    - list: append by default, replace if tuple
    - primitives: replace
    """
    from kubernetes import client

    api = client.ApiClient()
    pod_dict = api.sanitize_for_serialization(pod)

    def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        for key, value in overlay.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_merge(base[key], value)
            elif isinstance(value, tuple):
                base[key] = list(value)
            elif (
                isinstance(value, list) and key in base and isinstance(base[key], list)
            ):
                base[key].extend(value)
            else:
                base[key] = value

    deep_merge(pod_dict, overlay)

    merged_pod = api._ApiClient__deserialize(pod_dict, "V1Pod")
    pod.spec = merged_pod.spec
    pod.metadata = merged_pod.metadata


RETRY_POLICIES: Mapping[str, Iterable[Mapping[str, str]]] = {
    RetryPolicy.REPLICA: [],
    RetryPolicy.APPLICATION: [
        {"event": "PodEvicted", "action": "RestartJob"},
        {"event": "PodFailed", "action": "RestartJob"},
    ],
}

JOB_STATE: Dict[str, AppState] = {
    # Pending is the phase that job is pending in the queue, waiting for
    # scheduling decision
    "Pending": AppState.PENDING,
    # Aborting is the phase that job is aborted, waiting for releasing pods
    "Aborting": AppState.RUNNING,
    # Aborted is the phase that job is aborted by user or error handling
    "Aborted": AppState.CANCELLED,
    # Running is the phase that minimal available tasks of Job are running
    "Running": AppState.RUNNING,
    # Restarting is the phase that the Job is restarted, waiting for pod
    # releasing and recreating
    "Restarting": AppState.RUNNING,
    # Completed is the phase that all tasks of Job are completed successfully
    "Completed": AppState.SUCCEEDED,
    # Terminating is the phase that the Job is terminated, waiting for releasing
    # pods
    "Terminating": AppState.RUNNING,
    # Teriminated is the phase that the job is finished unexpected, e.g. events
    "Terminated": AppState.FAILED,
    "Failed": ReplicaState.FAILED,
}

TASK_STATE: Dict[str, ReplicaState] = {
    # Pending means the task is pending in the apiserver.
    "Pending": ReplicaState.PENDING,
    # Allocated means the scheduler assigns a host to it.
    "Allocated": ReplicaState.PENDING,
    # Pipelined means the scheduler assigns a host to wait for releasing
    # resource.
    "Pipelined": ReplicaState.PENDING,
    # Binding means the scheduler send Bind request to apiserver.
    "Binding": ReplicaState.PENDING,
    # Bound means the task/Pod bounds to a host.
    "Bound": ReplicaState.PENDING,
    # Running means a task is running on the host.
    "Running": ReplicaState.RUNNING,
    # Releasing means a task/pod is deleted.
    "Releasing": ReplicaState.RUNNING,
    # Succeeded means that all containers in the pod have voluntarily
    # terminated with a container exit code of 0, and the system is not
    # going to restart any of these containers.
    "Succeeded": ReplicaState.SUCCEEDED,
    # Failed means that all containers in the pod have terminated, and at
    # least one container has terminated in a failure (exited with a
    # non-zero exit code or was stopped by the system).
    "Failed": ReplicaState.FAILED,
    # Unknown means the status of task/pod is unknown to the scheduler.
    "Unknown": ReplicaState.UNKNOWN,
}

LABEL_VERSION = "torchx.pytorch.org/version"
LABEL_APP_NAME = "torchx.pytorch.org/app-name"
LABEL_ROLE_INDEX = "torchx.pytorch.org/role-index"
LABEL_ROLE_NAME = "torchx.pytorch.org/role-name"
LABEL_REPLICA_ID = "torchx.pytorch.org/replica-id"
LABEL_KUBE_APP_NAME = "app.kubernetes.io/name"
LABEL_ORGANIZATION = "app.kubernetes.io/managed-by"
LABEL_UNIQUE_NAME = "app.kubernetes.io/instance"

ANNOTATION_ISTIO_SIDECAR = "sidecar.istio.io/inject"

LABEL_INSTANCE_TYPE = "node.kubernetes.io/instance-type"

# role.env translates to static env variables in the yaml
# {"FOO" : "bar"}               =====>      - name: FOO
#                                             value: bar
# unless this placeholder is present at the start of the role.env value then the env variable
# in the yaml will be dynamically populated at runtime (placeholder is stripped out of the value)
# {"FOO" : "[FIELD_PATH]bar"}   =====>      - name: FOO
#                                             valueFrom:
#                                               fieldRef:
#                                                 fieldPath: bar
PLACEHOLDER_FIELD_PATH = "[FIELD_PATH]"


def sanitize_for_serialization(obj: object) -> object:
    from kubernetes import client

    api = client.ApiClient()
    return api.sanitize_for_serialization(obj)


def role_to_pod(name: str, role: Role, service_account: Optional[str]) -> "V1Pod":
    from kubernetes.client.models import (  # noqa: F811 redefinition of unused
        V1Container,
        V1ContainerPort,
        V1EmptyDirVolumeSource,
        V1EnvVar,
        V1EnvVarSource,
        V1HostPathVolumeSource,
        V1ObjectFieldSelector,
        V1ObjectMeta,
        V1PersistentVolumeClaimVolumeSource,
        V1Pod,
        V1PodSpec,
        V1ResourceRequirements,
        V1SecurityContext,
        V1Volume,
        V1VolumeMount,
    )

    # limits puts an upper cap on the resources a pod may consume.
    # requests is how much the scheduler allocates. We assume that the jobs will
    # be allocation the whole machine so requests is slightly lower than the
    # requested resources to account for the Kubernetes node reserved resources.
    limits = {}
    requests = {}

    resource = role.resource
    if resource.cpu > 0:
        mcpu = int(resource.cpu * 1000)
        limits["cpu"] = f"{mcpu}m"
        request_mcpu = max(mcpu - RESERVED_MILLICPU, 0)
        requests["cpu"] = f"{request_mcpu}m"
    if resource.memMB > 0:
        limits["memory"] = f"{int(resource.memMB)}M"
        request_memMB = max(int(resource.memMB) - RESERVED_MEMMB, 0)
        requests["memory"] = f"{request_memMB}M"
    if resource.gpu > 0:
        requests["nvidia.com/gpu"] = limits["nvidia.com/gpu"] = str(resource.gpu)

    for device_name, device_limit in resource.devices.items():
        limits[device_name] = str(device_limit)

    resources = V1ResourceRequirements(
        limits=limits,
        requests=requests,
    )

    node_selector: Dict[str, str] = {}
    if LABEL_INSTANCE_TYPE in resource.capabilities:
        node_selector[LABEL_INSTANCE_TYPE] = resource.capabilities[LABEL_INSTANCE_TYPE]

    # To support PyTorch dataloaders we need to set /dev/shm to larger than the
    # 64M default so we mount an unlimited sized tmpfs directory on it.
    SHM_VOL = "dshm"
    volumes = [
        V1Volume(
            name=SHM_VOL,
            empty_dir=V1EmptyDirVolumeSource(
                medium="Memory",
            ),
        ),
    ]
    volume_mounts = [
        V1VolumeMount(name=SHM_VOL, mount_path="/dev/shm"),
    ]
    security_context = V1SecurityContext()

    for i, mount in enumerate(role.mounts):
        mount_name = f"mount-{i}"
        if isinstance(mount, BindMount):
            volumes.append(
                V1Volume(
                    name=mount_name,
                    host_path=V1HostPathVolumeSource(
                        path=mount.src_path,
                    ),
                )
            )
            volume_mounts.append(
                V1VolumeMount(
                    name=mount_name,
                    mount_path=mount.dst_path,
                    read_only=mount.read_only,
                )
            )
        elif isinstance(mount, VolumeMount):
            volumes.append(
                V1Volume(
                    name=mount_name,
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name=mount.src,
                    ),
                )
            )
            volume_mounts.append(
                V1VolumeMount(
                    name=mount_name,
                    mount_path=mount.dst_path,
                    read_only=mount.read_only,
                )
            )
        elif isinstance(mount, DeviceMount):
            volumes.append(
                V1Volume(
                    name=mount_name,
                    host_path=V1HostPathVolumeSource(
                        path=mount.src_path,
                    ),
                )
            )
            volume_mounts.append(
                V1VolumeMount(
                    name=mount_name,
                    mount_path=mount.dst_path,
                    read_only=(
                        "w" not in mount.permissions and "m" not in mount.permissions
                    ),
                )
            )
            security_context.privileged = True
        else:
            raise TypeError(f"unknown mount type {mount}")

    container = V1Container(
        command=[role.entrypoint] + role.args,
        image=role.image,
        name=name,
        env=[
            (
                V1EnvVar(
                    name=name,
                    value_from=V1EnvVarSource(
                        field_ref=V1ObjectFieldSelector(
                            field_path=value.strip(PLACEHOLDER_FIELD_PATH)
                        )
                    ),
                )
                if value.startswith(PLACEHOLDER_FIELD_PATH)
                else V1EnvVar(
                    name=name,
                    value=value,
                )
            )
            for name, value in role.env.items()
        ],
        resources=resources,
        ports=[
            V1ContainerPort(
                name=name,
                container_port=port,
            )
            for name, port in role.port_map.items()
        ],
        volume_mounts=volume_mounts,
        security_context=security_context,
    )

    return V1Pod(
        spec=V1PodSpec(
            containers=[container],
            restart_policy="Never",
            service_account_name=service_account,
            volumes=volumes,
            node_selector=node_selector,
        ),
        metadata=V1ObjectMeta(
            annotations={
                # Disable the istio sidecar as it prevents the containers from
                # exiting once finished.
                ANNOTATION_ISTIO_SIDECAR: "false",
            },
            labels={},
        ),
    )


def app_to_resource(
    app: AppDef,
    queue: str,
    service_account: Optional[str],
    priority_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    app_to_resource creates a volcano job kubernetes resource definition from
    the provided AppDef. The resource definition can be used to launch the
    app on Kubernetes.

    To support macros we generate one task per replica instead of using the
    volcano `replicas` field since macros change the arguments on a per
    replica basis.

    Volcano has two levels of retries: one at the task level and one at the
    job level. When using the APPLICATION retry policy, the job level retry
    count is set to the minimum of the max_retries of the roles.
    """
    tasks = []
    unique_app_id = normalize_str(make_unique(app.name))
    for role_idx, role in enumerate(app.roles):
        for replica_id in range(role.num_replicas):
            values = macros.Values(
                img_root="",
                app_id=unique_app_id,
                replica_id=str(replica_id),
                rank0_env=f"VC_{normalize_str(app.roles[0].name)}_0_HOSTS".upper(),
            )
            if role_idx == 0 and replica_id == 0:
                values.rank0_env = "TORCHX_RANK0_HOST"
            name = normalize_str(f"{role.name}-{replica_id}")
            replica_role = values.apply(role)
            if role_idx == 0 and replica_id == 0:
                replica_role.env["TORCHX_RANK0_HOST"] = "localhost"
            replica_role.env["TORCHX_IMAGE"] = replica_role.image

            pod = role_to_pod(name, replica_role, service_account)
            if k8s_metadata := role.metadata.get("kubernetes"):
                if isinstance(k8s_metadata, str):
                    import fsspec

                    with fsspec.open(k8s_metadata, "r") as f:
                        k8s_metadata = yaml.unsafe_load(f)
                elif not isinstance(k8s_metadata, dict):
                    raise ValueError(
                        f"metadata['kubernetes'] must be a dict or resource URI, got {type(k8s_metadata)}"
                    )
                _apply_pod_overlay(pod, k8s_metadata)
            pod.metadata.labels.update(
                pod_labels(
                    app=app,
                    role_idx=role_idx,
                    role=role,
                    replica_id=replica_id,
                    app_id=unique_app_id,
                )
            )
            task: Dict[str, Any] = {
                "replicas": 1,
                "name": name,
                "template": pod,
            }
            if role.max_retries > 0:
                task["maxRetry"] = role.max_retries
                task["policies"] = RETRY_POLICIES[role.retry_policy]
                msg = f"""
Role {role.name} configured with restarts: {role.max_retries}. As of 1.4.0 Volcano
does NOT support retries correctly. More info: https://github.com/volcano-sh/volcano/issues/1651
                """
                warnings.warn(msg)
            if role.min_replicas is not None:
                # first min_replicas tasks are required, afterward optional
                task["minAvailable"] = 1 if replica_id < role.min_replicas else 0
            tasks.append(task)

    job_retries = min(role.max_retries for role in app.roles)
    job_spec = {
        "schedulerName": "volcano",
        "queue": queue,
        "tasks": tasks,
        "maxRetry": job_retries,
        "plugins": {
            # https://github.com/volcano-sh/volcano/issues/533
            "svc": ["--publish-not-ready-addresses"],
            "env": [],
        },
    }
    if priority_class is not None:
        job_spec["priorityClassName"] = priority_class

    resource: Dict[str, Any] = {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {"name": f"{unique_app_id}"},
        "spec": job_spec,
    }
    return resource


@dataclass
class KubernetesJob:
    images_to_push: Dict[str, Tuple[str, str]]
    resource: Dict[str, Any]

    def __str__(self) -> str:
        return yaml.dump(sanitize_for_serialization(self.resource))

    def __repr__(self) -> str:
        return str(self)


class KubernetesOpts(TypedDict, total=False):
    namespace: Optional[str]
    queue: str
    image_repo: Optional[str]
    service_account: Optional[str]
    priority_class: Optional[str]
    validate_spec: Optional[bool]


class KubernetesScheduler(
    DockerWorkspaceMixin,
    Scheduler[KubernetesOpts, AppDef, AppDryRunInfo[KubernetesJob]],
):
    """
    KubernetesScheduler is a TorchX scheduling interface to Kubernetes.

    Important: Volcano is required to be installed on the Kubernetes cluster.
    TorchX requires gang scheduling for multi-replica/multi-role execution
    and Volcano is currently the only supported scheduler with Kubernetes.
    For installation instructions see: https://github.com/volcano-sh/volcano

    This has been confirmed to work with Volcano v1.3.0 and Kubernetes versions
    v1.18-1.21. See https://github.com/meta-pytorch/torchx/issues/120 which is
    tracking Volcano support for Kubernetes v1.22.

    .. note::

        AppDefs that have more than 0 retries may not be displayed as pods if they failed.
        This occurs due to known bug in Volcano(as per 1.4.0 release):
        https://github.com/volcano-sh/volcano/issues/1651


    .. code-block:: bash

        $ pip install torchx[kubernetes]
        $ torchx run --scheduler kubernetes --scheduler_args namespace=default,queue=test utils.echo --image alpine:latest --msg hello
        kubernetes://torchx_user/1234
        $ torchx status kubernetes://torchx_user/1234
        ...

    **Config Options**

    .. runopts::
        class: torchx.schedulers.kubernetes_scheduler.create_scheduler

    **Mounts**

    Mounting external filesystems/volumes is via the HostPath and
    PersistentVolumeClaim support.

    * hostPath volumes: ``type=bind,src=<host path>,dst=<container path>[,readonly]``
    * PersistentVolumeClaim: ``type=volume,src=<claim>,dst=<container path>[,readonly]``
    * host devices: ``type=device,src=/dev/foo[,dst=<container path>][,perm=rwm]``
      If you specify a host device the job will run in privileged mode since
      Kubernetes doesn't expose a way to pass `--device` to the underlying
      container runtime. Users should prefer to use device plugins.

    See :py:func:`torchx.specs.parse_mounts` for more info.

    External docs: https://kubernetes.io/docs/concepts/storage/persistent-volumes/

    **Resources / Allocation**

    To select a specific machine type you can add a capability to your resources
    with ``node.kubernetes.io/instance-type`` which will constrain the launched
    jobs to nodes of that instance type.

    >>> from torchx import specs
    >>> specs.Resource(
    ...     cpu=4,
    ...     memMB=16000,
    ...     gpu=2,
    ...     capabilities={
    ...         "node.kubernetes.io/instance-type": "<cloud instance type>",
    ...     },
    ... )
    Resource(...)

    Kubernetes may reserve some memory for the host. TorchX assumes you're
    scheduling on whole hosts and thus will automatically reduce the resource
    request by a small amount to account for the node reserved CPU and memory.
    If you run into scheduling issues you may need to reduce the requested CPU
    and memory from the host values.

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. KubernetesScheduler will return job and replica
                status but does not provide the complete original AppSpec.
            workspaces: true
            mounts: true
            elasticity: Requires Volcano >1.6
    """

    def __init__(
        self,
        session_name: str,
        client: Optional["ApiClient"] = None,
        docker_client: Optional["DockerClient"] = None,
    ) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("kubernetes", session_name, docker_client=docker_client)

        self._client = client

    def _api_client(self) -> "ApiClient":
        from kubernetes import client, config

        c = self._client
        if c is None:
            configuration = client.Configuration()
            try:
                config.load_kube_config(client_configuration=configuration)
            except config.ConfigException as e:
                warnings.warn(f"failed to load kube config: {e}")

            c = self._client = client.ApiClient(configuration)

        return c

    def _custom_objects_api(self) -> "CustomObjectsApi":
        from kubernetes import client

        return client.CustomObjectsApi(self._api_client())

    def _get_job_name_from_exception(self, e: "ApiException") -> Optional[str]:
        try:
            return json.loads(e.body)["details"]["name"]
        except Exception as e:
            logger.exception("Unable to retrieve job name, got exception", e)
            return None

    def _get_active_context(self) -> Dict[str, Any]:
        from kubernetes import config

        contexts, active_context = config.list_kube_config_contexts()
        return active_context

    def schedule(self, dryrun_info: AppDryRunInfo[KubernetesJob]) -> str:
        from kubernetes.client.rest import ApiException

        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"
        namespace = cfg.get("namespace") or "default"

        images_to_push = dryrun_info.request.images_to_push
        self.push_images(images_to_push)

        resource = dryrun_info.request.resource
        try:
            resp = self._custom_objects_api().create_namespaced_custom_object(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=namespace,
                plural="jobs",
                body=resource,
            )
        except ApiException as e:
            if e.status == 409 and e.reason == "Conflict":
                job_name = self._get_job_name_from_exception(e)
                raise ValueError(
                    f"Job `{job_name}` already exists. This seems like a transient exception, try resubmitting job"
                ) from e
            else:
                raise

        return f"{namespace}:{resp['metadata']['name']}"

    def _submit_dryrun(
        self, app: AppDef, cfg: KubernetesOpts
    ) -> AppDryRunInfo[KubernetesJob]:
        queue = cfg.get("queue")
        if not isinstance(queue, str):
            raise TypeError(f"config value 'queue' must be a string, got {queue}")

        # map any local images to the remote image
        images_to_push = self.dryrun_push_images(app, cast(Mapping[str, CfgVal], cfg))

        service_account = cfg.get("service_account")
        assert service_account is None or isinstance(
            service_account, str
        ), "service_account must be a str"

        priority_class = cfg.get("priority_class")
        assert priority_class is None or isinstance(
            priority_class, str
        ), "priority_class must be a str"

        resource = app_to_resource(app, queue, service_account, priority_class)

        if cfg.get("validate_spec"):
            try:
                self._custom_objects_api().create_namespaced_custom_object(
                    group="batch.volcano.sh",
                    version="v1alpha1",
                    namespace=cfg.get("namespace") or "default",
                    plural="jobs",
                    body=resource,
                    dry_run="All",
                )
            except Exception as e:
                from kubernetes.client.rest import ApiException

                if isinstance(e, ApiException):
                    raise ValueError(f"Invalid job spec: {e.reason}") from e
                raise

            job_name = resource["metadata"]["name"]
            for task in resource["spec"]["tasks"]:
                task_name = task["name"]
                replicas = task.get("replicas", 1)
                max_index = replicas - 1
                pod_name = f"{job_name}-{task_name}-{max_index}"
                if len(pod_name) > 63:
                    raise ValueError(
                        f"Pod name '{pod_name}' ({len(pod_name)} chars) exceeds 63 character limit. "
                        f"Shorten app.name or role names"
                    )

        req = KubernetesJob(
            resource=resource,
            images_to_push=images_to_push,
        )
        return AppDryRunInfo(req, repr)

    def _validate(self, app: AppDef, scheduler: str, cfg: KubernetesOpts) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        namespace, name = app_id.split(":")
        self._custom_objects_api().delete_namespaced_custom_object(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            name=name,
        )

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "namespace",
            type_=str,
            help="Kubernetes namespace to schedule job in",
            default="default",
        )
        opts.add(
            "queue",
            type_=str,
            help="Volcano queue to schedule job in",
            required=True,
        )
        opts.add(
            "service_account",
            type_=str,
            help="The service account name to set on the pod specs",
        )
        opts.add(
            "priority_class",
            type_=str,
            help="The name of the PriorityClass to set on the job specs",
        )
        opts.add(
            "validate_spec",
            type_=bool,
            help="Validate job spec using Kubernetes API dry-run before submission",
            default=True,
        )
        return opts

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        from kubernetes.client.rest import ApiException

        namespace, name = app_id.split(":")
        roles = {}
        roles_statuses = {}
        try:
            resp = self._custom_objects_api().get_namespaced_custom_object_status(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=namespace,
                plural="jobs",
                name=name,
            )
        except ApiException as e:
            if e.status == 404:
                return None
            raise
        status = resp.get("status")
        if status:
            state_str = status["state"]["phase"]
            app_state = JOB_STATE[state_str]

            TASK_STATUS_COUNT = "taskStatusCount"

            if TASK_STATUS_COUNT in status:
                for name, status in status[TASK_STATUS_COUNT].items():
                    role, _, idx = name.rpartition("-")

                    state_str = next(iter(status["phase"].keys()))
                    state = TASK_STATE[state_str]

                    if role not in roles:
                        roles[role] = Role(name=role, num_replicas=0, image="")
                        roles_statuses[role] = RoleStatus(role, [])
                    roles[role].num_replicas += 1
                    roles_statuses[role].replicas.append(
                        ReplicaStatus(id=int(idx), role=role, state=state, hostname="")
                    )
        else:
            app_state = AppState.UNKNOWN
        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=app_state,
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
        assert until is None, "kubernetes API doesn't support until"

        if streams not in (None, Stream.COMBINED):
            raise ValueError("KubernetesScheduler only supports COMBINED log stream")

        from kubernetes import client, watch

        namespace, name = app_id.split(":")

        pod_name = normalize_str(f"{name}-{role_name}-{k}-0")

        args: Dict[str, object] = {
            "name": pod_name,
            "namespace": namespace,
            "timestamps": True,
        }
        if since is not None:
            args["since_seconds"] = (datetime.now() - since).total_seconds()

        core_api = client.CoreV1Api(self._api_client())
        if should_tail:
            w = watch.Watch()
            iterator = w.stream(core_api.read_namespaced_pod_log, **args)
        else:
            resp = core_api.read_namespaced_pod_log(**args)
            iterator = split_lines(resp)

        if regex:
            return filter_regex(regex, iterator)
        else:
            return iterator

    def list(self) -> List[ListAppResponse]:
        active_context = self._get_active_context()
        namespace = active_context["context"]["namespace"]
        resp = self._custom_objects_api().list_namespaced_custom_object(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            timeout_seconds=30,
        )
        return [
            ListAppResponse(
                app_id=f"{namespace}:{app['metadata']['name']}",
                state=JOB_STATE[app["status"]["state"]["phase"]],
            )
            for app in resp["items"]
        ]


def create_scheduler(
    session_name: str,
    client: Optional["ApiClient"] = None,
    docker_client: Optional["DockerClient"] = None,
    **kwargs: Any,
) -> KubernetesScheduler:
    return KubernetesScheduler(
        session_name=session_name,
        client=client,
        docker_client=docker_client,
    )


def pod_labels(
    app: AppDef, role_idx: int, role: Role, replica_id: int, app_id: str
) -> Dict[str, str]:

    def clean(label_value: str) -> str:
        # cleans the provided `label_value` to make it compliant
        # to pod label specs as described in
        # https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
        #
        # Valid label value:
        # must be 63 characters or less (can be empty),
        # unless empty, must begin and end with an alphanumeric character ([a-z0-9A-Z]),
        # could contain dashes (-), underscores (_), dots (.), and alphanumerics between.

        # Replace invalid characters (allow: alphanum, -, _, .) with "."
        label_value = re.sub(r"[^A-Za-z0-9\-_.]", ".", label_value)
        # Replace leading non-alphanumeric with "."
        label_value = re.sub(r"^[^A-Za-z0-9]+", ".", label_value)
        # Replace trailing non-alphanumeric with "."
        label_value = re.sub(r"[^A-Za-z0-9]+$", ".", label_value)

        # Trim to 63 characters
        return label_value[:63]

    return {
        LABEL_VERSION: clean(torchx.__version__),
        LABEL_APP_NAME: clean(app.name),
        LABEL_ROLE_INDEX: str(role_idx),
        LABEL_ROLE_NAME: clean(role.name),
        LABEL_REPLICA_ID: str(replica_id),
        LABEL_KUBE_APP_NAME: clean(app.name),
        LABEL_ORGANIZATION: "torchx.pytorch.org",
        LABEL_UNIQUE_NAME: clean(app_id),
    }
