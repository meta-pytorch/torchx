#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""

This contains the TorchX Kubernetes_MCAD scheduler which can be used to run TorchX
components on a Kubernetes cluster via the Multi-Cluster-Application-Dispatcher (MCAD).

Prerequisites
==============

TorchX Kubernetes_MCAD scheduler depends on AppWrapper + MCAD.

Install MCAD:
See deploying Multi-Cluster-Application-Dispatcher guide
https://github.com/project-codeflare/multi-cluster-app-dispatcher/blob/main/doc/deploy/deployment.md

This implementation requires MCAD v1.34.1 or higher.

TorchX uses `torch.distributed.run <https://pytorch.org/docs/stable/elastic/run.html>`_ to run distributed training.

Learn more about running distributed trainers :py:mod:`torchx.components.dist`

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
    Resource,
    RetryPolicy,
    Role,
    RoleStatus,
    runopts,
    VolumeMount,
)

from torchx.workspace.docker_workspace import DockerWorkspaceMixin

if TYPE_CHECKING:
    from docker import DockerClient
    from kubernetes.client import ApiClient, CustomObjectsApi
    from kubernetes.client.models import (  # noqa: F401 imported but unused
        V1Container,
        V1ContainerPort,
        V1EnvVar,
        V1Pod,
        V1PodSpec,
        V1ResourceRequirements,
        V1Service,
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

RETRY_POLICIES: Mapping[str, Iterable[Mapping[str, str]]] = {
    RetryPolicy.REPLICA: [],
    RetryPolicy.APPLICATION: [
        {"event": "PodEvicted", "action": "RestartJob"},
        {"event": "PodFailed", "action": "RestartJob"},
    ],
}

# The AppWrapper Status - holistic view of pods/services
JOB_STATE: Dict[str, AppState] = {
    # Pending is the AppWrapper condition waiting for scheduling by MCAD
    "Pending": AppState.PENDING,
    # Running is the AppWrapper condition in Running.
    "Running": AppState.RUNNING,
    # Deleted is the AppWrapper condition where the torchX job is cancelled
    # by the user, the AppWrapper is deleted by the user, or error handling
    "Deleted": AppState.CANCELLED,
    # Failed is the job finishes unexpectedly
    "Failed": AppState.FAILED,
}

TASK_STATE: Dict[str, ReplicaState] = {
    # Pending dispatch means the AppWrapped task is not yet scheduled by MCAD
    "Pending dispatch": ReplicaState.PENDING,
    # Pending means the task is scheduled by MCAD
    "pending": ReplicaState.PENDING,
    # Running means a task is running on the host.
    "running": ReplicaState.RUNNING,
    # Succeeded means that all containers in the pod have voluntarily
    # terminated with a container exit code of 0, and the system is not
    # going to restart any of these containers.
    "Succeeded": ReplicaState.SUCCEEDED,
    # Failed means that all containers in the pod have terminated, and at
    # least one container has terminated in a failure (exited with a
    # non-zero exit code or was stopped by the system).
    "failed": ReplicaState.FAILED,
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


def sanitize_for_serialization(obj: object) -> object:
    from kubernetes import client

    api = client.ApiClient()
    return api.sanitize_for_serialization(obj)


def role_to_pod(
    name: str,
    unique_app_id: str,
    namespace: str,
    role: Role,
    service_account: Optional[str],
    image_secret: Optional[str],
    coscheduler_name: Optional[str],
    priority_class_name: Optional[str],
    network: Optional[str],
) -> "V1Pod":
    from kubernetes.client.models import (  # noqa: F811 redefinition of unused
        V1Container,
        V1ContainerPort,
        V1EmptyDirVolumeSource,
        V1EnvVar,
        V1HostPathVolumeSource,
        V1LocalObjectReference,
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

    torchx_env_var = [
        V1EnvVar(
            name=name,
            value=value,
        )
        for name, value in role.env.items()
    ]

    my_env_var = [
        V1EnvVar(
            name=f"TORCHX_MCAD_{cleanup_str(role.name)}_0_HOSTS".upper().replace(
                "-", ""
            ),
            value=f"{unique_app_id}-0.{unique_app_id}",
        )
    ]

    container = V1Container(
        command=[role.entrypoint] + role.args,
        image=role.image,
        name=name,
        env=torchx_env_var + my_env_var,
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

    # Get correct formatting for image secret
    imagesecret = V1LocalObjectReference(name=image_secret)
    metadata = V1ObjectMeta(
        name=name,
        annotations={
            # Disable the istio sidecar as it prevents the containers from
            # exiting once finished.
            ANNOTATION_ISTIO_SIDECAR: "false",
        },
        labels={},
        namespace=namespace,
    )
    if network is not None:
        metadata.annotations.update({"k8s.v1.cni.cncf.io/networks": network})

    return V1Pod(
        api_version="v1",
        kind="Pod",
        spec=V1PodSpec(
            containers=[container],
            hostname=name,
            subdomain=unique_app_id,
            image_pull_secrets=[imagesecret],
            restart_policy="Never",
            service_account_name=service_account,
            volumes=volumes,
            node_selector=node_selector,
            scheduler_name=coscheduler_name,
            priority_class_name=priority_class_name,
        ),
        metadata=metadata,
    )


def create_pod_group(
    app: AppDef, role: Role, role_idx: int, namespace: str, app_id: str
) -> "Dict[str, Any]":
    pod_group_name = app_id + "-pg" + str(role_idx)

    labels = object_labels(app, app_id)
    labels.update({"appwrapper.workload.codeflare.dev": app_id})

    pod_group: Dict[str, Any] = {
        "apiVersion": "scheduling.sigs.k8s.io/v1alpha1",
        "kind": "PodGroup",
        "metadata": {
            "name": pod_group_name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "minMember": role.num_replicas,
        },
    }

    genericitem_pod_group: Dict[str, Any] = {
        "replicas": 1,
        "generictemplate": pod_group,
    }
    return genericitem_pod_group


def mcad_svc(
    app: AppDef, svc_name: str, namespace: str, service_port: str
) -> "V1Service":
    from kubernetes.client.models import (  # noqa: F401, F811
        V1Container,
        V1ContainerPort,
        V1EmptyDirVolumeSource,
        V1EnvVar,
        V1HostPathVolumeSource,
        V1ObjectMeta,
        V1PersistentVolumeClaimVolumeSource,
        V1Pod,
        V1PodSpec,
        V1ResourceRequirements,
        V1SecurityContext,
        V1Service,
        V1ServicePort,
        V1ServiceSpec,
        V1ServiceStatus,
        V1Volume,
        V1VolumeMount,
    )

    labels = object_labels(app, svc_name)

    return V1Service(
        api_version="v1",
        kind="Service",
        metadata=V1ObjectMeta(
            name=svc_name,
            namespace=namespace,
            labels=labels,
        ),
        spec=V1ServiceSpec(
            cluster_ip="None",
            publish_not_ready_addresses=True,
            ports=[
                V1ServicePort(
                    protocol="TCP",
                    port=int(service_port),
                    target_port=int(service_port),
                )
            ],
            selector={LABEL_UNIQUE_NAME: svc_name},
            session_affinity="None",
            type="ClusterIP",
        ),
        status=V1ServiceStatus(
            load_balancer={},
        ),
    )


def cleanup_str(data: str) -> str:
    """
    Invokes ``lower`` on thes string and removes all
    characters that do not satisfy ``[a-z0-9]`` pattern.
    This method is mostly used to make sure kubernetes scheduler gets
    the job name that does not violate its validation.
    """
    if data.startswith("-"):
        data = data[1:]
    pattern = r"[a-z0-9\-]"
    return "".join(re.findall(pattern, data.lower())).lstrip("0123456789")


def get_unique_truncated_appid(app: AppDef) -> str:
    """
    Some Kubernetes objects need to have names that are
    63 characters or less. When creating the unique app_id,
    this function calculates the max size to pass to
    make_unique. The PodGroup name includes 3 characters plus
    the role_id characters. The minimum number of characters
    for the unique identifier is 4.  These amounts are taken into account.
    """
    default_size = 14
    uid_chars = 4
    pg_chars = 3 + len(app.roles)
    size = 63 - (len(app.name) + uid_chars + pg_chars)

    unique_id_size = default_size if size > default_size else size

    if unique_id_size <= 3:
        msg = "Name size has too many characters for some Kubernetes objects. Truncating \
application name."
        warnings.warn(msg)
        end = 63 - uid_chars - pg_chars
        substring = app.name[0:end]
        app.name = substring
        unique_id_size = 3

    unique_app_id = cleanup_str(make_unique(app.name, unique_id_size))
    return unique_app_id


def get_port_for_service(app: AppDef) -> str:
    # Initialize port to default
    port = "29500"

    for role_idx, role in enumerate(app.roles):
        if role.port_map is None:
            continue
        for value in role.port_map.values():
            port = str(value)

    if not (0 < int(port) <= 65535):
        msg = """Warning: port_map set to invalid port number. Value must be between 1-65535, with torchx default = 29500. Setting port to default = 29500"""
        port = "29500"
        warnings.warn(msg)

    return port


def enable_retry(
    job_spec: Dict[str, Any], appwrapper_retries: int, total_pods: int
) -> None:
    requeue_dict = {
        "timeInSeconds": 300,
        "maxTimeInSeconds": 0,
        "growthType": "exponential",
        "maxNumRequeuings": appwrapper_retries,
    }
    nested_specs = {"minAvailable": total_pods, "requeuing": requeue_dict}
    job_spec["schedulingSpec"] = nested_specs


def app_to_resource(
    app: AppDef,
    namespace: str,
    service_account: Optional[str],
    image_secret: Optional[str],
    coscheduler_name: Optional[str],
    priority_class_name: Optional[str],
    network: Optional[str],
    priority: Optional[int] = None,
) -> Dict[str, Any]:
    """
    app_to_resource creates a AppWrapper/MCAD Kubernetes resource definition from
    the provided AppDef. The resource definition can be used to launch the
    app on Kubernetes.

    MCAD supports retries at the APPLICATION level. In the case of multiple TorchX Roles,
    the AppWrapper maximum number of retries
    count is set to the minimum of the max_retries of the roles.
    """

    genericitems = []

    unique_app_id = get_unique_truncated_appid(app)

    if coscheduler_name is not None:
        for role_idx, role in enumerate(app.roles):
            genericitem_pod_group = create_pod_group(
                app, role, role_idx, namespace, unique_app_id
            )
            genericitems.append(genericitem_pod_group)

    for role_idx, role in enumerate(app.roles):
        for replica_id in range(role.num_replicas):
            values = macros.Values(
                img_root="",
                app_id=unique_app_id,
                replica_id=str(replica_id),
                rank0_env=f"TORCHX_MCAD_{cleanup_str(app.roles[0].name)}_0_HOSTS".upper().replace(
                    "-", ""
                ),
            )

            if role_idx == 0 and replica_id == 0:
                values.rank0_env = "TORCHX_RANK0_HOST"
            name = cleanup_str(f"{unique_app_id}-{replica_id}")
            replica_role = values.apply(role)
            if role_idx == 0 and replica_id == 0:
                replica_role.env["TORCHX_RANK0_HOST"] = "localhost"

            pod = role_to_pod(
                name,
                unique_app_id,
                namespace,
                replica_role,
                service_account,
                image_secret,
                coscheduler_name,
                priority_class_name,
                network,
            )
            pod.metadata.labels.update(
                pod_labels(
                    app=app,
                    role_idx=role_idx,
                    role=role,
                    replica_id=replica_id,
                    coscheduler_name=coscheduler_name,
                    app_id=unique_app_id,
                )
            )

            genericitem: Dict[str, Any] = {
                "replicas": 1,
                "generictemplate": pod,
            }
            genericitems.append(genericitem)

    """
    Create Service:
    The selector will have the key 'appwrapper.workload.codeflare.dev', and the value will be
    the appwrapper name
    """

    service_port = get_port_for_service(app)

    svc_obj = mcad_svc(
        app=app, svc_name=unique_app_id, namespace=namespace, service_port=service_port
    )

    genericitem_svc: Dict[str, Any] = {
        "replicas": 1,
        "generictemplate": svc_obj,
    }
    genericitems.append(genericitem_svc)

    job_spec: Dict[str, Any] = {
        "resources": {
            "GenericItems": genericitems,
        },
    }

    if priority is not None:
        job_spec["priority"] = priority

    appwrapper_retries = min(role.max_retries for role in app.roles)
    if appwrapper_retries > 0:
        total_pods = sum(role.num_replicas for role in app.roles)
        enable_retry(job_spec, appwrapper_retries, total_pods)

    resource: Dict[str, object] = {
        "apiVersion": "workload.codeflare.dev/v1beta1",
        "kind": "AppWrapper",
        "metadata": {"name": unique_app_id, "namespace": namespace},
        "spec": job_spec,
    }

    return resource


# Helper function for MCAD generic items information -> TorchX Role
def get_role_information(generic_items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    # Store unique role information
    roles = {}

    # nested dictionary keys
    # meta data information
    GT_KEY = "generictemplate"
    METADATA_KEY = "metadata"
    LABEL_KEY = "labels"
    ROLE_KEY = "torchx.pytorch.org/role-name"

    # containers information
    SPEC_KEY = "spec"
    CONTAINER_KEY = "containers"
    IMAGE_KEY = "image"
    ARGS_KEY = "command"
    ENV_KEY = "env"
    RESOURCE_KEY = "resources"
    PORTS_KEY = "ports"
    MOUNTS_KEY = "volumeMounts"

    # resource keys
    CPU_KEY = "cpu"
    GPU_KEY = "gpu"
    REQUEST_KEY = "requests"
    MEM_KEY = "memory"

    for generic_item in generic_items:
        if GT_KEY not in generic_item.keys():
            continue
        gt_result = generic_item[GT_KEY]

        if METADATA_KEY not in gt_result.keys():
            continue
        # Note: options in meta data : annotations, labels, name, namespace
        metadata_result = gt_result[METADATA_KEY]

        if LABEL_KEY not in metadata_result.keys():
            continue
        label_result = metadata_result[LABEL_KEY]
        if ROLE_KEY not in label_result.keys():
            continue
        role_name: str = label_result[ROLE_KEY]
        # save role
        if role_name not in roles:
            roles[role_name] = Role(name=role_name, num_replicas=0, image="")
            roles[role_name].num_replicas += 1

            # Only get specs for first instance of TorchX role
            if SPEC_KEY not in gt_result.keys():
                continue
            # Note: options in spec data: containers, hostname, imagePullSecrets, nodeSelector
            #      restartPolicy, subdomain, volumes
            spec_result = gt_result[SPEC_KEY]
            if CONTAINER_KEY not in spec_result.keys():
                continue
            container_result = spec_result[CONTAINER_KEY]
            if IMAGE_KEY not in container_result[0].keys():
                continue
            roles[role_name].image = container_result[0][IMAGE_KEY]

            if ARGS_KEY not in container_result[0].keys():
                continue
            roles[role_name].args = container_result[0][ARGS_KEY]
            if ENV_KEY not in container_result[0].keys():
                continue
            roles[role_name].env = container_result[0][ENV_KEY]
            if RESOURCE_KEY not in container_result[0].keys():
                continue
            roles[role_name].resources = container_result[0][RESOURCE_KEY]
            resource_req = Resource(cpu=-1, gpu=-1, memMB=-1)
            if CPU_KEY not in container_result[0][RESOURCE_KEY][REQUEST_KEY]:
                continue
            resource_req.cpu = container_result[0][RESOURCE_KEY][REQUEST_KEY][CPU_KEY]
            # Substring matching to accomodate different gpu types
            gpu_key_values = dict(
                filter(
                    lambda item: GPU_KEY in item[0],
                    container_result[0][RESOURCE_KEY][REQUEST_KEY].items(),
                )
            )
            if len(gpu_key_values) != 0:
                for key, value in gpu_key_values.items():
                    resource_req.gpu = value
            if MEM_KEY not in container_result[0][RESOURCE_KEY][REQUEST_KEY]:
                continue
            resource_req.memMB = container_result[0][RESOURCE_KEY][REQUEST_KEY][MEM_KEY]
            roles[role_name].resource = resource_req

            if PORTS_KEY not in container_result[0].keys():
                continue
            roles[role_name].port_map = container_result[0][PORTS_KEY]
            if MOUNTS_KEY not in container_result[0].keys():
                continue
            roles[role_name].mounts = container_result[0][MOUNTS_KEY]
        else:
            roles[role_name].num_replicas += 1

    return roles


def get_appwrapper_status(app: Dict[str, str]) -> AppState:
    if "status" in app.keys():
        # pyre-fixme
        return JOB_STATE[app["status"]["state"]]
    else:
        # Handle case where appwrapper is created but pending dispatch
        return JOB_STATE["Pending"]


# Does not handle not ready to dispatch case
def get_tasks_status_description(status: Dict[str, str]) -> Dict[str, int]:
    results = {}

    # Keys related to tasks and status
    KEY_RUN = "running"
    KEY_PEND = "pending"
    KEY_FAIL = "failed"
    KEY_SUCCESS = "Succeeded"

    if KEY_RUN in status.keys():
        results[KEY_RUN] = status[KEY_RUN]
    if KEY_PEND in status.keys():
        results[KEY_PEND] = status[KEY_PEND]
    if KEY_FAIL in status.keys():
        results[KEY_FAIL] = status[KEY_FAIL]
    if KEY_SUCCESS in status.keys():
        results[KEY_SUCCESS] = status[KEY_SUCCESS]

    return results


@dataclass
class KubernetesMCADJob:
    images_to_push: Dict[str, Tuple[str, str]]
    resource: Dict[str, object]

    def __str__(self) -> str:
        return yaml.dump(sanitize_for_serialization(self.resource))

    def __repr__(self) -> str:
        return str(self)


class KubernetesMCADOpts(TypedDict, total=False):
    namespace: Optional[str]
    image_repo: Optional[str]
    service_account: Optional[str]
    priority: Optional[int]
    priority_class_name: Optional[str]
    image_secret: Optional[str]
    coscheduler_name: Optional[str]
    network: Optional[str]


class KubernetesMCADScheduler(
    DockerWorkspaceMixin,
    Scheduler[KubernetesMCADOpts, AppDef, AppDryRunInfo[KubernetesMCADJob]],
):
    """
    KubernetesMCADScheduler is a TorchX scheduling interface to Kubernetes.

    Important: AppWrapper/MCAD is required to be installed on the Kubernetes cluster.
    TorchX requires gang scheduling for multi-replica/multi-role execution.
    Note that AppWrapper/MCAD supports gang scheduling among any app-wrapped jobs on Kubernetes.
    However, for true gang scheduling AppWrapper/MCAD needs to be used with an additional Kubernetes
    co-scheduler.
    For installation instructions see: https://github.com/project-codeflare/multi-cluster-app-dispatcher/blob/main/doc/deploy/deployment.md

    This has been confirmed to work with MCAD main branch v1.34.1 or higher and OpenShift Kubernetes
    Client Version: 4.10.13
    Server Version: 4.9.18
    Kubernetes Version: v1.22.3+e790d7f

    .. code-block:: bash

        $ torchx run --scheduler kubernetes_mcad --scheduler_args namespace=default,image_repo=<your_image_repo> utils.echo --image alpine:latest --msg hello
        ...

    The TorchX-MCAD scheduler can be used with a secondary scheduler on Kubernetes.
    To enable this, the user must provide the name of the coscheduler.
    With this feature, a PodGroup is defined for each TorchX role and the coscheduler
    handles secondary scheduling on the Kubernetes cluster. For additional resources, see:
    1. PodGroups and Coscheduling: https://github.com/kubernetes-sigs/scheduler-plugins/tree/release-1.24/pkg/coscheduling
    2. Installing Secondary schedulers: https://github.com/kubernetes-sigs/scheduler-plugins/blob/release-1.24/doc/install.md
    3. PodGroup CRD: https://github.com/kubernetes-sigs/scheduler-plugins/blob/release-1.24/config/crd/bases/scheduling.sigs.k8s.io_podgroups.yaml

    The MCAD scheduler supports priorities at the AppWrapper level and optionally at the pod level on clusters with PriorityClass definitions.
    At the AppWrapper level, higher integer values means higher priorities. Kubernetes clusters may have additional priorityClass
    definitions that can be applied at the pod level. While these different levels of priorities can be set independently,
    it is recommended to check with your Kubernetes cluster admin to see if additional guidance is in place. For more on Kubernetes
    PriorityClass, see: https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/ .

    In order to use the network option, the Kubernetes cluster must have multus installed.
    For multus installation instructions and how to set up a network custom network attachment definition, see:
    https://github.com/k8snetworkplumbingwg/multus-cni/blob/master/docs/how-to-use.md

    **Config Options**

    .. runopts::
        class: torchx.schedulers.kubernetes_mcad_scheduler.KubernetesMCADScheduler

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
            describe: true
            workspaces: true
            mounts: true
            elasticity: false
    """

    def __init__(
        self,
        session_name: str,
        client: Optional["ApiClient"] = None,
        docker_client: Optional["DockerClient"] = None,
    ) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("kubernetes_mcad", session_name, docker_client=docker_client)

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

        api_client = client.CustomObjectsApi(self._api_client())
        return api_client

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

    def schedule(self, dryrun_info: AppDryRunInfo[KubernetesMCADJob]) -> str:
        from kubernetes.client.rest import ApiException

        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"
        namespace = cfg.get("namespace") or "default"

        images_to_push = dryrun_info.request.images_to_push
        self.push_images(images_to_push)

        resource = dryrun_info.request.resource

        try:
            resp = self._custom_objects_api().create_namespaced_custom_object(
                group="workload.codeflare.dev",
                version="v1beta1",
                namespace=namespace,
                plural="appwrappers",
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

        return f'{namespace}:{resp["metadata"]["name"]}'

    def _submit_dryrun(
        self, app: AppDef, cfg: KubernetesMCADOpts
    ) -> AppDryRunInfo[KubernetesMCADJob]:
        # map any local images to the remote image
        # images_to_push = self._update_app_images(app, cfg.get("image_repo"))
        images_to_push = self.dryrun_push_images(app, cast(Mapping[str, CfgVal], cfg))

        service_account = cfg.get("service_account")
        assert service_account is None or isinstance(
            service_account, str
        ), "service_account must be a str"

        priority = cfg.get("priority")
        assert priority is None or isinstance(priority, int), "priority must be a int"

        image_secret = cfg.get("image_secret")
        assert image_secret is None or isinstance(
            image_secret, str
        ), "image_secret must be a str"

        if image_secret is not None and service_account is not None:
            msg = """Service Account and Image Secret names are both provided.
 Depending on the Service Account configuration, an ImagePullSecret may be defined in your Service Account.
 If this is the case, check service account and image secret configurations to understand the expected behavior for
 patched image push access."""
            warnings.warn(msg)
        namespace = cfg.get("namespace")
        assert isinstance(namespace, str), "namespace must be a str"

        coscheduler_name = cfg.get("coscheduler_name")
        assert coscheduler_name is None or isinstance(
            coscheduler_name, str
        ), "coscheduler_name must be a string"

        priority_class_name = cfg.get("priority_class_name")
        assert priority_class_name is None or isinstance(
            priority_class_name, str
        ), "priority_class_name must be a string"

        network = cfg.get("network")
        assert network is None or isinstance(network, str), "network must be a string"

        resource = app_to_resource(
            app=app,
            namespace=namespace,
            service_account=service_account,
            image_secret=image_secret,
            coscheduler_name=coscheduler_name,
            priority_class_name=priority_class_name,
            network=network,
            priority=priority,
        )

        req = KubernetesMCADJob(
            resource=resource,
            images_to_push=images_to_push,
        )

        info = AppDryRunInfo(req, repr)
        info._app = app
        # pyre-fixme
        info._cfg = cfg
        return info

    def _validate(self, app: AppDef, scheduler: str, cfg: KubernetesMCADOpts) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        namespace, name = app_id.split(":")
        self._custom_objects_api().delete_namespaced_custom_object(
            group="workload.codeflare.dev",
            version="v1beta1",
            namespace=namespace,
            plural="appwrappers",
            name=name,
        )

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "namespace",
            type_=str,
            help="Kubernetes namespace to schedule job in",
            default="default",
        )
        opts.add(
            "image_repo",
            type_=str,
            help="The image repository to use when pushing patched images, must have push access. Ex: example.com/your/container",
        )
        opts.add(
            "service_account",
            type_=str,
            help="The service account name to set on the pod specs",
        )
        opts.add(
            "priority",
            type_=int,
            help="The priority level to set on the job specs. Higher integer value means higher priority",
        )
        opts.add(
            "priority_class_name",
            type_=str,
            help="Pod specific priority level. Check with your Kubernetes cluster admin if Priority classes are defined on your system",
        )
        opts.add(
            "image_secret",
            type_=str,
            help="The name of the Kubernetes/OpenShift secret set up for private images",
        )
        opts.add(
            "coscheduler_name",
            type_=str,
            help="Option to run TorchX-MCAD with a co-scheduler. User must provide the co-scheduler name.",
        )
        opts.add(
            "network",
            type_=str,
            help="Name of additional pod-to-pod network beyond default Kubernetes network",
        )
        return opts

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        namespace, name = app_id.split(":")
        from kubernetes.client.rest import ApiException

        roles = {}
        roles_statuses = {}

        # Production section
        api_instance = self._custom_objects_api
        group = "workload.codeflare.dev"
        version = "v1beta1"
        plural = "appwrappers"
        try:
            api_resp = api_instance().get_namespaced_custom_object(
                group, version, namespace, plural, name
            )
        except ApiException as e:
            api_resp = {}
            if e.status == 404 and e.reason == "Not Found":
                raise ValueError(
                    "Kubernetes client not found. Check access to Kubernetes cluster."
                ) from e
            elif e.status == 401 and e.reason == "Unauthorized":
                raise ValueError("Unauthorized Kubernetes access error.") from e
            else:
                raise

        task_status = []
        if "status" in api_resp.keys():
            status = api_resp["status"]
            tasks_results = get_tasks_status_description(status)
            # Handle case where waiting for dispatch
            if not tasks_results:
                tasks_results["Pending dispatch"] = (
                    len(api_resp["spec"]["resources"]["GenericItems"]) - 1
                )

            # Convert MCAD status to TorchX replica set status format
            # Warning: Status is not necessarily the match for a particular Replica ID
            for key, value in tasks_results.items():
                for id in range(0, value):
                    task_status.append(key)

            state = status["state"]
            app_state = JOB_STATE[state]

            # Roles
            spec = api_resp["spec"]
            resources = spec["resources"]
            generic_items = resources["GenericItems"]

            # Note MCAD service is not considered a TorchX role
            roles = get_role_information(generic_items)

            task_count = 0
            for role in roles:
                msg = "Warning - MCAD does not report individual replica statuses, but overall task status. Replica id  may not match status"
                warnings.warn(msg)

                roles_statuses[role] = RoleStatus(role, [])
                for idx in range(0, roles[role].num_replicas):
                    state = TASK_STATE[task_status[task_count]]
                    roles_statuses[role].replicas.append(
                        ReplicaStatus(id=int(idx), role=role, state=state, hostname="")
                    )
                    task_count += 1

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
            raise ValueError(
                "KubernetesMCADScheduler only supports COMBINED log stream"
            )

        from kubernetes import client, watch

        namespace, name = app_id.split(":")

        pod_name = cleanup_str(f"{name}-{k}")

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
            group="workload.codeflare.dev",
            version="v1beta1",
            namespace=namespace,
            plural="appwrappers",
            timeout_seconds=30,
        )

        return [
            ListAppResponse(
                app_id=f"{namespace}:{item['metadata']['name']}",
                state=get_appwrapper_status(item),
            )
            for item in resp["items"]
        ]


def create_scheduler(
    session_name: str,
    client: Optional["ApiClient"] = None,
    docker_client: Optional["DockerClient"] = None,
    **kwargs: Any,
) -> KubernetesMCADScheduler:
    return KubernetesMCADScheduler(
        session_name=session_name,
        client=client,
        docker_client=docker_client,
    )


def object_labels(
    app: AppDef,
    app_id: str,
) -> Dict[str, str]:
    return {
        LABEL_KUBE_APP_NAME: app.name,
        LABEL_ORGANIZATION: "torchx.pytorch.org",
        LABEL_UNIQUE_NAME: app_id,
    }


def pod_labels(
    app: AppDef,
    role_idx: int,
    role: Role,
    replica_id: int,
    coscheduler_name: Optional[str],
    app_id: str,
) -> Dict[str, str]:
    labels = object_labels(app, app_id)
    pod_labels = {
        LABEL_VERSION: torchx.__version__,
        LABEL_APP_NAME: app.name,
        LABEL_ROLE_INDEX: str(role_idx),
        LABEL_ROLE_NAME: role.name,
        LABEL_REPLICA_ID: str(replica_id),
    }
    if coscheduler_name is not None:
        pod_group = app_id + "-pg" + str(role_idx)
        pod_labels.update({"pod-group.scheduling.sigs.k8s.io": pod_group})

    labels.update(pod_labels)
    return labels
