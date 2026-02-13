# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import base64
import importlib
import sys
import unittest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import torchx
from torchx import schedulers, specs

# @manual=//torchx/schedulers:kubernetes_scheduler
from torchx.schedulers import kubernetes_scheduler
from torchx.schedulers.api import DescribeAppResponse, ListAppResponse
from torchx.schedulers.docker_scheduler import has_docker
from torchx.schedulers.ids import make_unique
from torchx.schedulers.kubernetes_scheduler import (
    app_to_resource,
    create_scheduler,
    KubernetesJob,
    KubernetesScheduler,
    LABEL_APP_NAME,
    LABEL_INSTANCE_TYPE,
    LABEL_KUBE_APP_NAME,
    LABEL_ORGANIZATION,
    LABEL_REPLICA_ID,
    LABEL_ROLE_INDEX,
    LABEL_ROLE_NAME,
    LABEL_UNIQUE_NAME,
    LABEL_VERSION,
    Opts,
    PLACEHOLDER_FIELD_PATH,
    role_to_pod,
)
from torchx.specs import AppDryRunInfo, AppState
from torchx.specs.overlays import JOIN, PUT, set_overlay
from torchx.util.strings import normalize_str

SKIP_DOCKER: bool = not has_docker()

TEST_KUBE_CONFIG: dict[str, Any] = {
    "current-context": "default",
    "contexts": [
        {
            "name": "default",
            "context": {
                "cluster": "default",
                "user": "torchx_fake_token",
                "namespace": "default",
            },
        }
    ],
    "clusters": [{"name": "default", "cluster": {"server": "torchx_test_host"}}],
    "users": [
        {
            "name": "torchx_fake_token",
            "user": {
                "token": base64.standard_b64encode(
                    "torchx-test-token".encode()
                ).decode(),
                "username": "me",
                "password": "password1234",
            },
        }
    ],
}


def _test_app(num_replicas: int = 1) -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer_foo",
        image="pytorch/torchx:latest",
        entrypoint="main",
        args=[
            "--output-path",
            specs.macros.img_root,
            "--app-id",
            specs.macros.app_id,
            "--rank0-env",
            specs.macros.rank0_env,
        ],
        env={"FOO": "bar", "FOO_FIELD_PATH": f"{PLACEHOLDER_FIELD_PATH}bar"},
        resource=specs.Resource(
            cpu=2,
            memMB=3000,
            gpu=4,
        ),
        port_map={"foo": 1234},
        num_replicas=num_replicas,
        max_retries=3,
        mounts=[
            specs.BindMount(src_path="/src", dst_path="/dst", read_only=True),
        ],
    )

    return specs.AppDef("test", roles=[trainer_role])


class KubernetesSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        # Mock create_namespaced_custom_object for validation calls in submit_dryrun
        # This prevents tests from calling real k8s endpoint during validation
        self.mock_create_patcher = patch(
            "kubernetes.client.CustomObjectsApi.create_namespaced_custom_object"
        )
        self.mock_create = self.mock_create_patcher.start()
        self.mock_create.return_value = {}

    def tearDown(self) -> None:
        self.mock_create_patcher.stop()

    def test_create_scheduler(self) -> None:
        client = MagicMock()
        docker_client = MagicMock
        scheduler = create_scheduler("foo", client=client, docker_client=docker_client)
        self.assertIsInstance(scheduler, kubernetes_scheduler.KubernetesScheduler)
        self.assertEqual(scheduler._docker_client, docker_client)
        self.assertEqual(scheduler._client, client)

    def test_app_to_resource_resolved_macros(self) -> None:
        app = _test_app()
        unique_app_name = "app-name-42"
        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = unique_app_name
            resource = app_to_resource(app, "test_queue", service_account=None)
            actual_cmd = (
                resource["spec"]["tasks"][0]["template"].spec.containers[0].command
            )
            expected_cmd = [
                "main",
                "--output-path",
                "",
                "--app-id",
                unique_app_name,
                "--rank0-env",
                "TORCHX_RANK0_HOST",
            ]
            self.assertEqual(expected_cmd, actual_cmd)

    def test_retry_policy_not_set(self) -> None:
        app = _test_app()
        resource = app_to_resource(app, "test_queue", service_account=None)
        self.assertListEqual(
            [
                {"event": "PodEvicted", "action": "RestartJob"},
                {"event": "PodFailed", "action": "RestartJob"},
            ],
            resource["spec"]["tasks"][0]["policies"],
        )
        for role in app.roles:
            role.max_retries = 0
        resource = app_to_resource(app, "test_queue", service_account=None)
        self.assertFalse("policies" in resource["spec"]["tasks"][0])
        self.assertFalse("maxRetry" in resource["spec"]["tasks"][0])

    def test_role_to_pod(self) -> None:
        from kubernetes.client.models import (
            V1Container,
            V1ContainerPort,
            V1EmptyDirVolumeSource,
            V1EnvVar,
            V1EnvVarSource,
            V1HostPathVolumeSource,
            V1ObjectFieldSelector,
            V1ObjectMeta,
            V1Pod,
            V1PodSpec,
            V1ResourceRequirements,
            V1SecurityContext,
            V1Volume,
            V1VolumeMount,
        )

        app = _test_app()
        pod = role_to_pod("name", app.roles[0], service_account="srvacc")

        limits = {
            "cpu": "2000m",
            "memory": "3000M",
            "nvidia.com/gpu": "4",
        }
        requests = {
            "cpu": "1900m",
            "memory": "1976M",
            "nvidia.com/gpu": "4",
        }
        resources = V1ResourceRequirements(
            limits=limits,
            requests=requests,
        )
        container = V1Container(
            command=[
                "main",
                "--output-path",
                specs.macros.img_root,
                "--app-id",
                specs.macros.app_id,
                "--rank0-env",
                specs.macros.rank0_env,
            ],
            image="pytorch/torchx:latest",
            name="name",
            env=[
                V1EnvVar(name="FOO", value="bar"),
                V1EnvVar(
                    name="FOO_FIELD_PATH",
                    value_from=V1EnvVarSource(
                        field_ref=V1ObjectFieldSelector(field_path="bar")
                    ),
                ),
            ],
            resources=resources,
            ports=[V1ContainerPort(name="foo", container_port=1234)],
            security_context=V1SecurityContext(),
            volume_mounts=[
                V1VolumeMount(
                    name="dshm",
                    mount_path="/dev/shm",
                ),
                V1VolumeMount(
                    name="mount-0",
                    mount_path="/dst",
                    read_only=True,
                ),
            ],
        )
        want = V1Pod(
            spec=V1PodSpec(
                containers=[container],
                restart_policy="Never",
                service_account_name="srvacc",
                volumes=[
                    V1Volume(
                        name="dshm",
                        empty_dir=V1EmptyDirVolumeSource(
                            medium="Memory",
                        ),
                    ),
                    V1Volume(
                        name="mount-0",
                        host_path=V1HostPathVolumeSource(
                            path="/src",
                        ),
                    ),
                ],
                node_selector={},
            ),
            metadata=V1ObjectMeta(
                annotations={
                    "sidecar.istio.io/inject": "false",
                },
                labels={},
            ),
        )

        print(want)

        self.assertEqual(
            pod,
            want,
        )

    def test_submit_dryrun(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = Opts(queue="testqueue")
        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler.submit_dryrun(app, cfg)

        resource = str(info.request)
        self.mock_create.assert_called_once()
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["dry_run"], "All")

        print(resource)

        self.assertEqual(
            resource,
            f"""apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: app-name-42
spec:
  maxRetry: 3
  plugins:
    env: []
    svc:
    - --publish-not-ready-addresses
  queue: testqueue
  schedulerName: volcano
  tasks:
  - maxRetry: 3
    name: trainerfoo-0
    policies:
    - action: RestartJob
      event: PodEvicted
    - action: RestartJob
      event: PodFailed
    replicas: 1
    template:
      metadata:
        annotations:
          sidecar.istio.io/inject: 'false'
        labels:
          app.kubernetes.io/instance: app-name-42
          app.kubernetes.io/managed-by: torchx.pytorch.org
          app.kubernetes.io/name: test
          torchx.pytorch.org/app-name: test
          torchx.pytorch.org/replica-id: '0'
          torchx.pytorch.org/role-index: '0'
          torchx.pytorch.org/role-name: trainer_foo
          torchx.pytorch.org/version: {torchx.__version__.replace("+", ".")}
      spec:
        containers:
        - command:
          - main
          - --output-path
          - ''
          - --app-id
          - app-name-42
          - --rank0-env
          - TORCHX_RANK0_HOST
          env:
          - name: FOO
            value: bar
          - name: FOO_FIELD_PATH
            valueFrom:
              fieldRef:
                fieldPath: bar
          - name: TORCHX_RANK0_HOST
            value: localhost
          - name: TORCHX_IMAGE
            value: pytorch/torchx:latest
          image: pytorch/torchx:latest
          name: trainerfoo-0
          ports:
          - containerPort: 1234
            name: foo
          resources:
            limits:
              cpu: 2000m
              memory: 3000M
              nvidia.com/gpu: '4'
            requests:
              cpu: 1900m
              memory: 1976M
              nvidia.com/gpu: '4'
          securityContext: {{}}
          volumeMounts:
          - mountPath: /dev/shm
            name: dshm
          - mountPath: /dst
            name: mount-0
            readOnly: true
        nodeSelector: {{}}
        restartPolicy: Never
        volumes:
        - emptyDir:
            medium: Memory
          name: dshm
        - hostPath:
            path: /src
          name: mount-0
""",
        )

    def test_volume_mounts(self) -> None:
        scheduler = create_scheduler("test")
        from kubernetes.client.models import (
            V1EmptyDirVolumeSource,
            V1PersistentVolumeClaimVolumeSource,
            V1Volume,
            V1VolumeMount,
        )

        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.VolumeMount(src="name", dst_path="/dst", read_only=True),
            ],
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.volumes,
            [
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory",
                    ),
                ),
                V1Volume(
                    name="mount-0",
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name="name",
                    ),
                ),
            ],
        )
        self.assertEqual(
            pod.spec.containers[0].volume_mounts,
            [
                V1VolumeMount(
                    name="dshm",
                    mount_path="/dev/shm",
                ),
                V1VolumeMount(
                    name="mount-0",
                    mount_path="/dst",
                    read_only=True,
                ),
            ],
        )

    def test_device_mounts(self) -> None:
        scheduler = create_scheduler("test")
        from kubernetes.client.models import (
            V1HostPathVolumeSource,
            V1Volume,
            V1VolumeMount,
        )

        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.DeviceMount(src_path="foo", dst_path="bar", permissions="rwm"),
                specs.DeviceMount(src_path="foo2", dst_path="bar2", permissions="r"),
            ],
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.volumes[1:],
            [
                V1Volume(
                    name="mount-0",
                    host_path=V1HostPathVolumeSource(
                        path="foo",
                    ),
                ),
                V1Volume(
                    name="mount-1",
                    host_path=V1HostPathVolumeSource(
                        path="foo2",
                    ),
                ),
            ],
        )
        self.assertEqual(
            pod.spec.containers[0].volume_mounts[1:],
            [
                V1VolumeMount(
                    name="mount-0",
                    mount_path="bar",
                    read_only=False,
                ),
                V1VolumeMount(
                    name="mount-1",
                    mount_path="bar2",
                    read_only=True,
                ),
            ],
        )
        self.assertTrue(pod.spec.containers[0].security_context.privileged)

    def test_efa_device_override(self) -> None:
        """Test EFA device count can be overridden via efa_device_count parameter."""
        role_with_efa = specs.Role(
            name="foo",
            image="",
            resource=specs.Resource(
                cpu=2,
                memMB=3000,
                gpu=4,
                devices={"vpc.amazonaws.com/efa": 4},
            ),
        )
        role_without_efa = specs.Role(
            name="foo",
            image="",
            resource=specs.Resource(cpu=2, memMB=3000, gpu=4),
        )

        # Default: use resource spec's EFA count (or no EFA if not in spec)
        pod = role_to_pod("foo", role_with_efa, service_account="")
        self.assertEqual(
            pod.spec.containers[0].resources.limits["vpc.amazonaws.com/efa"], "4"
        )

        pod = role_to_pod("foo", role_without_efa, service_account="")
        self.assertNotIn(
            "vpc.amazonaws.com/efa", pod.spec.containers[0].resources.limits
        )

        # Override to 0: remove EFA entirely
        pod = role_to_pod("foo", role_with_efa, service_account="", efa_device_count=0)
        self.assertNotIn(
            "vpc.amazonaws.com/efa", pod.spec.containers[0].resources.limits
        )

        # Override to different count: use override value
        pod = role_to_pod("foo", role_with_efa, service_account="", efa_device_count=8)
        self.assertEqual(
            pod.spec.containers[0].resources.limits["vpc.amazonaws.com/efa"], "8"
        )

        # Add EFA when not in resource spec
        pod = role_to_pod(
            "foo", role_without_efa, service_account="", efa_device_count=32
        )
        self.assertEqual(
            pod.spec.containers[0].resources.limits["vpc.amazonaws.com/efa"], "32"
        )

    def test_reserved_resources_override(self) -> None:
        """Test that reserved_millicpu and reserved_memmb overrides work correctly."""
        role = specs.Role(
            name="foo",
            image="",
            resource=specs.Resource(cpu=2, gpu=0, memMB=3000),
        )

        # Default: 100 millicpu and 1024 memmb reserved
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(pod.spec.containers[0].resources.limits["cpu"], "2000m")
        self.assertEqual(pod.spec.containers[0].resources.limits["memory"], "3000M")
        self.assertEqual(
            pod.spec.containers[0].resources.requests["cpu"], "1900m"
        )  # 2000 - 100
        self.assertEqual(
            pod.spec.containers[0].resources.requests["memory"], "1976M"
        )  # 3000 - 1024

        # Custom overrides for both CPU and memory
        pod = role_to_pod(
            "foo", role, service_account="", reserved_millicpu=300, reserved_memmb=1000
        )
        self.assertEqual(
            pod.spec.containers[0].resources.requests["cpu"], "1700m"
        )  # 2000 - 300
        self.assertEqual(
            pod.spec.containers[0].resources.requests["memory"], "2000M"
        )  # 3000 - 1000

        # Zero reserved: requests equal limits
        pod = role_to_pod(
            "foo", role, service_account="", reserved_millicpu=0, reserved_memmb=0
        )
        self.assertEqual(pod.spec.containers[0].resources.requests["cpu"], "2000m")
        self.assertEqual(pod.spec.containers[0].resources.requests["memory"], "3000M")

    def test_instance_type(self) -> None:
        scheduler = create_scheduler("test")
        role = specs.Role(
            name="foo",
            image="",
            mounts=[],
            resource=specs.Resource(
                cpu=4,
                memMB=4000,
                gpu=8,
                capabilities={
                    LABEL_INSTANCE_TYPE: "some_instance",
                },
            ),
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.node_selector,
            {
                "node.kubernetes.io/instance-type": "some_instance",
            },
        )

    def test_rank0_env(self) -> None:
        from kubernetes.client.models import V1EnvVar

        scheduler = create_scheduler("test")
        app = _test_app(num_replicas=2)
        cfg = Opts(queue="testqueue")
        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler.submit_dryrun(app, cfg)

        tasks = info.request.resource["spec"]["tasks"]
        container0 = tasks[0]["template"].spec.containers[0]
        self.assertIn("TORCHX_RANK0_HOST", container0.command)
        self.assertIn(
            V1EnvVar(name="TORCHX_RANK0_HOST", value="localhost"), container0.env
        )
        self.assertIn(
            V1EnvVar(name="TORCHX_IMAGE", value="pytorch/torchx:latest"), container0.env
        )
        container1 = tasks[1]["template"].spec.containers[0]
        self.assertIn("VC_TRAINERFOO_0_HOSTS", container1.command)
        self.mock_create.assert_called_once()
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["dry_run"], "All")
        self.assertEqual(call_kwargs["namespace"], "default")

    def test_submit_dryrun_patch(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].image = "sha256:testhash"
        cfg = Opts(queue="testqueue", image_repo="example.com/some/repo")
        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler.submit_dryrun(app, cfg)

        self.assertIn("example.com/some/repo:testhash", str(info.request.resource))
        self.assertEqual(
            info.request.images_to_push,
            {
                "sha256:testhash": (
                    "example.com/some/repo",
                    "testhash",
                ),
            },
        )
        self.mock_create.assert_called_once()
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["dry_run"], "All")

    def test_submit_dryrun_service_account(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("service_account", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = Opts(queue="testqueue", service_account="srvacc")
        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn("'service_account_name': 'srvacc'", str(info.request.resource))

        cfg = Opts(queue="testqueue")
        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn("service_account_name': None", str(info.request.resource))

        self.assertEqual(self.mock_create.call_count, 2)
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["dry_run"], "All")

    def test_submit_dryrun_priority_class(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("priority_class", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = Opts(queue="testqueue", priority_class="high")

        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn("'priorityClassName': 'high'", str(info.request.resource))

        cfg = Opts(queue="testqueue")
        info = scheduler.submit_dryrun(app, cfg)
        self.assertNotIn("'priorityClassName'", str(info.request.resource))

        self.assertEqual(self.mock_create.call_count, 2)
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["dry_run"], "All")

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit(self, create_namespaced_custom_object: MagicMock) -> None:
        create_namespaced_custom_object.return_value = {
            "metadata": {"name": "testid"},
        }
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = Opts(namespace="testnamespace", queue="testqueue")

        info = scheduler.submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "testnamespace:testid")
        call = create_namespaced_custom_object.call_args
        args, kwargs = call
        self.assertEqual(kwargs["group"], "batch.volcano.sh")
        self.assertEqual(kwargs["version"], "v1alpha1")
        self.assertEqual(kwargs["namespace"], "testnamespace")
        self.assertEqual(kwargs["plural"], "jobs")
        self.assertEqual(kwargs["body"], info.request.resource)

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit_job_name_conflict(
        self, create_namespaced_custom_object: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=409, reason="Conflict")
        api_exc.body = '{"details":{"name": "test_job"}}'
        create_namespaced_custom_object.side_effect = [{}, api_exc]

        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = Opts(namespace="testnamespace", queue="testqueue")
        info = scheduler.submit_dryrun(app, cfg)
        with self.assertRaises(ValueError):
            scheduler.schedule(info)

        self.assertEqual(create_namespaced_custom_object.call_count, 2)
        # First call is spec validation
        first_call_kwargs = create_namespaced_custom_object.call_args_list[0][1]
        self.assertEqual(first_call_kwargs["dry_run"], "All")
        # Second call is actual schedule
        second_call_kwargs = create_namespaced_custom_object.call_args_list[1][1]
        self.assertNotIn("dry_run", second_call_kwargs)

    @patch("kubernetes.client.CoreV1Api")
    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe(
        self,
        get_namespaced_custom_object_status: MagicMock,
        mock_core_api_class: MagicMock,
    ) -> None:
        get_namespaced_custom_object_status.return_value = {
            "status": {
                "state": {"phase": "Completed"},
                "succeeded": 1,
                "taskStatusCount": {"echo-0": {"phase": {"Succeeded": 1}}},
            }
        }
        # Mock the pod response with a pod IP
        mock_pod = MagicMock()
        mock_pod.status.pod_ip = "10.244.1.5"
        mock_core_api_instance = MagicMock()
        mock_core_api_instance.read_namespaced_pod.return_value = mock_pod
        mock_core_api_class.return_value = mock_core_api_instance

        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        call = get_namespaced_custom_object_status.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "batch.volcano.sh",
                "version": "v1alpha1",
                "namespace": "testnamespace",
                "plural": "jobs",
                "name": "testid",
            },
        )
        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.SUCCEEDED,
                roles_statuses=[
                    specs.RoleStatus(
                        "echo",
                        [
                            specs.ReplicaStatus(
                                id=0,
                                role="echo",
                                state=specs.AppState.SUCCEEDED,
                                hostname="10-244-1-5.testnamespace.pod.cluster.local",
                            )
                        ],
                    ),
                ],
                roles=[
                    specs.Role(name="echo", image="", num_replicas=1),
                ],
            ),
        )

    @patch("kubernetes.client.CoreV1Api")
    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_completing_state(
        self,
        get_namespaced_custom_object_status: MagicMock,
        mock_core_api_class: MagicMock,
    ) -> None:
        get_namespaced_custom_object_status.return_value = {
            "status": {
                "state": {"phase": "Completing"},
                "taskStatusCount": {"echo-0": {"phase": {"Running": 1}}},
            }
        }
        mock_pod = MagicMock()
        mock_pod.status.pod_ip = "10.244.1.5"
        mock_core_api_instance = MagicMock()
        mock_core_api_instance.read_namespaced_pod.return_value = mock_pod
        mock_core_api_class.return_value = mock_core_api_instance

        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        self.assertIsNotNone(info)
        self.assertEqual(info.state, specs.AppState.RUNNING)

    @patch("kubernetes.client.CoreV1Api")
    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_pod_ip_not_assigned(
        self,
        get_namespaced_custom_object_status: MagicMock,
        mock_core_api_class: MagicMock,
    ) -> None:
        """Test that describe() returns empty hostname if pod IP is not yet assigned."""
        get_namespaced_custom_object_status.return_value = {
            "status": {
                "state": {"phase": "Pending"},
                "succeeded": 0,
                "taskStatusCount": {"echo-0": {"phase": {"Pending": 1}}},
            }
        }
        # Mock the pod response - returns None for pod_ip
        mock_pod_no_ip = MagicMock()
        mock_pod_no_ip.status.pod_ip = None

        mock_core_api_instance = MagicMock()
        mock_core_api_instance.read_namespaced_pod.return_value = mock_pod_no_ip
        mock_core_api_class.return_value = mock_core_api_instance

        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")

        info = scheduler.describe(app_id)

        # Should only call once (no retries)
        self.assertEqual(mock_core_api_instance.read_namespaced_pod.call_count, 1)
        # hostname should be empty since pod IP is not yet assigned
        self.assertIsNotNone(info)
        self.assertEqual(info.roles_statuses[0].replicas[0].hostname, "")

    @patch("kubernetes.client.CoreV1Api")
    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_pod_not_found(
        self,
        get_namespaced_custom_object_status: MagicMock,
        mock_core_api_class: MagicMock,
    ) -> None:
        """Test that describe() returns empty hostname if pod is not found (ApiException)."""
        from kubernetes.client.rest import ApiException

        get_namespaced_custom_object_status.return_value = {
            "status": {
                "state": {"phase": "Running"},
                "succeeded": 0,
                "taskStatusCount": {"echo-0": {"phase": {"Running": 1}}},
            }
        }
        # Mock the pod lookup to raise ApiException (pod not found)
        mock_core_api_instance = MagicMock()
        mock_core_api_instance.read_namespaced_pod.side_effect = ApiException(
            status=404, reason="Not Found"
        )
        mock_core_api_class.return_value = mock_core_api_instance

        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")

        info = scheduler.describe(app_id)

        # Verify the pod lookup was attempted
        self.assertEqual(mock_core_api_instance.read_namespaced_pod.call_count, 1)
        # hostname should be empty since pod was not found
        self.assertIsNotNone(info)
        self.assertEqual(info.roles_statuses[0].replicas[0].hostname, "")
        # App state should still be set correctly
        self.assertEqual(info.state, specs.AppState.RUNNING)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_unknown(
        self, get_namespaced_custom_object_status: MagicMock
    ) -> None:
        get_namespaced_custom_object_status.return_value = {}
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        call = get_namespaced_custom_object_status.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "batch.volcano.sh",
                "version": "v1alpha1",
                "namespace": "testnamespace",
                "plural": "jobs",
                "name": "testid",
            },
        )
        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.UNKNOWN,
            ),
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_api_exception_404(
        self, get_namespaced_custom_object_status: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=404, reason="Not Found")
        get_namespaced_custom_object_status.side_effect = api_exc
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        self.assertIsNone(info)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_api_exception_other(
        self, get_namespaced_custom_object_status: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=500, reason="Internal Server Error")
        get_namespaced_custom_object_status.side_effect = api_exc
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        with self.assertRaises(ApiException):
            scheduler.describe(app_id)

    def test_runopts(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        runopts = scheduler.run_opts()
        self.assertEqual(
            set(runopts._opts.keys()),
            {
                "quiet",
                "queue",
                "namespace",
                "image_repo",
                "service_account",
                "priority_class",
                "validate_spec",
                "reserved_millicpu",
                "reserved_memmb",
                "efa_device_count",
            },
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    @patch("kubernetes.client.CustomObjectsApi.replace_namespaced_custom_object_status")
    def test_cancel_existing(
        self,
        replace_namespaced_custom_object_status: MagicMock,
        get_namespaced_custom_object: MagicMock,
    ) -> None:
        scheduler = create_scheduler("test")
        get_namespaced_custom_object.return_value = {
            "status": {"state": {"phase": "Running"}}
        }
        scheduler._cancel_existing("testnamespace:testjob")
        get_namespaced_custom_object.assert_called_once_with(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace="testnamespace",
            plural="jobs",
            name="testjob",
        )
        call = replace_namespaced_custom_object_status.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "batch.volcano.sh",
                "version": "v1alpha1",
                "namespace": "testnamespace",
                "plural": "jobs",
                "name": "testjob",
                "body": {"status": {"state": {"phase": "Aborted"}}},
            },
        )

    @patch("kubernetes.client.CustomObjectsApi.delete_namespaced_custom_object")
    @patch("torchx.schedulers.kubernetes_scheduler.KubernetesScheduler.exists")
    def test_delete(
        self, exists: MagicMock, delete_namespaced_custom_object: MagicMock
    ) -> None:
        scheduler = create_scheduler("test")
        exists.return_value = True
        scheduler.delete("testnamespace:testjob")
        delete_namespaced_custom_object.assert_called_once_with(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace="testnamespace",
            plural="jobs",
            name="testjob",
        )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list(self, list_namespaced_custom_object: MagicMock) -> None:
        with patch(
            "torchx.schedulers.kubernetes_scheduler.KubernetesScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]
            scheduler = create_scheduler("test")

            scheduler.list()
            call = list_namespaced_custom_object.call_args
            args, kwargs = call

            self.assertEqual(
                kwargs,
                {
                    "group": "batch.volcano.sh",
                    "version": "v1alpha1",
                    "namespace": "default",
                    "plural": "jobs",
                    "timeout_seconds": 30,
                },
            )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list_values(self, list_namespaced_custom_object: MagicMock) -> None:
        list_namespaced_custom_object.return_value = {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "items": [
                {
                    "apiVersion": "batch.volcano.sh/v1alpha1",
                    "kind": "Job",
                    "metadata": {
                        "creationTimestamp": "2021-10-11T20:49:35Z",
                        "name": "cifar-trainer-something",
                        "namespace": "default",
                        "resourceVersion": "100000000",
                        "selfLink": "/apis/batch.volcano.sh/v1alpha1/namespaces/default/jobs/cifar-trainer-something",
                        "uid": "ab6a11d3-aaaa-aaaa-aaaa-88220d5190ee",
                    },
                    "status": {
                        "runningDuration": "3262h8m50.910883962s",
                        "state": {
                            "lastTransitionTime": "2021-10-11T20:52:08Z",
                            "phase": "Completed",
                        },
                        "succeeded": 2,
                    },
                },
                {
                    "apiVersion": "batch.volcano.sh/v1alpha1",
                    "kind": "Job",
                    "metadata": {
                        "creationTimestamp": "2021-10-11T20:49:35Z",
                        "name": "test-trainer",
                        "namespace": "default",
                        "resourceVersion": "100000000",
                        "selfLink": "/apis/batch.volcano.sh/v1alpha1/namespaces/default/jobs/test-trainer",
                        "uid": "ab6a11d3-bbbb-bbbb-bbbb-88220d5190ee",
                    },
                    "status": {
                        "runningDuration": "3262h8m50.910883962s",
                        "state": {
                            "lastTransitionTime": "2021-10-11T20:52:08Z",
                            "phase": "Terminated",
                        },
                    },
                },
            ],
        }
        with patch(
            "torchx.schedulers.kubernetes_scheduler.KubernetesScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]

            scheduler = create_scheduler("test")

            apps = scheduler.list()
            call = list_namespaced_custom_object.call_args
            args, kwargs = call

            self.assertEqual(
                kwargs,
                {
                    "group": "batch.volcano.sh",
                    "version": "v1alpha1",
                    "namespace": "default",
                    "plural": "jobs",
                    "timeout_seconds": 30,
                },
            )
            self.assertEqual(
                apps,
                [
                    ListAppResponse(
                        app_id="default:cifar-trainer-something",
                        state=AppState.SUCCEEDED,
                    ),
                    ListAppResponse(
                        app_id="default:test-trainer", state=AppState.FAILED
                    ),
                ],
            )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list_failure(self, list_namespaced_custom_object: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(
            status=404, reason="Invalid kube-config file. No configuration found."
        )
        list_namespaced_custom_object.side_effect = api_exc
        with patch(
            "torchx.schedulers.kubernetes_scheduler.KubernetesScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]
            scheduler = create_scheduler("test")
            with self.assertRaises(ApiException):
                scheduler.list()

    @patch("kubernetes.client.CoreV1Api.read_namespaced_pod_log")
    def test_log_iter(self, read_namespaced_pod_log: MagicMock) -> None:
        scheduler = create_scheduler("test")
        read_namespaced_pod_log.return_value = "foo reg\nfoo\nbar reg\n"
        lines = scheduler.log_iter(
            app_id="testnamespace:testjob",
            role_name="role_blah",
            k=1,
            regex="reg",
            since=datetime.now(),
        )
        self.assertEqual(
            list(lines),
            [
                "foo reg\n",
                "bar reg\n",
            ],
        )
        call = read_namespaced_pod_log.call_args
        args, kwargs = call
        self.assertGreaterEqual(kwargs["since_seconds"], 0)
        del kwargs["since_seconds"]
        self.assertEqual(
            kwargs,
            {
                "namespace": "testnamespace",
                "name": "testjob-roleblah-1-0",
                "timestamps": True,
            },
        )

    @patch("kubernetes.watch.Watch.stream")
    def test_log_iter_tail(self, watch_stream: MagicMock) -> None:
        scheduler = create_scheduler("test")
        watch_stream.return_value = iter(["line1", "line2", "line3"])
        lines = scheduler.log_iter(
            app_id="testnamespace:testjob",
            role_name="role_blah",
            k=1,
            should_tail=True,
        )
        self.assertEqual(
            list(lines),
            [
                "line1\n",
                "line2\n",
                "line3\n",
            ],
        )

    def test_push_patches(self) -> None:
        # Configure mock to return proper response for schedule() call
        self.mock_create.return_value = {"metadata": {"name": "testjob"}}

        client = MagicMock()
        scheduler = KubernetesScheduler(
            "foo",
            client=MagicMock(),
            docker_client=client,
        )

        job = KubernetesJob(
            images_to_push={
                "sha256:testimage": ("repo.com/img", "testimage"),
            },
            resource={},
        )

        out = scheduler.schedule(AppDryRunInfo(job, repr))
        self.assertTrue(out)

        self.assertEqual(client.images.get.call_count, 1)
        self.assertEqual(client.images.get().tag.call_count, 1)
        self.assertEqual(client.images.push.call_count, 1)

    def test_min_replicas(self) -> None:
        app = _test_app(num_replicas=3)
        app.roles[0].min_replicas = 2

        resource = app_to_resource(app, "test_queue", service_account=None)
        min_available = [task["minAvailable"] for task in resource["spec"]["tasks"]]
        self.assertEqual(min_available, [1, 1, 0])

    def test_validate_spec_invalid_name(self) -> None:
        from kubernetes.client.rest import ApiException

        scheduler = create_scheduler("test")
        app = _test_app()
        app.name = "Invalid_Name"

        # Override the default mock behavior for this test
        self.mock_create.side_effect = ApiException(
            status=422,
            reason="Invalid",
        )

        cfg = Opts(queue="testqueue", validate_spec=True)

        with self.assertRaises(ValueError) as ctx:
            scheduler.submit_dryrun(app, cfg)

        self.assertIn("Invalid job spec", str(ctx.exception))
        self.mock_create.assert_called_once()
        call_kwargs = self.mock_create.call_args[1]
        self.assertEqual(call_kwargs["dry_run"], "All")

    def test_validate_spec_disabled(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()

        cfg = Opts(queue="testqueue", validate_spec=False)

        info = scheduler.submit_dryrun(app, cfg)

        self.assertIsNotNone(info)
        self.mock_create.assert_not_called()

    def test_validate_spec_invalid_task_name(self) -> None:
        from kubernetes.client.rest import ApiException

        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].name = "Invalid-Task-Name"

        # Override the default mock behavior for this test
        self.mock_create.side_effect = ApiException(
            status=422,
            reason="Invalid",
        )

        cfg = Opts(queue="testqueue", validate_spec=True)
        with self.assertRaises(ValueError) as ctx:
            scheduler.submit_dryrun(app, cfg)
            self.assertIn("Invalid job spec", str(ctx.exception))

    def test_apply_pod_overlay_dict_merge(self) -> None:
        """Overlay merges nodeSelector, adds tolerations and affinity."""
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="test", image="test:latest")],
                node_selector={"existing": "label"},
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                "nodeSelector": {"gpu": "true"},
                "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
                "affinity": {
                    "nodeAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": {
                            "nodeSelectorTerms": [
                                {
                                    "matchExpressions": [
                                        {
                                            "key": "gpu",
                                            "operator": "In",
                                            "values": ["true"],
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                },
            }
        }

        _apply_pod_overlay(pod, overlay)

        # Existing + new nodeSelector merged
        self.assertEqual(pod.spec.node_selector, {"existing": "label", "gpu": "true"})
        # New fields added
        self.assertEqual(len(pod.spec.tolerations), 1)
        self.assertEqual(pod.spec.tolerations[0].key, "nvidia.com/gpu")
        self.assertIsNotNone(pod.spec.affinity)
        self.assertIsNotNone(pod.spec.affinity.node_affinity)

    def test_submit_dryrun_with_pod_overlay(self) -> None:
        scheduler = create_scheduler("test")

        trainer_role = specs.Role(
            name="trainer",
            image="pytorch/torchx:latest",
            entrypoint="main",
            resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
        )
        set_overlay(
            trainer_role,
            "kubernetes",
            "V1Pod",
            {"spec": {"nodeSelector": {"gpu": "true"}}},
        )
        app = specs.AppDef("test", roles=[trainer_role])
        cfg = Opts(queue="testqueue")

        info = scheduler.submit_dryrun(app, cfg)
        resource = info.request.resource

        # Check that overlay was applied to all pods
        tasks = resource["spec"]["tasks"]
        for task in tasks:
            pod = task["template"]
            self.assertIn("gpu", pod.spec.node_selector)
            self.assertEqual(pod.spec.node_selector["gpu"], "true")

    def test_submit_dryrun_with_pod_overlay_file_uri(self) -> None:
        import tempfile

        import yaml

        scheduler = create_scheduler("test")

        # Create overlay file
        overlay = {"spec": {"nodeSelector": {"instance-type": "p4d.24xlarge"}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(overlay, f)
            overlay_path = f.name

        try:
            # Create app with file URI
            trainer_role = specs.Role(
                name="trainer",
                image="pytorch/torchx:latest",
                entrypoint="main",
                resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
                metadata={"kubernetes": f"file://{overlay_path}"},
            )
            app = specs.AppDef("test", roles=[trainer_role])
            cfg = Opts(queue="testqueue")

            info = scheduler.submit_dryrun(app, cfg)
            resource = info.request.resource

            # Check that overlay was applied
            tasks = resource["spec"]["tasks"]
            for task in tasks:
                pod = task["template"]
                self.assertIn("instance-type", pod.spec.node_selector)
                self.assertEqual(
                    pod.spec.node_selector["instance-type"], "p4d.24xlarge"
                )
        finally:
            import os

            os.unlink(overlay_path)

    def test_apply_pod_overlay_list_append(self) -> None:
        from kubernetes.client.models import (
            V1Container,
            V1ObjectMeta,
            V1Pod,
            V1PodSpec,
            V1Toleration,
        )
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="test", image="test:latest")],
                tolerations=[V1Toleration(key="existing", operator="Exists")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(len(pod.spec.tolerations), 2)
        self.assertEqual(pod.spec.tolerations[0].key, "existing")
        self.assertEqual(pod.spec.tolerations[1].key, "nvidia.com/gpu")

    def test_apply_pod_overlay_list_replace_tuple(self) -> None:
        from kubernetes.client.models import (
            V1Container,
            V1ObjectMeta,
            V1Pod,
            V1PodSpec,
            V1Toleration,
        )
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="test", image="test:latest")],
                tolerations=[V1Toleration(key="existing", operator="Exists")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                "tolerations": ({"key": "nvidia.com/gpu", "operator": "Exists"},),
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(len(pod.spec.tolerations), 1)
        self.assertEqual(pod.spec.tolerations[0].key, "nvidia.com/gpu")

    def test_apply_pod_overlay_container_new_name_appends(self) -> None:
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="trainer-0", image="pytorch:latest")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                JOIN("containers", on="name"): [
                    {
                        "name": "sidecar",
                        "image": "sidecar:latest",
                    },
                ],
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.containers), 2, "expected 2 containers after append"
        )
        self.assertEqual(pod.spec.containers[0].name, "trainer-0")
        self.assertEqual(pod.spec.containers[1].name, "sidecar")
        self.assertEqual(pod.spec.containers[1].image, "sidecar:latest")

    def test_apply_pod_overlay_container_merge_does_not_mutate_overlay(self) -> None:
        import copy

        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="trainer-0", image="pytorch:latest")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                JOIN("containers", on="name"): [
                    {
                        "name": "trainer-0",
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8080},
                        },
                    },
                ],
                "nodeSelector": {"gpu": "true"},
            }
        }
        overlay_before = copy.deepcopy(overlay)

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            overlay,
            overlay_before,
            "overlay dict should not be mutated by _apply_pod_overlay",
        )

    def test_apply_pod_overlay_container_replace_tuple(self) -> None:
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[
                    V1Container(name="app", image="pytorch:latest"),
                    V1Container(name="log-collector", image="pytorch:latest"),
                ],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        # Tuple = replace entire list, bypass strategic merge
        overlay = {
            "spec": {
                "containers": ({"name": "only-one", "image": "new:latest"},),
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.containers), 1, "tuple should replace entire list"
        )
        self.assertEqual(pod.spec.containers[0].name, "only-one")

    def test_apply_pod_overlay_init_container_merge_by_name(self) -> None:
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="main", image="main:latest")],
                init_containers=[V1Container(name="init", image="init:latest")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                JOIN("initContainers", on="name"): [
                    {"name": "init", "command": ["/bin/setup"]},
                ],
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.init_containers),
            1,
            "initContainer should be merged, not duplicated",
        )
        self.assertEqual(pod.spec.init_containers[0].image, "init:latest")
        self.assertEqual(pod.spec.init_containers[0].command, ["/bin/setup"])

    def test_apply_pod_overlay_container_mixed_match_and_append(self) -> None:
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="trainer-0", image="pytorch:latest")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                JOIN("containers", on="name"): [
                    {
                        "name": "trainer-0",
                        "livenessProbe": {
                            "httpGet": {"path": "/health", "port": 8080},
                        },
                    },
                    {
                        "name": "sidecar",
                        "image": "sidecar:latest",
                    },
                ],
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.containers),
            2,
            "expected merge + append = 2 containers",
        )
        self.assertEqual(pod.spec.containers[0].name, "trainer-0")
        self.assertEqual(pod.spec.containers[0].image, "pytorch:latest")
        self.assertIsNotNone(
            pod.spec.containers[0].liveness_probe,
            "livenessProbe should be merged into existing container",
        )
        self.assertEqual(pod.spec.containers[1].name, "sidecar")
        self.assertEqual(pod.spec.containers[1].image, "sidecar:latest")

    def test_apply_pod_overlay_container_empty_list(self) -> None:
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="trainer-0", image="pytorch:latest")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {"spec": {"containers": []}}

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.containers),
            1,
            "empty overlay containers should not change base",
        )
        self.assertEqual(pod.spec.containers[0].name, "trainer-0")

    def test_apply_pod_overlay_put_replaces_containers(self) -> None:
        """PUT replaces containers list entirely."""
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[
                    V1Container(name="app", image="pytorch:latest"),
                    V1Container(name="log-collector", image="pytorch:latest"),
                ],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        overlay = {
            "spec": {
                PUT("containers"): [{"name": "only-one", "image": "new:latest"}],
            },
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.containers),
            1,
            "PUT should replace containers list entirely",
        )
        self.assertEqual(pod.spec.containers[0].name, "only-one")
        self.assertEqual(pod.spec.containers[0].image, "new:latest")

    def test_apply_pod_overlay_no_join_appends(self) -> None:
        """Without JOIN, container lists are appended (not merged)."""
        from kubernetes.client.models import V1Container, V1ObjectMeta, V1Pod, V1PodSpec
        from torchx.schedulers.kubernetes_scheduler import _apply_pod_overlay

        pod = V1Pod(
            spec=V1PodSpec(
                containers=[V1Container(name="trainer-0", image="pytorch:latest")],
            ),
            metadata=V1ObjectMeta(name="test-pod"),
        )

        # No JOIN -> append semantics (even for same name)
        overlay = {
            "spec": {
                "containers": [
                    {"name": "trainer-0", "memory": "1Gi"},
                ],
            }
        }

        _apply_pod_overlay(pod, overlay)

        self.assertEqual(
            len(pod.spec.containers),
            2,
            "without JOIN, containers should be appended, not merged",
        )

    def test_submit_dryrun_with_pod_overlay_yaml_replace_tag(self) -> None:
        import tempfile

        scheduler = create_scheduler("test")

        overlay_yaml = """
spec:
  tolerations: !!python/tuple
    - key: nvidia.com/gpu
      operator: Exists
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(overlay_yaml)
            overlay_path = f.name

        try:
            trainer_role = specs.Role(
                name="trainer",
                image="pytorch/torchx:latest",
                entrypoint="main",
                resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
                metadata={"kubernetes": f"file://{overlay_path}"},
            )
            app = specs.AppDef("test", roles=[trainer_role])
            cfg = Opts(queue="testqueue")

            info = scheduler.submit_dryrun(app, cfg)
            resource = info.request.resource

            tasks = resource["spec"]["tasks"]
            for task in tasks:
                pod = task["template"]
                self.assertEqual(len(pod.spec.tolerations), 1)
                self.assertEqual(pod.spec.tolerations[0].key, "nvidia.com/gpu")
        finally:
            import os

            os.unlink(overlay_path)

    def test_submit_dryrun_with_pod_overlay_invalid_type(self) -> None:
        scheduler = create_scheduler("test")

        # Non-dict, non-str metadata is silently ignored by get_overlay
        trainer_role = specs.Role(
            name="trainer",
            image="pytorch/torchx:latest",
            entrypoint="main",
            resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
            metadata={"kubernetes": 123},  # Invalid type, silently skipped
        )
        app = specs.AppDef("test", roles=[trainer_role])
        cfg = Opts(queue="testqueue")

        # Should not raise — overlay is silently skipped
        info = scheduler.submit_dryrun(app, cfg)
        self.assertIsNotNone(
            info, "dryrun should succeed even with invalid overlay type"
        )

    def test_submit_dryrun_strategic_merge_probes(self) -> None:
        """End-to-end: JOIN merges liveness/readiness probes into main container."""
        scheduler = create_scheduler("test")

        trainer_role = specs.Role(
            name="trainer",
            image="pytorch/torchx:latest",
            entrypoint="main",
            resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
        )
        set_overlay(
            trainer_role,
            "kubernetes",
            "V1Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {
                            "name": "trainer-0",
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                                "initialDelaySeconds": 30,
                            },
                            "readinessProbe": {
                                "httpGet": {"path": "/ready", "port": 8080},
                            },
                        }
                    ],
                }
            },
        )
        app = specs.AppDef("test", roles=[trainer_role])
        cfg = Opts(queue="testqueue")

        info = scheduler.submit_dryrun(app, cfg)
        tasks = info.request.resource["spec"]["tasks"]
        for task in tasks:
            pod = task["template"]
            containers = pod.spec.containers
            self.assertEqual(len(containers), 1, "should still have 1 container")
            c = containers[0]
            self.assertIsNotNone(
                c.liveness_probe, "livenessProbe should be set from overlay"
            )
            self.assertEqual(c.liveness_probe.initial_delay_seconds, 30)
            self.assertIsNotNone(
                c.readiness_probe, "readinessProbe should be set from overlay"
            )
            self.assertEqual(c.readiness_probe.http_get.path, "/ready")

    def test_submit_dryrun_flat_overlay_bc(self) -> None:
        """BC: flat metadata["kubernetes"] = {overlay} still applies via get_overlay."""
        scheduler = create_scheduler("test")

        trainer_role = specs.Role(
            name="trainer",
            image="pytorch/torchx:latest",
            entrypoint="main",
            resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
            metadata={"kubernetes": {"spec": {"nodeSelector": {"gpu": "true"}}}},
        )
        app = specs.AppDef("test", roles=[trainer_role])
        cfg = Opts(queue="testqueue")

        info = scheduler.submit_dryrun(app, cfg)
        tasks = info.request.resource["spec"]["tasks"]
        for task in tasks:
            pod = task["template"]
            self.assertIn(
                "gpu",
                pod.spec.node_selector,
                "flat overlay should still be applied via BC path",
            )
            self.assertEqual(pod.spec.node_selector["gpu"], "true")

    def test_validate_spec_long_pod_name(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.name = "x" * 50
        app.roles[0].name = "y" * 20

        cfg = Opts(queue="testqueue", validate_spec=True)

        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "x" * 50
            with self.assertRaises(ValueError) as ctx:
                scheduler.submit_dryrun(app, cfg)

        self.assertIn("Pod name", str(ctx.exception))
        self.assertIn("exceeds 63 character limit", str(ctx.exception))

    def test_pod_label(self) -> None:
        _UNUSED = "__UNUSED__"

        app = specs.AppDef(
            name="foo+bar",
            roles=[specs.Role(name="a/b", image=_UNUSED)],
        )
        app_id = normalize_str(make_unique(app.name))
        labels = kubernetes_scheduler.pod_labels(
            app=app,
            role_idx=0,
            role=app.roles[0],
            replica_id=1,
            app_id=app_id,
        )

        self.assertDictEqual(
            labels,
            {
                # torchx version complies with PEP-440
                # while typically it is 0.x.x or 0.x.xdev0
                # there could be org specific builds that are of the form
                # 0.x.xdev0+org_name (e.g. 0.8.0dev0+fb)
                # "+" is not a valid pod label char
                # we expect that the version str would've been "cleaned"
                # to replace invalid chars with "." (a valid char)
                LABEL_VERSION: torchx.__version__.replace("+", "."),
                LABEL_APP_NAME: "foo.bar",
                LABEL_ROLE_INDEX: "0",
                LABEL_ROLE_NAME: "a.b",
                LABEL_REPLICA_ID: "1",
                LABEL_KUBE_APP_NAME: "foo.bar",
                LABEL_ORGANIZATION: "torchx.pytorch.org",
                LABEL_UNIQUE_NAME: app_id,
            },
        )


class KubernetesSchedulerNoImportTest(unittest.TestCase):
    """
    KubernetesSchedulerNoImportTest tests the kubernetes scheduler behavior when
    Kubernetes is not available.
    """

    def setUp(self) -> None:
        # make all kubernetes modules unable to be imported
        for mod in list(sys.modules.keys()) + ["kubernetes"]:
            if mod.startswith("kubernetes"):
                sys.modules[mod] = None  # pyre-ignore

        # reload to ensure kubernetes_scheduler doesn't depend on them at import
        # time
        importlib.reload(kubernetes_scheduler)
        importlib.reload(schedulers)

    def tearDown(self) -> None:
        # reset all kubernetes modules we patched
        for mod in list(sys.modules.keys()):
            if mod.startswith("kubernetes"):
                del sys.modules[mod]
        # reimport kubernetes_scheduler to get to a clean state
        importlib.reload(kubernetes_scheduler)

    def test_runopts(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        self.assertIsNotNone(scheduler.run_opts())

    def test_describe(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        with self.assertRaises(ModuleNotFoundError):
            scheduler.describe("foo:bar")

    def test_dryrun(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        app = _test_app()
        cfg = Opts(namespace="testnamespace", queue="testqueue")

        with self.assertRaises(ModuleNotFoundError):
            scheduler.submit_dryrun(app, cfg)
