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
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import torchx
from torchx import schedulers, specs

# @manual=//torchx/schedulers:kubernetes_mcad_scheduler
from torchx.schedulers import kubernetes_mcad_scheduler
from torchx.schedulers.api import DescribeAppResponse, ListAppResponse
from torchx.schedulers.docker_scheduler import has_docker
from torchx.schedulers.kubernetes_mcad_scheduler import (
    app_to_resource,
    cleanup_str,
    create_pod_group,
    create_scheduler,
    get_appwrapper_status,
    get_port_for_service,
    get_role_information,
    get_tasks_status_description,
    get_unique_truncated_appid,
    KubernetesMCADJob,
    KubernetesMCADOpts,
    KubernetesMCADScheduler,
    LABEL_INSTANCE_TYPE,
    mcad_svc,
    role_to_pod,
)
from torchx.specs import AppDryRunInfo, AppState, Resource, Role

SKIP_DOCKER: bool = not has_docker()

TEST_KUBE_CONFIG: Dict[str, Any] = {
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


def _test_mcad_generic_item() -> Dict[str, Any]:
    generic_item = {
        "status": {
            "state": "Running",
            "Succeeded": 1,
        },
        "spec": {
            "resources": {
                "GenericItems": [
                    {
                        "generictemplate": {
                            "metadata": {
                                "labels": {
                                    "torchx.pytorch.org/role-name": "echo",
                                },
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "command": [
                                            "bash",
                                            "-c",
                                            "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                                        ],
                                        "env": {
                                            "name": "TORCH_DISTRIBUTED_DEBUG",
                                            "value": "DETAIL",
                                        },
                                        "image": "echoImage",
                                        "resources": {
                                            "limits": {
                                                "cpu": "1000m",
                                                "memory": "514000M",
                                                "gpu": "2",
                                            },
                                            "requests": {
                                                "cpu": 900,
                                                "memory": 512976,
                                                "gpu": 2,
                                            },
                                        },
                                        "ports": {"c10d": 29500},
                                        "volumeMounts": [
                                            specs.VolumeMount(
                                                src="dshm", dst_path="/dev/shm"
                                            )
                                        ],
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
        },
    }
    return generic_item


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
        env={"FOO": "bar"},
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


class KubernetesMCADSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        client = MagicMock()
        docker_client = MagicMock()
        scheduler = create_scheduler("foo", client=client, docker_client=docker_client)
        self.assertIsInstance(
            scheduler, kubernetes_mcad_scheduler.KubernetesMCADScheduler
        )
        self.assertEqual(client, scheduler._client)
        self.assertEqual(docker_client, scheduler._docker_client)

    def test_app_to_resource_resolved_macros(self) -> None:
        app = _test_app()
        unique_app_name = "app-name"
        with patch(
            "torchx.schedulers.kubernetes_mcad_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = unique_app_name
            resource = app_to_resource(
                app,
                "default",
                service_account=None,
                image_secret=None,
                coscheduler_name=None,
                priority_class_name=None,
                network=None,
                priority=0,
            )
            actual_cmd = (
                resource["spec"]["resources"]["GenericItems"][0]["generictemplate"]
                .spec.containers[0]
                .command
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

    def test_retry_not_set(self) -> None:
        app = _test_app()
        resource = app_to_resource(
            app,
            "default",
            service_account=None,
            image_secret=None,
            coscheduler_name=None,
            priority_class_name=None,
            network=None,
            priority=0,
        )

        item0 = resource["spec"]
        expected = {
            "minAvailable": 1,
            "requeuing": {
                "growthType": "exponential",
                "maxNumRequeuings": 3,
                "maxTimeInSeconds": 0,
                "timeInSeconds": 300,
            },
        }
        self.assertEqual(item0["schedulingSpec"], expected)

        for role in app.roles:
            role.max_retries = 0
        resource = app_to_resource(
            app,
            "default",
            service_account=None,
            image_secret=None,
            coscheduler_name=None,
            priority_class_name=None,
            network=None,
            priority=0,
        )
        item0 = resource["spec"]
        self.assertFalse("schedulingSpec" in item0)

    def test_role_to_pod(self) -> None:
        from kubernetes.client.models import (
            V1Container,
            V1ContainerPort,
            V1EmptyDirVolumeSource,
            V1EnvVar,
            V1HostPathVolumeSource,
            V1LocalObjectReference,
            V1ObjectMeta,
            V1Pod,
            V1PodSpec,
            V1ResourceRequirements,
            V1SecurityContext,
            V1Volume,
            V1VolumeMount,
        )

        app = _test_app()
        unique_app_name = "app-name"
        image_secret = "secret-name"
        coscheduler_name = "test-co-scheduler-name"
        priority_class_name = "default-priority"
        network = "test-network-conf"
        pod = role_to_pod(
            "app-name-0",
            unique_app_name,
            "test_namespace",
            app.roles[0],
            service_account="srvacc",
            image_secret=image_secret,
            coscheduler_name=coscheduler_name,
            priority_class_name=priority_class_name,
            network=network,
        )
        imagesecret = V1LocalObjectReference(name=image_secret)

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
            name="app-name-0",
            env=[
                V1EnvVar(name="FOO", value="bar"),
                V1EnvVar(
                    name="TORCHX_MCAD_TRAINERFOO_0_HOSTS", value="app-name-0.app-name"
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
            api_version="v1",
            kind="Pod",
            spec=V1PodSpec(
                containers=[container],
                hostname="app-name-0",
                subdomain="app-name",
                image_pull_secrets=[imagesecret],
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
                scheduler_name=coscheduler_name,
                node_selector={},
                priority_class_name=priority_class_name,
            ),
            metadata=V1ObjectMeta(
                annotations={
                    "sidecar.istio.io/inject": "false",
                    "k8s.v1.cni.cncf.io/networks": "test-network-conf",
                },
                labels={},
                name="app-name-0",
                namespace="test_namespace",
            ),
        )

        self.assertEqual(
            pod,
            want,
        )

    def test_create_pod_group(self) -> None:
        app = _test_app()
        unique_app_name = "app-name"
        namespace = "default"
        podgroup = create_pod_group(app, app.roles[0], 0, namespace, unique_app_name)

        pod_group_name = unique_app_name + "-pg0"
        expected_pod_group: Dict[str, Any] = {
            "apiVersion": "scheduling.sigs.k8s.io/v1alpha1",
            "kind": "PodGroup",
            "metadata": {
                "name": pod_group_name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "test",
                    "app.kubernetes.io/managed-by": "torchx.pytorch.org",
                    "app.kubernetes.io/instance": "app-name",
                    "appwrapper.workload.codeflare.dev": unique_app_name,
                },
            },
            "spec": {
                "minMember": 1,
            },
        }
        want: Dict[str, Any] = {
            "replicas": 1,
            "generictemplate": expected_pod_group,
        }
        self.assertEqual(
            podgroup,
            want,
        )

    def test_create_mcad_service(self) -> None:
        from kubernetes.client.models import (  # noqa: F401 imported but unused
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

        app = _test_app()
        service_name = "test_service"
        service_port = "1234"
        namespace = "default"
        test_service = mcad_svc(app, service_name, namespace, service_port)

        want = V1Service(
            api_version="v1",
            kind="Service",
            metadata=V1ObjectMeta(
                name=service_name,
                namespace=namespace,
                labels={
                    "app.kubernetes.io/name": "test",
                    "app.kubernetes.io/managed-by": "torchx.pytorch.org",
                    "app.kubernetes.io/instance": "test_service",
                },
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
                selector={"app.kubernetes.io/instance": service_name},
                session_affinity="None",
                type="ClusterIP",
            ),
            status=V1ServiceStatus(
                load_balancer={},
            ),
        )
        self.assertEqual(
            test_service,
            want,
        )

    def test_validate(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        scheduler._validate(
            app,
            "kubernetes_mcad",
            cfg=KubernetesMCADOpts({"namespace": "test_namespace"}),
        )

    def test_cleanup_str(self) -> None:
        self.assertEqual("abcd123", cleanup_str("abcd123"))
        self.assertEqual("abcd123", cleanup_str("-/_a/b/CD!123!"))
        self.assertEqual("a-bcd123", cleanup_str("-a-bcd123"))
        self.assertEqual("", cleanup_str("!!!"))
        self.assertEqual("abcd1234", cleanup_str("1234abcd1234"))

    def test_get_unique_truncated_appid(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.name = "abcde"
        self.assertEqual(20, len(get_unique_truncated_appid(app)))
        self.assertIn(app.name, get_unique_truncated_appid(app))

        app.name = "abcdefghijklmnopqrstuvwxyz012345678910111213141516"
        self.assertEqual(56, len(get_unique_truncated_appid(app)))
        self.assertIn(app.name, get_unique_truncated_appid(app))

        app.name = (
            "abcdefghijklmnopqrstuvwxyz012345678910111213141516171819202122232425"
        )
        self.assertEqual(59, len(get_unique_truncated_appid(app)))
        self.assertIn(
            "abcdefghijklmnopqrstuvwxyz01234567891011121314151617181",
            get_unique_truncated_appid(app),
        )
        self.assertNotIn("9202122232425", get_unique_truncated_appid(app))

    def test_get_port_for_service(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        test_port = get_port_for_service(app)
        self.assertEqual(test_port, "1234")

    def test_no_port(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].port_map = {}
        test_port = get_port_for_service(app)
        self.assertEqual(test_port, "29500")

    def test_invalid_port(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].port_map = {"foo": 65536}
        test_port = get_port_for_service(app)
        self.assertEqual(test_port, "29500")

    def test_get_pending_appwrapper_status(self) -> None:
        test_item = {}
        aw_status = get_appwrapper_status(test_item)
        self.assertEqual(aw_status, AppState.PENDING)

    def test_get_tasks_status_description(self) -> None:
        test_status = {
            "state": "Running",
            "running": "1",
            "pending": "2",
            "failed": "3",
            "Succeeded": "4",
        }
        results = get_tasks_status_description(test_status)
        expect = {"running": "1", "pending": "2", "failed": "3", "Succeeded": "4"}
        self.assertEqual(results, expect)

    def test_submit_dryrun(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "priority": 0,
                "namespace": "test_namespace",
                "coscheduler_name": "test_coscheduler",
                "network": "test-network-conf",
            }
        )
        with patch(
            "torchx.schedulers.kubernetes_mcad_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name"
            info = scheduler._submit_dryrun(app, cfg)

        resource = str(info.request)

        self.assertEqual(
            resource,
            f"""apiVersion: workload.codeflare.dev/v1beta1
kind: AppWrapper
metadata:
  name: app-name
  namespace: test_namespace
spec:
  priority: 0
  resources:
    GenericItems:
    - generictemplate:
        apiVersion: scheduling.sigs.k8s.io/v1alpha1
        kind: PodGroup
        metadata:
          labels:
            app.kubernetes.io/instance: app-name
            app.kubernetes.io/managed-by: torchx.pytorch.org
            app.kubernetes.io/name: test
            appwrapper.workload.codeflare.dev: app-name
          name: app-name-pg0
          namespace: test_namespace
        spec:
          minMember: 1
      replicas: 1
    - generictemplate:
        apiVersion: v1
        kind: Pod
        metadata:
          annotations:
            k8s.v1.cni.cncf.io/networks: test-network-conf
            sidecar.istio.io/inject: 'false'
          labels:
            app.kubernetes.io/instance: app-name
            app.kubernetes.io/managed-by: torchx.pytorch.org
            app.kubernetes.io/name: test
            pod-group.scheduling.sigs.k8s.io: app-name-pg0
            torchx.pytorch.org/app-name: test
            torchx.pytorch.org/replica-id: '0'
            torchx.pytorch.org/role-index: '0'
            torchx.pytorch.org/role-name: trainer_foo
            torchx.pytorch.org/version: {torchx.__version__}
          name: app-name-0
          namespace: test_namespace
        spec:
          containers:
          - command:
            - main
            - --output-path
            - ''
            - --app-id
            - app-name
            - --rank0-env
            - TORCHX_RANK0_HOST
            env:
            - name: FOO
              value: bar
            - name: TORCHX_RANK0_HOST
              value: localhost
            - name: TORCHX_MCAD_TRAINERFOO_0_HOSTS
              value: app-name-0.app-name
            image: pytorch/torchx:latest
            name: app-name-0
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
          hostname: app-name-0
          imagePullSecrets:
          - {{}}
          nodeSelector: {{}}
          restartPolicy: Never
          schedulerName: test_coscheduler
          subdomain: app-name
          volumes:
          - emptyDir:
              medium: Memory
            name: dshm
          - hostPath:
              path: /src
            name: mount-0
      replicas: 1
    - generictemplate:
        apiVersion: v1
        kind: Service
        metadata:
          labels:
            app.kubernetes.io/instance: app-name
            app.kubernetes.io/managed-by: torchx.pytorch.org
            app.kubernetes.io/name: test
          name: app-name
          namespace: test_namespace
        spec:
          clusterIP: None
          ports:
          - port: 1234
            protocol: TCP
            targetPort: 1234
          publishNotReadyAddresses: true
          selector:
            app.kubernetes.io/instance: app-name
          sessionAffinity: None
          type: ClusterIP
        status:
          loadBalancer: {{}}
      replicas: 1
  schedulingSpec:
    minAvailable: 1
    requeuing:
      growthType: exponential
      maxNumRequeuings: 3
      maxTimeInSeconds: 0
      timeInSeconds: 300
""",
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        get_namespaced_custom_object.return_value = _test_mcad_generic_item()

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="echoImage",
                min_replicas=None,
                base_image=None,
                args=[
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                env={"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=900, gpu=2, memMB=512976, capabilities={}, devices={}
                ),
                port_map={"c10d": 29500},
                metadata={},
                mounts=[specs.VolumeMount(src="dshm", dst_path="/dev/shm")],
            )
        }

        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_generic_items(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0] = {}
        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {}

        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_metadata(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"] = {
            "spec": {
                "containers": [
                    {
                        "command": [
                            "bash",
                            "-c",
                            "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                        ],
                        "env": {
                            "name": "TORCH_DISTRIBUTED_DEBUG",
                            "value": "DETAIL",
                        },
                        "image": "echoImage",
                        "resources": {
                            "limits": {
                                "cpu": "1000m",
                                "memory": "514000M",
                                "gpu": "2",
                            },
                            "requests": {
                                "cpu": 900,
                                "memory": 512976,
                                "gpu": 2,
                            },
                        },
                        "ports": {"c10d": 29500},
                        "volumeMounts": [
                            specs.VolumeMount(src="dshm", dst_path="/dev/shm")
                        ],
                    }
                ],
            },
        }

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {}

        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_label(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "metadata"
        ] = {}

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)
        expect = {}
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_role_name(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "metadata"
        ]["labels"] = {}

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)
        expect = {}
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_specs(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0] = {
            "generictemplate": {
                "metadata": {
                    "labels": {
                        "torchx.pytorch.org/role-name": "echo",
                    },
                },
            }
        }

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="",
                min_replicas=None,
                base_image=None,
                args=[],
                env={},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=-1, gpu=-1, memMB=-1, capabilities={}, devices={}
                ),
                port_map={},
                metadata={},
                mounts=[],
            )
        }

        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_image_name(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "spec"
        ]["containers"] = [
            {
                "command": [
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                "env": {
                    "name": "TORCH_DISTRIBUTED_DEBUG",
                    "value": "DETAIL",
                },
                "resources": {
                    "limits": {
                        "cpu": "1000m",
                        "memory": "514000M",
                        "gpu": "2",
                    },
                    "requests": {
                        "cpu": 900,
                        "memory": 512976,
                        "gpu": 2,
                    },
                },
                "ports": {"c10d": 29500},
                "volumeMounts": [specs.VolumeMount(src="dshm", dst_path="/dev/shm")],
            }
        ]

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="",
                min_replicas=None,
                base_image=None,
                args=[],
                env={},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=-1, gpu=-1, memMB=-1, capabilities={}, devices={}
                ),
                port_map={},
                metadata={},
                mounts=[],
            )
        }
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_resources(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "spec"
        ]["containers"] = [
            {
                "command": [
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                "env": {
                    "name": "TORCH_DISTRIBUTED_DEBUG",
                    "value": "DETAIL",
                },
                "image": "echoImage",
                "ports": {"c10d": 29500},
                "volumeMounts": [specs.VolumeMount(src="dshm", dst_path="/dev/shm")],
            }
        ]

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="echoImage",
                min_replicas=None,
                base_image=None,
                args=[
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                env={"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=-1, gpu=-1, memMB=-1, capabilities={}, devices={}
                ),
                port_map={},
                metadata={},
                mounts=[],
            )
        }
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_cpu(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "spec"
        ]["containers"] = [
            {
                "command": [
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                "env": {
                    "name": "TORCH_DISTRIBUTED_DEBUG",
                    "value": "DETAIL",
                },
                "image": "echoImage",
                "resources": {
                    "limits": {
                        "cpu": "1000m",
                        "memory": "514000M",
                        "gpu": "2",
                    },
                    "requests": {
                        "memory": 512976,
                        "gpu": 2,
                    },
                },
                "ports": {"c10d": 29500},
                "volumeMounts": [specs.VolumeMount(src="dshm", dst_path="/dev/shm")],
            }
        ]

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="echoImage",
                min_replicas=None,
                base_image=None,
                args=[
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                env={"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=-1, gpu=-1, memMB=-1, capabilities={}, devices={}
                ),
                port_map={},
                metadata={},
                mounts=[],
            )
        }
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_memory(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "spec"
        ]["containers"] = [
            {
                "command": [
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                "env": {
                    "name": "TORCH_DISTRIBUTED_DEBUG",
                    "value": "DETAIL",
                },
                "image": "echoImage",
                "resources": {
                    "limits": {
                        "cpu": "1000m",
                        "memory": "514000M",
                        "gpu": "2",
                    },
                    "requests": {
                        "cpu": 900,
                        "gpu": 2,
                    },
                },
                "ports": {"c10d": 29500},
                "volumeMounts": [specs.VolumeMount(src="dshm", dst_path="/dev/shm")],
            }
        ]

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="echoImage",
                min_replicas=None,
                base_image=None,
                args=[
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                env={"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=-1, gpu=-1, memMB=-1, capabilities={}, devices={}
                ),
                port_map={},
                metadata={},
                mounts=[],
            )
        }
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_ports(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "spec"
        ]["containers"] = [
            {
                "command": [
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                "env": {
                    "name": "TORCH_DISTRIBUTED_DEBUG",
                    "value": "DETAIL",
                },
                "image": "echoImage",
                "resources": {
                    "limits": {
                        "cpu": "1000m",
                        "memory": "514000M",
                        "gpu": "2",
                    },
                    "requests": {
                        "cpu": 900,
                        "memory": 512976,
                        "gpu": 2,
                    },
                },
                "volumeMounts": [specs.VolumeMount(src="dshm", dst_path="/dev/shm")],
            }
        ]

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="echoImage",
                min_replicas=None,
                base_image=None,
                args=[
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                env={"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=900, gpu=2, memMB=512976, capabilities={}, devices={}
                ),
                port_map={},
                metadata={},
                mounts=[],
            )
        }
        self.assertEqual(roles, expect)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_get_role_information_no_volume_mounts(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        test_generic_item = _test_mcad_generic_item()
        test_generic_item["spec"]["resources"]["GenericItems"][0]["generictemplate"][
            "spec"
        ]["containers"] = [
            {
                "command": [
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                "env": {
                    "name": "TORCH_DISTRIBUTED_DEBUG",
                    "value": "DETAIL",
                },
                "image": "echoImage",
                "resources": {
                    "limits": {
                        "cpu": "1000m",
                        "memory": "514000M",
                        "gpu": "2",
                    },
                    "requests": {
                        "cpu": 900,
                        "memory": 512976,
                        "gpu": 2,
                    },
                },
                "ports": {"c10d": 29500},
            }
        ]

        get_namespaced_custom_object.return_value = test_generic_item

        spec = get_namespaced_custom_object.return_value["spec"]
        resources = spec["resources"]
        genericItems = resources["GenericItems"]

        roles = get_role_information(genericItems)

        expect = {
            "echo": Role(
                name="echo",
                image="echoImage",
                min_replicas=None,
                base_image=None,
                args=[
                    "bash",
                    "-c",
                    "python -m torch.distributed.run --rdzv_backend c10d --rdzv_endpoint $TORCHX_MCAD_ECHO_0_HOSTS:29500 --rdzv_id 'echo-nr36fcswzdb63c' --nnodes 1 --nproc_per_node 2 --tee 3 --role '' echo.py",
                ],
                env={"name": "TORCH_DISTRIBUTED_DEBUG", "value": "DETAIL"},
                num_replicas=1,
                max_retries=0,
                resource=Resource(
                    cpu=900, gpu=2, memMB=512976, capabilities={}, devices={}
                ),
                port_map={"c10d": 29500},
                metadata={},
                mounts=[],
            )
        }
        self.assertEqual(roles, expect)

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
        pod = role_to_pod(
            "foo",
            "foo-unique",
            "testnamespace",
            role,
            service_account="",
            image_secret="",
            coscheduler_name="",
            priority_class_name="",
            network="",
        )
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
        pod = role_to_pod(
            "foo",
            "foo-unique",
            "testnamespace",
            role,
            service_account="",
            image_secret="",
            coscheduler_name="",
            priority_class_name="",
            network="",
        )
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

    def test_resource_devices(self) -> None:
        scheduler = create_scheduler("test")

        role = specs.Role(
            name="foo",
            image="",
            resource=specs.Resource(
                cpu=2,
                memMB=3000,
                gpu=4,
                devices={
                    "vpc.amazonaws.com/efa": 4,
                },
            ),
        )
        pod = role_to_pod(
            "foo",
            "foo-unique",
            "testnamespace",
            role,
            service_account="",
            image_secret="",
            coscheduler_name="",
            priority_class_name="",
            network="",
        )
        self.assertEqual(
            pod.spec.containers[0].resources.limits,
            {
                "cpu": "2000m",
                "memory": "3000M",
                "nvidia.com/gpu": "4",
                "vpc.amazonaws.com/efa": "4",
            },
        )
        self.assertFalse(pod.spec.containers[0].security_context.privileged)

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
        pod = role_to_pod(
            "foo",
            "foo-unique",
            "testnamespace",
            role,
            service_account="",
            image_secret="",
            coscheduler_name="",
            priority_class_name="",
            network="",
        )
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
        cfg = KubernetesMCADOpts({"namespace": "test_namespace"})
        with patch(
            "torchx.schedulers.kubernetes_mcad_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name"
            info = scheduler._submit_dryrun(app, cfg)

        # pyre-fixme
        tasks = info.request.resource["spec"]["resources"]["GenericItems"]
        container0 = tasks[0]["generictemplate"].spec.containers[0]
        self.assertIn("TORCHX_RANK0_HOST", container0.command)
        self.assertIn(
            V1EnvVar(name="TORCHX_RANK0_HOST", value="localhost"), container0.env
        )
        container1 = tasks[1]["generictemplate"].spec.containers[0]
        self.assertIn("TORCHX_MCAD_TRAINERFOO_0_HOSTS", container1.command)

    def test_submit_dryrun_patch(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].image = "sha256:testhash"
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
                "image_repo": "example.com/some/repo",
            }
        )
        with patch(
            "torchx.schedulers.kubernetes_mcad_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler._submit_dryrun(app, cfg)

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

    def test_submit_dryrun_service_account(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("service_account", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
                "service_account": "srvacc",
            }
        )
        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn("'service_account_name': 'srvacc'", str(info.request.resource))

        del cfg["service_account"]
        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn("service_account_name': None", str(info.request.resource))

    def test_submit_dryrun_secret_name(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("image_secret", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
                "image_secret": "secret_name",
            }
        )
        info = scheduler._submit_dryrun(app, cfg)
        want = "image_pull_secrets': [{'name': 'secret_name'}]"
        self.assertIn(want, str(info.request.resource))

        del cfg["image_secret"]
        info = scheduler._submit_dryrun(app, cfg)
        want = "image_pull_secrets': [{'name': None}]"
        self.assertIn(want, str(info.request.resource))

    def test_submit_dryrun_priority(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("priority", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
                "priority": 10,
            }
        )

        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn("'priority': 10", str(info.request.resource))

        del cfg["priority"]
        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn("'priority': None", str(info.request.resource))

    def test_submit_dryrun_priority_class_name(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("priority_class_name", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
                "priority_class_name": "test-priority",
            }
        )

        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn(
            "'priority_class_name': 'test-priority'", str(info.request.resource)
        )

        del cfg["priority_class_name"]
        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn("'priority_class_name': None", str(info.request.resource))

    def test_submit_dryrun_network(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("network", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
                "network": "test-network-conf",
            }
        )

        info = scheduler._submit_dryrun(app, cfg)
        self.assertIn(
            "'k8s.v1.cni.cncf.io/networks': 'test-network-conf'",
            str(info.request.resource),
        )

        del cfg["network"]
        info = scheduler._submit_dryrun(app, cfg)
        self.assertNotIn("'k8s.v1.cni.cncf.io/networks'", str(info.request.resource))

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit(self, create_namespaced_custom_object: MagicMock) -> None:
        create_namespaced_custom_object.return_value = {
            "metadata": {"name": "testid"},
        }
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
            }
        )

        info = scheduler._submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "testnamespace:testid")
        call = create_namespaced_custom_object.call_args
        args, kwargs = call
        self.assertEqual(kwargs["group"], "workload.codeflare.dev")
        self.assertEqual(kwargs["version"], "v1beta1")
        self.assertEqual(kwargs["namespace"], "testnamespace")
        self.assertEqual(kwargs["plural"], "appwrappers")
        self.assertEqual(kwargs["body"], info.request.resource)

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit_job_name_conflict(
        self, create_namespaced_custom_object: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=409, reason="Conflict")
        api_exc.body = '{"details":{"name": "test_job"}}'
        create_namespaced_custom_object.side_effect = api_exc

        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
            }
        )
        info = scheduler._submit_dryrun(app, cfg)
        with self.assertRaises(ValueError):
            scheduler.schedule(info)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_describe(self, get_namespaced_custom_object: MagicMock) -> None:
        get_namespaced_custom_object.return_value = {
            "status": {
                "state": "Running",
                "Succeeded": 1,
            },
            "spec": {
                "resources": {
                    "GenericItems": [
                        {
                            "generictemplate": {
                                "metadata": {
                                    "labels": {
                                        "torchx.pytorch.org/role-name": "echo",
                                    },
                                },
                                "spec": {},
                            },
                        }
                    ],
                },
            },
        }

        app_id = "foo:bar"
        scheduler = create_scheduler("foo")
        info = scheduler.describe(app_id)
        call = get_namespaced_custom_object.call_args
        args, kwargs = call

        assert "workload.codeflare.dev" in args
        assert "v1beta1" in args
        assert "appwrappers" in args
        assert "foo" in args
        assert "bar" in args

        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.RUNNING,
                roles_statuses=[
                    specs.RoleStatus(
                        "echo",
                        [
                            specs.ReplicaStatus(
                                id=0,
                                role="echo",
                                state=specs.ReplicaState.SUCCEEDED,
                                hostname="",
                            )
                        ],
                    ),
                ],
                roles=[
                    specs.Role(name="echo", image="", num_replicas=1),
                ],
            ),
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_describe_pending_dispatch(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        get_namespaced_custom_object.return_value = {
            "status": {
                "state": "Pending",
            },
            "spec": {
                "resources": {
                    "GenericItems": [
                        {
                            "generictemplate": {
                                "metadata": {
                                    "labels": {
                                        "torchx.pytorch.org/role-name": "echo",
                                    },
                                },
                                "spec": {},
                            },
                        },
                        {
                            "generictemplate": {
                                "kind": "Service",
                                "metadata": {
                                    "name": "echo-service",
                                },
                                "spec": {},
                            },
                        },
                    ],
                },
            },
        }

        app_id = "foo:bar"
        scheduler = create_scheduler("foo")
        info = scheduler.describe(app_id)

        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.PENDING,
                roles_statuses=[
                    specs.RoleStatus(
                        "echo",
                        [
                            specs.ReplicaStatus(
                                id=0,
                                role="echo",
                                state=specs.ReplicaState.PENDING,
                                hostname="",
                            )
                        ],
                    ),
                ],
                roles=[
                    specs.Role(name="echo", image="", num_replicas=1),
                ],
            ),
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_describe_unknown(self, get_namespaced_custom_object: MagicMock) -> None:
        get_namespaced_custom_object.return_value = {}
        app_id = "foo:bar"
        scheduler = create_scheduler("foo")
        info = scheduler.describe(app_id)
        call = get_namespaced_custom_object.call_args
        args, kwargs = call

        assert "workload.codeflare.dev" in args
        assert "v1beta1" in args
        assert "appwrappers" in args
        assert "foo" in args
        assert "bar" in args

        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.UNKNOWN,
            ),
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_describe_failure(self, get_namespaced_custom_object: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=404, reason="Not Found")
        get_namespaced_custom_object.side_effect = api_exc
        app_id = "foo:bar"
        scheduler = create_scheduler("foo")
        with self.assertRaises(ValueError):
            scheduler.describe(app_id)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_describe_unauthorized(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=401, reason="Unauthorized")
        get_namespaced_custom_object.side_effect = api_exc
        app_id = "foo:bar"
        scheduler = create_scheduler("foo")
        with self.assertRaises(ValueError):
            scheduler.describe(app_id)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object")
    def test_describe_unknown_failure(
        self, get_namespaced_custom_object: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=400, reason="Bad Request")
        get_namespaced_custom_object.side_effect = api_exc
        app_id = "foo:bar"
        scheduler = create_scheduler("foo")
        with self.assertRaises(ApiException):
            scheduler.describe(app_id)

    def test_runopts(self) -> None:
        scheduler = kubernetes_mcad_scheduler.create_scheduler("foo")
        runopts = scheduler.run_opts()
        self.assertEqual(
            set(runopts._opts.keys()),
            {
                "namespace",
                "image_repo",
                "service_account",
                "priority",
                "priority_class_name",
                "image_secret",
                "coscheduler_name",
                "network",
            },
        )

    @patch("kubernetes.client.CustomObjectsApi.delete_namespaced_custom_object")
    def test_cancel_existing(self, delete_namespaced_custom_object: MagicMock) -> None:
        scheduler = create_scheduler("test")
        scheduler._cancel_existing("testnamespace:testjob")
        call = delete_namespaced_custom_object.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "workload.codeflare.dev",
                "version": "v1beta1",
                "namespace": "testnamespace",
                "plural": "appwrappers",
                "name": "testjob",
            },
        )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list(self, list_namespaced_custom_object: MagicMock) -> None:
        with patch(
            "torchx.schedulers.kubernetes_mcad_scheduler.KubernetesMCADScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]
            scheduler = create_scheduler("test")
            scheduler.list()
            call = list_namespaced_custom_object.call_args
            args, kwargs = call

            self.assertEqual(
                kwargs,
                {
                    "group": "workload.codeflare.dev",
                    "version": "v1beta1",
                    "namespace": "default",
                    "plural": "appwrappers",
                    "timeout_seconds": 30,
                },
            )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list_values(self, list_namespaced_custom_object: MagicMock) -> None:
        list_namespaced_custom_object.return_value = {
            "apiVersion": "workload.codeflare.dev/v1beta1",
            "name": "test-training",
            "namespace": "default",
            "items": [
                {
                    "apiVersion": "workload.codeflare.dev/v1beta1",
                    "kind": "AppWrapper",
                    "metadata": {
                        "name": "test-training",
                        "namespace": "default",
                    },
                    "status": {
                        "canrun": "true",
                        "conditions": [
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:23:55.036212Z",
                                "lastUpdateMicroTime": "2023-01-10T16:23:55.036211Z",
                                "status": "True",
                                "type": "Init",
                            },
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:23:55.036419Z",
                                "lastUpdateMicroTime": "2023-01-10T16:23:55.036419Z",
                                "reason": "AwaitingHeadOfLine",
                                "status": "True",
                                "type": "Queueing",
                            },
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:23:55.050841Z",
                                "lastUpdateMicroTime": "2023-01-10T16:23:55.050840Z",
                                "reason": "FrontOfQueue.",
                                "status": "True",
                                "type": "HeadOfLine",
                            },
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:24:06.762455Z",
                                "lastUpdateMicroTime": "2023-01-10T16:24:06.762455Z",
                                "reason": "AppWrapperRunnable",
                                "status": "True",
                                "type": "Dispatched",
                            },
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:24:06.780635Z",
                                "lastUpdateMicroTime": "2023-01-10T16:24:06.780635Z",
                                "reason": "PodsRunning",
                                "status": "True",
                                "type": "Running",
                            },
                        ],
                        "controllerfirsttimestamp": "2023-01-10T16:23:55.035192Z",
                        "filterignore": "true",
                        "queuejobstate": "Running",
                        "running": "2",
                        "sender": "before [syncQueueJob] setRunning",
                        "state": "Running",
                    },
                },
                {
                    "apiVersion": "workload.codeflare.dev/v1beta1",
                    "kind": "AppWrapper",
                    "metadata": {
                        "name": "test-training",
                        "namespace": "default",
                    },
                    "status": {
                        "canrun": "true",
                        "conditions": [
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:23:55.036212Z",
                                "lastUpdateMicroTime": "2023-01-10T16:23:55.036211Z",
                                "status": "True",
                                "type": "Init",
                            },
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:23:55.036419Z",
                                "lastUpdateMicroTime": "2023-01-10T16:23:55.036419Z",
                                "reason": "AwaitingHeadOfLine",
                                "status": "True",
                                "type": "Queueing",
                            },
                            {
                                "lastTransitionMicroTime": "2023-01-10T16:23:55.050841Z",
                                "lastUpdateMicroTime": "2023-01-10T16:23:55.050840Z",
                                "reason": "FrontOfQueue.",
                                "status": "True",
                                "type": "HeadOfLine",
                            },
                        ],
                        "controllerfirsttimestamp": "2023-01-10T16:23:55.035192Z",
                        "filterignore": "true",
                        "queuejobstate": "HeadOfLine",
                        "sender": "before ScheduleNext - setHOL",
                        "state": "Pending",
                    },
                },
            ],
        }

        with patch(
            "torchx.schedulers.kubernetes_mcad_scheduler.KubernetesMCADScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]
            scheduler = create_scheduler("test")

            apps = scheduler.list()
            call = list_namespaced_custom_object.call_args
            args, kwargs = call

            self.assertEqual(
                apps,
                [
                    ListAppResponse(
                        app_id="default:test-training", state=AppState.RUNNING
                    ),
                    ListAppResponse(
                        app_id="default:test-training", state=AppState.PENDING
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
            "torchx.schedulers.kubernetes_mcad_scheduler.KubernetesMCADScheduler._get_active_context"
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
                "name": "testjob-1",
                "timestamps": True,
            },
        )

    def test_push_patches(self) -> None:
        client = MagicMock()
        scheduler = KubernetesMCADScheduler(
            "foo",
            client=MagicMock(),
            docker_client=client,
        )

        job = KubernetesMCADJob(
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


class KubernetesMCADSchedulerNoImportTest(unittest.TestCase):
    """
    KubernetesMCADSchedulerNoImportTest tests the kubernetes scheduler behavior when
    Kubernetes is not available.
    """

    def setUp(self) -> None:
        # make all kubernetes modules unable to be imported
        for mod in list(sys.modules.keys()) + ["kubernetes"]:
            if mod.startswith("kubernetes"):
                sys.modules[mod] = None  # pyre-ignore

        # reload to ensure kubernetes_mcad_scheduler doesn't depend on them at import
        # time
        importlib.reload(kubernetes_mcad_scheduler)
        importlib.reload(schedulers)

    def tearDown(self) -> None:
        # reset all kubernetes modules we patched
        for mod in list(sys.modules.keys()):
            if mod.startswith("kubernetes"):
                del sys.modules[mod]
        # reimport kubernetes_mcad_scheduler to get to a clean state
        importlib.reload(kubernetes_mcad_scheduler)

    def test_runopts(self) -> None:
        scheduler = kubernetes_mcad_scheduler.create_scheduler("foo")
        self.assertIsNotNone(scheduler.run_opts())

    def test_describe(self) -> None:
        scheduler = kubernetes_mcad_scheduler.create_scheduler("foo")
        with self.assertRaises(ModuleNotFoundError):
            scheduler.describe("foo:bar")

    def test_dryrun(self) -> None:
        scheduler = kubernetes_mcad_scheduler.create_scheduler("foo")
        app = _test_app()
        cfg = KubernetesMCADOpts(
            {
                "namespace": "testnamespace",
            }
        )

        with self.assertRaises(ModuleNotFoundError):
            scheduler._submit_dryrun(app, cfg)
