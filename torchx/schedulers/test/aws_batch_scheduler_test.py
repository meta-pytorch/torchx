# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import threading
import unittest
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, Optional
from unittest.mock import MagicMock, patch

import torchx
from torchx import specs
from torchx.schedulers.api import ListAppResponse
from torchx.schedulers.aws_batch_scheduler import (
    _local_session,
    _parse_num_replicas,
    _role_to_node_properties,
    AWSBatchOpts,
    AWSBatchScheduler,
    create_scheduler,
    ENV_TORCHX_ROLE_NAME,
    parse_ulimits,
    resource_from_resource_requirements,
    resource_requirements_from_resource,
    to_millis_since_epoch,
)
from torchx.specs import AppState, Resource


def _test_app(
    num_replicas: int = 2, resource: Optional[Resource] = None
) -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer",
        image="pytorch/torchx:latest",
        entrypoint="bash",
        args=[
            "-c",
            f"--output-path {specs.macros.img_root}"
            f" --app-id {specs.macros.app_id}"
            f" --replica_id {specs.macros.replica_id}"
            f" --rank0_host $${{{specs.macros.rank0_env}:=localhost}}",
        ],
        env={"FOO": "bar"},
        resource=resource
        or specs.Resource(
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

    return specs.AppDef("test", roles=[trainer_role], metadata={"FIZZ": "buzz"})


@contextmanager
def mock_rand() -> Generator[None, None, None]:
    with patch("torchx.schedulers.aws_batch_scheduler.make_unique") as make_unique_ctx:
        make_unique_ctx.return_value = "app-name-42"
        yield


boto3Response = Dict[str, Any]  # boto3 responses are JSON


class MockPaginator:
    """
    Used for mocking ``boto3.client("<SERVICE>").get_paginator("<API>")`` calls.
    """

    def __init__(self, **op_to_pages: Iterable[boto3Response]) -> None:
        # boto3 paginators return an iterable of API responses
        self.op_to_pages: Dict[str, Iterable[boto3Response]] = op_to_pages
        self.op_name: Optional[str] = None

    def __call__(self, op_name: str) -> "MockPaginator":
        self.op_name = op_name
        return self

    def paginate(self, *_1: Any, **_2: Any) -> Iterable[Dict[str, Any]]:
        if self.op_name:
            return self.op_to_pages[self.op_name]
        else:
            raise RuntimeError(
                "`op_name` not set. Did you forget to call `__call__(op_name)`?"
            )


# paginators return iterables of the API responses


class AWSBatchSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        client = MagicMock()
        log_client = MagicMock()
        docker_client = MagicMock()
        scheduler = create_scheduler(
            "foo", client=client, log_client=log_client, docker_client=docker_client
        )
        self.assertIsInstance(scheduler, AWSBatchScheduler)
        self.assertEqual(scheduler._client, client)
        self.assertEqual(scheduler._log_client, log_client)
        self.assertEqual(scheduler._docker_client, docker_client)

    def test_submit_dryrun_with_share_id(self) -> None:
        app = _test_app()
        cfg = AWSBatchOpts({"queue": "testqueue", "share_id": "fooshare"})
        info = create_scheduler("test").submit_dryrun(app, cfg)

        req = info.request
        job_def = req.job_def
        self.assertEqual(req.share_id, "fooshare")
        # must be set for jobs submitted to a queue with scheduling policy
        self.assertEqual(job_def["schedulingPriority"], 0)

    def test_submit_dryrun_with_priority_but_not_share_id(self) -> None:
        cfg = AWSBatchOpts({"queue": "testqueue", "priority": 42})
        dryrun_info = create_scheduler("test").submit_dryrun(_test_app(), cfg)
        self.assertFalse("schedulingPriority" in dryrun_info.request.job_def)
        self.assertIsNone(dryrun_info.request.share_id)

    def test_submit_dryrun_with_priority(self) -> None:
        cfg = AWSBatchOpts({"queue": "testqueue", "share_id": "foo", "priority": 42})
        info = create_scheduler("test").submit_dryrun(_test_app(), cfg)

        req = info.request
        job_def = req.job_def
        self.assertEqual(req.share_id, "foo")
        self.assertEqual(job_def["schedulingPriority"], 42)

    @patch(
        "torchx.schedulers.aws_batch_scheduler.getpass.getuser", return_value="testuser"
    )
    def test_submit_dryrun_tags(self, _) -> None:
        # intentionally not specifying user in cfg to test default
        cfg = AWSBatchOpts({"queue": "ignored_in_test"})
        info = create_scheduler("test").submit_dryrun(_test_app(), cfg)
        self.assertEqual(
            {
                "torchx.pytorch.org/version": torchx.__version__,
                "torchx.pytorch.org/app-name": "test",
                "torchx.pytorch.org/user": "testuser",
                "FIZZ": "buzz",
            },
            info.request.job_def["tags"],
        )

    def test_submit_dryrun_job_role_arn(self) -> None:
        cfg = AWSBatchOpts({"queue": "ignored_in_test", "job_role_arn": "fizzbuzz"})
        info = create_scheduler("test").submit_dryrun(_test_app(), cfg)
        # pyre-ignore[16]
        node_groups = info.request.job_def["nodeProperties"]["nodeRangeProperties"]
        self.assertEqual(1, len(node_groups))
        self.assertEqual(cfg["job_role_arn"], node_groups[0]["container"]["jobRoleArn"])

    def test_submit_dryrun_execution_role_arn(self) -> None:
        cfg = AWSBatchOpts(
            {"queue": "ignored_in_test", "execution_role_arn": "veryexecutive"}
        )
        info = create_scheduler("test").submit_dryrun(_test_app(), cfg)
        # pyre-ignore[16]
        node_groups = info.request.job_def["nodeProperties"]["nodeRangeProperties"]
        self.assertEqual(1, len(node_groups))
        self.assertEqual(
            cfg["execution_role_arn"], node_groups[0]["container"]["executionRoleArn"]
        )

    def test_submit_dryrun_privileged(self) -> None:
        cfg = AWSBatchOpts({"queue": "ignored_in_test", "privileged": True})
        info = create_scheduler("test").submit_dryrun(_test_app(), cfg)
        # pyre-ignore[16]
        node_groups = info.request.job_def["nodeProperties"]["nodeRangeProperties"]
        self.assertEqual(1, len(node_groups))
        self.assertTrue(node_groups[0]["container"]["privileged"])

    def test_submit_dryrun_instance_type_multinode(self) -> None:
        cfg = AWSBatchOpts({"queue": "ignored_in_test", "privileged": True})
        resource = specs.named_resources_aws.aws_p3dn_24xlarge()
        app = _test_app(num_replicas=2, resource=resource)
        info = create_scheduler("test").submit_dryrun(app, cfg)
        # pyre-ignore[16]
        node_groups = info.request.job_def["nodeProperties"]["nodeRangeProperties"]
        self.assertEqual(1, len(node_groups))
        self.assertEqual(
            resource.capabilities[specs.named_resources_aws.K8S_ITYPE],
            node_groups[0]["container"]["instanceType"],
        )

    def test_submit_dryrun_instance_type_singlenode(self) -> None:
        cfg = AWSBatchOpts({"queue": "ignored_in_test", "privileged": True})
        resource = specs.named_resources_aws.aws_p3dn_24xlarge()
        app = _test_app(num_replicas=1, resource=resource)
        info = create_scheduler("test").submit_dryrun(app, cfg)
        # pyre-ignore[16]
        node_groups = info.request.job_def["nodeProperties"]["nodeRangeProperties"]
        self.assertEqual(1, len(node_groups))
        self.assertTrue("instanceType" in node_groups[0]["container"])

    def test_submit_dryrun_no_instance_type_non_aws(self) -> None:
        cfg = AWSBatchOpts({"queue": "ignored_in_test", "privileged": True})
        resource = specs.named_resources_aws.aws_p3dn_24xlarge()
        app = _test_app(num_replicas=2)
        info = create_scheduler("test").submit_dryrun(app, cfg)
        # pyre-ignore[16]
        node_groups = info.request.job_def["nodeProperties"]["nodeRangeProperties"]
        self.assertEqual(1, len(node_groups))
        self.assertTrue("instanceType" not in node_groups[0]["container"])

    @mock_rand()
    def test_submit_dryrun(self) -> None:
        cfg = AWSBatchOpts({"queue": "testqueue", "user": "testuser"})
        info = create_scheduler("test").submit_dryrun(_test_app(), cfg)

        req = info.request
        self.assertEqual(req.share_id, None)
        self.assertEqual(req.queue, "testqueue")
        job_def = req.job_def

        print(job_def)

        self.assertEqual(
            job_def,
            {
                "jobDefinitionName": "app-name-42",
                "type": "multinode",
                "nodeProperties": {
                    "numNodes": 2,
                    "mainNode": 0,
                    "nodeRangeProperties": [
                        {
                            "targetNodes": "0:1",
                            "container": {
                                "command": [
                                    "bash",
                                    "-c",
                                    "--output-path "
                                    " --app-id app-name-42"
                                    " --replica_id $AWS_BATCH_JOB_NODE_INDEX"
                                    " --rank0_host ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:=localhost}",
                                ],
                                "image": "pytorch/torchx:latest",
                                "environment": [
                                    {"name": "FOO", "value": "bar"},
                                    {"name": "TORCHX_ROLE_IDX", "value": "0"},
                                    {"name": "TORCHX_ROLE_NAME", "value": "trainer"},
                                ],
                                "privileged": False,
                                "resourceRequirements": [
                                    {"type": "VCPU", "value": "2"},
                                    {"type": "MEMORY", "value": "3000"},
                                    {"type": "GPU", "value": "4"},
                                ],
                                "linuxParameters": {
                                    "sharedMemorySize": 3000,
                                    "devices": [],
                                },
                                "logConfiguration": {"logDriver": "awslogs"},
                                "mountPoints": [
                                    {
                                        "containerPath": "/dst",
                                        "readOnly": True,
                                        "sourceVolume": "mount_0",
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "mount_0",
                                        "host": {
                                            "sourcePath": "/src",
                                        },
                                    }
                                ],
                            },
                        },
                    ],
                },
                "retryStrategy": {
                    "attempts": 3,
                    "evaluateOnExit": [{"onExitCode": "0", "action": "EXIT"}],
                },
                "tags": {
                    "torchx.pytorch.org/version": torchx.__version__,
                    "torchx.pytorch.org/app-name": "test",
                    "torchx.pytorch.org/user": "testuser",
                    "FIZZ": "buzz",
                },
            },
        )

    def test_volume_mounts(self) -> None:
        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.VolumeMount(src="efsid", dst_path="/dst", read_only=True),
            ],
            resource=specs.Resource(
                cpu=1,
                memMB=1000,
                gpu=0,
            ),
        )
        props = _role_to_node_properties(role, 0)
        self.assertEqual(
            props["container"]["volumes"],
            [
                {
                    "name": "mount_0",
                    "efsVolumeConfiguration": {
                        "fileSystemId": "efsid",
                    },
                }
            ],
        )
        self.assertEqual(
            props["container"]["mountPoints"],
            [
                {
                    "containerPath": "/dst",
                    "readOnly": True,
                    "sourceVolume": "mount_0",
                }
            ],
        )

    def test_device_mounts(self) -> None:
        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.DeviceMount(
                    src_path="/dev/foo", dst_path="/dev/bar", permissions="rwm"
                )
            ],
            resource=specs.Resource(
                cpu=1,
                memMB=1000,
                gpu=0,
            ),
        )
        props = _role_to_node_properties(role, 0)
        self.assertEqual(
            props["container"]["linuxParameters"]["devices"],
            [
                {
                    "hostPath": "/dev/foo",
                    "containerPath": "/dev/bar",
                    "permissions": ["READ", "WRITE", "MKNOD"],
                }
            ],
        )

    def test_resource_devices(self) -> None:
        role = specs.Role(
            name="foo",
            image="",
            mounts=[],
            resource=specs.Resource(
                cpu=1,
                memMB=1000,
                gpu=0,
                devices={"vpc.amazonaws.com/efa": 2, "aws.amazon.com/neurondevice": 1},
            ),
        )
        props = _role_to_node_properties(role, 0)
        self.assertEqual(
            props["container"]["linuxParameters"]["devices"],
            [
                {
                    "hostPath": "/dev/infiniband/uverbs0",
                    "containerPath": "/dev/infiniband/uverbs0",
                    "permissions": ["READ", "WRITE", "MKNOD"],
                },
                {
                    "hostPath": "/dev/infiniband/uverbs1",
                    "containerPath": "/dev/infiniband/uverbs1",
                    "permissions": ["READ", "WRITE", "MKNOD"],
                },
                {
                    "hostPath": "/dev/neuron0",
                    "containerPath": "/dev/neuron0",
                    "permissions": ["READ", "WRITE", "MKNOD"],
                },
            ],
        )

    def test_role_to_node_properties_ulimits(self) -> None:
        role = specs.Role(
            name="test",
            image="test:latest",
            entrypoint="test",
            args=["test"],
            resource=specs.Resource(cpu=1, memMB=1000, gpu=0),
        )
        ulimits = [
            {"name": "nofile", "softLimit": 65536, "hardLimit": 65536},
            {"name": "memlock", "softLimit": -1, "hardLimit": -1},
        ]
        props = _role_to_node_properties(role, 0, ulimits=ulimits)
        self.assertEqual(
            props["container"]["ulimits"],
            ulimits,
        )

    def test_parse_ulimits(self) -> None:
        # Test single ulimit
        result = parse_ulimits(["nofile:65536:65536"])
        expected = [{"name": "nofile", "softLimit": 65536, "hardLimit": 65536}]
        self.assertEqual(result, expected)

        # Test multiple ulimits
        result = parse_ulimits(["nofile:65536:65536", "memlock:-1:-1"])
        expected = [
            {"name": "nofile", "softLimit": 65536, "hardLimit": 65536},
            {"name": "memlock", "softLimit": -1, "hardLimit": -1},
        ]
        self.assertEqual(result, expected)

        # Test empty list
        result = parse_ulimits([])
        self.assertEqual(result, [])

        # Test invalid format
        with self.assertRaises(ValueError):
            parse_ulimits(["invalid"])

    def _mock_scheduler_running_job(self) -> AWSBatchScheduler:
        scheduler = AWSBatchScheduler(
            "test",
            client=MagicMock(),
            log_client=MagicMock(),
        )
        scheduler._client.get_paginator.side_effect = MockPaginator(
            list_jobs=[
                {
                    "jobSummaryList": [
                        {
                            "jobArn": "arn:aws:batch:us-east-1:761163492645:job/7b78f42f-fab7-4746-abb8-be761b858ddb",
                            "jobId": "7b78f42f-fab7-4746-abb8-be761b858ddb",
                            "jobName": "fairseq-train-wzt5p7p5j3tbqd",
                            "createdAt": 1656651215531,
                            "status": "RUNNING",
                            "nodeProperties": {"numNodes": 2},
                            "jobDefinition": "arn:aws:batch:us-east-1:761163492645:job-definition/fairseq-train-foo:1",
                        }
                    ]
                }
            ]
        )

        scheduler._client.describe_jobs.side_effect = [
            {
                "jobs": [
                    {
                        "jobArn": "arn:aws:batch:us-east-1:761163492645:job/7b78f42f-fab7-4746-abb8-be761b858ddb",
                        "jobName": "fairseq-train-wzt5p7p5j3tbqd",
                        "jobId": "7b78f42f-fab7-4746-abb8-be761b858ddb",
                        "jobQueue": "arn:aws:batch:us-east-1:761163492645:job-queue/torchx-proto-queue",
                        "status": "RUNNING",
                        "attempts": [],  # This is empty on running jobs (unlike completed jobs)
                        "createdAt": 1656651215531,
                        "retryStrategy": {
                            "attempts": 1,
                            "evaluateOnExit": [{"onExitCode": "0", "action": "exit"}],
                        },
                        "startedAt": 1656651662589,
                        "dependsOn": [],
                        "jobDefinition": "arn:aws:batch:us-east-1:761163492645:job-definition/fairseq-train-foo:1",
                        "parameters": {},
                        "nodeProperties": {
                            "numNodes": 2,
                            "mainNode": 0,
                            "nodeRangeProperties": [
                                {
                                    "targetNodes": "0:2",
                                    "container": {
                                        "image": "1234567890.dkr.ecr.us-west-2.amazonaws.com/foo/bar",
                                        "command": [
                                            "bash",
                                            "-c",
                                            "torchrun ...<omitted for test>...",
                                        ],
                                        "environment": [
                                            {
                                                "name": ENV_TORCHX_ROLE_NAME,
                                                "value": "echo",
                                            }
                                        ],
                                    },
                                }
                            ],
                        },
                        "tags": {
                            "torchx.pytorch.org/version": "0.3.0dev0",
                            "torchx.pytorch.org/app-name": "fairseq-train",
                        },
                        "platformCapabilities": [],
                    }
                ]
            },
            {
                "jobs": [
                    {
                        "jobArn": "arn:aws:batch:us-east-1:761163492645:job/7b78f42f-fab7-4746-abb8-be761b858ddb#1",
                        "jobName": "fairseq-train-wzt5p7p5j3tbqd",
                        "jobId": "7b78f42f-fab7-4746-abb8-be761b858ddb",
                        "jobQueue": "arn:aws:batch:us-east-1:761163492645:job-queue/torchx-proto-queue",
                        "status": "RUNNING",
                        "attempts": [],  # This is empty on running jobs (unlike completed jobs)
                        "createdAt": 1656651215531,
                        "retryStrategy": {
                            "attempts": 1,
                            "evaluateOnExit": [{"onExitCode": "0", "action": "exit"}],
                        },
                        "startedAt": 1656651662589,
                        "dependsOn": [],
                        "jobDefinition": "arn:aws:batch:us-east-1:761163492645:job-definition/fairseq-train-foo:1",
                        "parameters": {},
                        "container": {
                            "logStreamName": "running_log_stream",
                        },
                        "nodeProperties": {
                            "numNodes": 2,
                            "mainNode": 0,
                            "nodeRangeProperties": [],
                        },
                        "tags": {
                            "torchx.pytorch.org/version": "0.3.0dev0",
                            "torchx.pytorch.org/app-name": "fairseq-train",
                        },
                        "platformCapabilities": [],
                    }
                ]
            },
        ]

        scheduler._log_client.get_log_events.return_value = {
            "nextForwardToken": "some_token",
            "events": [
                {
                    "message": "foo",
                    "timestamp": to_millis_since_epoch(
                        datetime(2023, 3, 14, 16, 00, 1)
                    ),
                },
                {
                    "message": "foobar",
                    "timestamp": to_millis_since_epoch(
                        datetime(2023, 3, 14, 16, 00, 2)
                    ),
                },
                {
                    "message": "bar",
                    "timestamp": to_millis_since_epoch(
                        datetime(2023, 3, 14, 16, 00, 3)
                    ),
                },
            ],
        }

        return scheduler

    def _mock_scheduler(self) -> AWSBatchScheduler:
        scheduler = AWSBatchScheduler(
            "test",
            client=MagicMock(),
            log_client=MagicMock(),
        )

        scheduler._client.get_paginator.side_effect = MockPaginator(
            describe_job_queues=[
                {
                    "ResponseMetadata": {},
                    "jobQueues": [
                        {
                            "jobQueueName": "torchx",
                            "jobQueueArn": "arn:aws:batch:test-region:4000005:job-queue/torchx",
                            "state": "ENABLED",
                        },
                    ],
                }
            ],
            list_jobs=[
                {
                    "jobSummaryList": [
                        {
                            "jobArn": "arn:aws:batch:us-west-2:495572122715:job/6afc27d7-3559-43ca-89fd-1007b6bf2546",
                            "jobId": "6afc27d7-3559-43ca-89fd-1007b6bf2546",
                            "jobName": "app-name-42",
                            "createdAt": 1643949940162,
                            "status": "SUCCEEDED",
                            "stoppedAt": 1643950324125,
                            "container": {"exitCode": 0},
                            "nodeProperties": {"numNodes": 2},
                            "jobDefinition": "arn:aws:batch:us-west-2:495572122715:job-definition/app-name-42:1",
                        }
                    ]
                }
            ],
        )

        scheduler._client.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobArn": "arn:aws:batch:us-west-2:495572122715:job/6afc27d7-3559-43ca-89fd-1007b6bf2546",
                    "jobName": "app-name-42",
                    "jobId": "6afc27d7-3559-43ca-89fd-1007b6bf2546",
                    "jobQueue": "testqueue",
                    "status": "SUCCEEDED",
                    "attempts": [
                        {
                            "container": {
                                "exitCode": 0,
                                "logStreamName": "log_stream",
                                "networkInterfaces": [],
                            },
                            "startedAt": 1643950310819,
                            "stoppedAt": 1643950324125,
                            "statusReason": "Essential container in task exited",
                        }
                    ],
                    "statusReason": "Essential container in task exited",
                    "createdAt": 1643949940162,
                    "retryStrategy": {
                        "attempts": 1,
                        "evaluateOnExit": [{"onExitCode": "0", "action": "exit"}],
                    },
                    "startedAt": 1643950310819,
                    "stoppedAt": 1643950324125,
                    "dependsOn": [],
                    "jobDefinition": "job-def",
                    "parameters": {},
                    "nodeProperties": {
                        "numNodes": 2,
                        "mainNode": 0,
                        "nodeRangeProperties": [
                            {
                                "targetNodes": "0:1",
                                "container": {
                                    "image": "ghcr.io/pytorch/torchx:0.1.2dev0",
                                    "command": ["echo", "your name"],
                                    "volumes": [],
                                    "environment": [
                                        {"name": "TORCHX_ROLE_IDX", "value": "0"},
                                        {"name": "TORCHX_ROLE_NAME", "value": "echo"},
                                        {
                                            "name": "TORCHX_RANK0_HOST",
                                            "value": "localhost",
                                        },
                                    ],
                                    "mountPoints": [],
                                    "ulimits": [],
                                    "resourceRequirements": [
                                        {"value": "1", "type": "VCPU"},
                                        {"value": "1000", "type": "MEMORY"},
                                    ],
                                    "logConfiguration": {
                                        "logDriver": "awslogs",
                                        "options": {},
                                        "secretOptions": [],
                                    },
                                    "secrets": [],
                                },
                            },
                        ],
                    },
                    "tags": {
                        "torchx.pytorch.org/version": "0.1.2dev0",
                        "torchx.pytorch.org/app-name": "echo",
                    },
                    "platformCapabilities": [],
                }
            ]
        }

        scheduler._log_client.get_log_events.return_value = {
            "nextForwardToken": "some_token",
            "events": [
                {
                    "message": "foo",
                    "timestamp": to_millis_since_epoch(
                        datetime(2023, 3, 14, 16, 00, 1)
                    ),
                },
                {
                    "message": "foobar",
                    "timestamp": to_millis_since_epoch(
                        datetime(2023, 3, 14, 16, 00, 2)
                    ),
                },
                {
                    "message": "bar",
                    "timestamp": to_millis_since_epoch(
                        datetime(2023, 3, 14, 16, 00, 3)
                    ),
                },
            ],
        }

        return scheduler

    @mock_rand()
    def test_submit(self) -> None:
        scheduler = self._mock_scheduler()
        app = _test_app()
        cfg = AWSBatchOpts({"queue": "testqueue"})

        info = scheduler.submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "testqueue:app-name-42")
        self.assertEqual(scheduler._client.register_job_definition.call_count, 1)
        self.assertEqual(scheduler._client.submit_job.call_count, 1)

    def test_describe(self) -> None:
        scheduler = self._mock_scheduler()
        status = scheduler.describe("testqueue:app-name-42")
        self.assertIsNotNone(status)
        self.assertEqual(status.state, specs.AppState.SUCCEEDED)
        self.assertEqual(status.app_id, "testqueue:app-name-42")
        self.assertEqual(
            status.ui_url,
            "https://us-west-2.console.aws.amazon.com/batch/home?region=us-west-2#jobs/mnp-job/6afc27d7-3559-43ca-89fd-1007b6bf2546",
        )
        self.assertEqual(
            status.roles[0],
            specs.Role(
                name="echo",
                num_replicas=2,
                image="ghcr.io/pytorch/torchx:0.1.2dev0",
                entrypoint="echo",
                args=["your name"],
                env={
                    "TORCHX_ROLE_IDX": "0",
                    "TORCHX_ROLE_NAME": "echo",
                    "TORCHX_RANK0_HOST": "localhost",
                },
                resource=Resource(
                    cpu=1,
                    gpu=0,
                    memMB=1000,
                ),
            ),
        )

    def test_list(self) -> None:
        scheduler = self._mock_scheduler()
        expected_apps = [
            ListAppResponse(app_id="torchx:app-name-42", state=AppState.SUCCEEDED)
        ]
        apps = scheduler.list()
        self.assertEqual(expected_apps, apps)

    def test_list_no_jobs(self) -> None:
        scheduler = AWSBatchScheduler("test", client=MagicMock())
        scheduler._client.get_paginator.side_effect = MockPaginator(
            describe_job_queues=[
                {
                    "jobQueues": [
                        {"jobQueueName": "torchx", "state": "ENABLED"},
                    ],
                }
            ],
            list_jobs=[{"jobSummaryList": []}],
        )

        self.assertEqual([], scheduler.list())

    def test_log_iter(self) -> None:
        scheduler = self._mock_scheduler()
        logs = scheduler.log_iter("testqueue:app-name-42", "echo", k=1, regex="foo.*")
        self.assertEqual(list(logs), ["foo\n", "foobar\n"])

    def test_log_iter_running_job(self) -> None:
        scheduler = self._mock_scheduler_running_job()
        logs = scheduler.log_iter("testqueue:app-name-42", "echo", k=1, regex="foo.*")
        self.assertEqual(["foo\n", "foobar\n"], list(logs))

    def test_local_session(self) -> None:
        a: object = _local_session()
        self.assertIs(a, _local_session())

        def worker() -> None:
            b = _local_session()
            self.assertIs(b, _local_session())
            self.assertIsNot(a, b)

        t = threading.Thread(target=worker)
        t.start()
        t.join()

    def test_resource_requirement_from_resource(self) -> None:
        cpu_resource = Resource(cpu=2, memMB=1024, gpu=0)
        self.assertEqual(
            [
                {"type": "VCPU", "value": "2"},
                {"type": "MEMORY", "value": "1024"},
            ],
            resource_requirements_from_resource(cpu_resource),
        )

        zero_cpu_resource = Resource(cpu=0, memMB=1024, gpu=0)
        self.assertEqual(
            [
                {"type": "VCPU", "value": "1"},
                {"type": "MEMORY", "value": "1024"},
            ],
            resource_requirements_from_resource(zero_cpu_resource),
        )

        gpu_resource = Resource(cpu=1, memMB=1024, gpu=2)
        self.assertEqual(
            [
                {"type": "VCPU", "value": "1"},
                {"type": "MEMORY", "value": "1024"},
                {"type": "GPU", "value": "2"},
            ],
            resource_requirements_from_resource(gpu_resource),
        )

        zero_mem_resource = Resource(cpu=1, memMB=0, gpu=0)
        with self.assertRaises(AssertionError):
            resource_requirements_from_resource(zero_mem_resource)

    def test_resource_from_resource_requirements(self) -> None:
        gpu_resource_requirements = [
            {"type": "VCPU", "value": "2"},
            {"type": "GPU", "value": "3"},
            {"type": "MEMORY", "value": "1024"},
        ]

        self.assertEqual(
            Resource(cpu=2, gpu=3, memMB=1024),
            resource_from_resource_requirements(gpu_resource_requirements),
        )

        cpu_resource_requirements = [
            {"type": "VCPU", "value": "2"},
            {"type": "MEMORY", "value": "1024"},
        ]
        self.assertEqual(
            Resource(cpu=2, gpu=0, memMB=1024),
            resource_from_resource_requirements(cpu_resource_requirements),
        )

    def test_parse_num_replicas(self) -> None:
        self.assertEqual(4, _parse_num_replicas("2:5", num_nodes=6))
        self.assertEqual(6, _parse_num_replicas("0:5", num_nodes=6))
        self.assertEqual(2, _parse_num_replicas("2:3", num_nodes=6))
        self.assertEqual(3, _parse_num_replicas(":2", num_nodes=6))
        self.assertEqual(1, _parse_num_replicas("5:", num_nodes=6))
        self.assertEqual(1, _parse_num_replicas("0", num_nodes=1))
