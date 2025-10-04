# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, Optional
from unittest import TestCase
from unittest.mock import MagicMock, patch

from torchx.schedulers.aws_sagemaker_scheduler import (
    _local_session,
    AWSSageMakerJob,
    AWSSageMakerOpts,
    AWSSageMakerScheduler,
    create_scheduler,
    JOB_STATE,
)
from torchx.specs.api import AppDryRunInfo, runopts

ENV_TORCHX_ROLE_NAME = "TORCHX_ROLE_NAME"
MODULE = "torchx.schedulers.aws_sagemaker_scheduler"


def to_millis_since_epoch(ts: datetime) -> int:
    # datetime's timestamp returns seconds since epoch
    return int(round(ts.timestamp() * 1000))


class AWSSageMakerOptsTest(TestCase):
    def setUp(self) -> None:
        self.test_dict: AWSSageMakerOpts = {
            "role": "test-arn",
            "subnets": ["subnet-1", "subnet-2"],
            "security_group_ids": ["sg-1", "sg-2"],
        }

    def test_role(self) -> None:
        self.assertEqual(self.test_dict["role"], "test-arn")
        self.assertIsInstance(self.test_dict["role"], str)

    def test_subnets(self) -> None:
        self.assertEqual(self.test_dict["subnets"], ["subnet-1", "subnet-2"])
        self.assertIsInstance(self.test_dict["subnets"], list)

    def test_security_group_ids(self) -> None:
        self.assertEqual(self.test_dict["security_group_ids"], ["sg-1", "sg-2"])
        self.assertIsInstance(self.test_dict["security_group_ids"], list)


@contextmanager
def mock_rand() -> Generator[None, None, None]:
    with patch(f"{MODULE}.make_unique") as make_unique_ctx:
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
        raise RuntimeError(
            "`op_name` not set. Did you forget to call `__call__(op_name)`?"
        )


class AWSSageMakerSchedulerTest(TestCase):
    def setUp(self) -> None:
        self.sagemaker_client = MagicMock()
        self.scheduler = AWSSageMakerScheduler(
            session_name="test-session",
            client=self.sagemaker_client,
            docker_client=MagicMock(),
        )
        self.job = AWSSageMakerJob(
            job_name="test-name",
            job_def={
                "entry_point": "some_entry_point",
                "image_uri": "some_image_uri",
                "role_arn": "some_role_arn",
            },
            images_to_push={"image1": ("tag1", "repo1")},
        )
        self.dryrun_info = AppDryRunInfo(self.job, repr)

    def _mock_scheduler(self) -> AWSSageMakerScheduler:
        scheduler = AWSSageMakerScheduler(
            "test",
            client=MagicMock(),
            docker_client=MagicMock(),
        )

        scheduler._client.get_paginator.side_effect = MockPaginator(
            describe_job_queues=[
                {
                    "ResponseMetadata": {},
                    "jobQueues": [
                        {
                            "jobQueueName": "torchx",
                            "jobQueueArn": "arn:aws:sagemaker:test-region:4000005:job-queue/torchx",
                            "state": "ENABLED",
                        },
                    ],
                }
            ],
            list_jobs=[
                {
                    "jobSummaryList": [
                        {
                            "jobArn": "arn:aws:sagemaker:us-west-2:1234567890:job/6afc27d7-3559-43ca-89fd-1007b6bf2546",
                            "jobId": "6afc27d7-3559-43ca-89fd-1007b6bf2546",
                            "jobName": "app-name-42",
                            "createdAt": 1643949940162,
                            "status": "SUCCEEDED",
                            "stoppedAt": 1643950324125,
                            "container": {"exitCode": 0},
                            "nodeProperties": {"numNodes": 2},
                            "jobDefinition": "arn:aws:sagemaker:us-west-2:1234567890:job-definition/app-name-42:1",
                        }
                    ]
                }
            ],
        )

        scheduler._client.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobArn": "arn:aws:sagemaker:us-west-2:1234567890:job/6afc27d7-3559-43ca-89fd-1007b6bf2546",
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
                                        {
                                            "name": "TORCHX_ROLE_IDX",
                                            "value": "0",
                                        },
                                        {
                                            "name": "TORCHX_ROLE_NAME",
                                            "value": "echo",
                                        },
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

        return scheduler

    @patch(f"{MODULE}.PyTorch")
    def test_schedule(self, mock_pytorch_estimator: MagicMock) -> None:
        expected_name = "test-name"
        returned_name = self.scheduler.schedule(self.dryrun_info)
        self.assertEqual(returned_name, expected_name)

    def test_run_opts(self) -> None:
        scheduler = self._mock_scheduler()
        # Call the _run_opts method
        result = scheduler._run_opts()
        # Assert that the returned value is an instance of runopts
        self.assertIsInstance(result, runopts)

    def test_cancel_existing(self) -> None:
        scheduler = self._mock_scheduler()
        # Call the _cancel_existing method
        scheduler._cancel_existing(app_id="testqueue:app-name-42")
        # Assert that it's called once
        scheduler._client.stop_training_job.assert_called_once()

    def test_list(self) -> None:
        with self.assertRaises(NotImplementedError):
            scheduler = self._mock_scheduler()
            scheduler.list()

    def test_describe_job(self) -> None:
        region = "us-east-1"
        job_id = "42"
        state = "InProgress"
        training_job = {
            "TrainingJobStatus": state,
            "TrainingJobArn": f"arn:aws:sagemaker:{region}:1234567890:training-job/{job_id})",
        }
        self.sagemaker_client.describe_training_job.return_value = training_job
        job = self.scheduler.describe(app_id=(app_id := "testqueue:app-name-42"))
        self.assertIsNotNone(job)
        self.assertEqual(job.app_id, app_id)
        self.assertEqual(job.state, JOB_STATE[state])
        self.assertEqual(
            job.ui_url,
            f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#jobs/{job_id}",
        )

    def test_log_iter(self) -> None:
        with self.assertRaises(NotImplementedError):
            scheduler = self._mock_scheduler()
            scheduler.log_iter(
                app_id="testqueue:app-name-42",
                role_name="echo",
                k=1,
                regex="foo.*",
            )

    def test_get_job(self) -> None:
        # Arrange
        scheduler = self._mock_scheduler()

        # Act
        test_job = scheduler._get_job(app_id="testqueue:app-name-42")

        # Assert
        self.assertEqual(test_job, scheduler._client.describe_training_job.return_value)

    def test_job_ui_url(self) -> None:
        # Set up the input job ARN and expected URL
        job_arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job/job-id"
        expected_url = "https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#jobs/job-id"

        # Call the _job_ui_url method
        result = self.scheduler._job_ui_url(job_arn)

        # Assert that the returned URL matches the expected URL
        self.assertEqual(result, expected_url)

    def test_job_ui_url_with_invalid_arn(self) -> None:
        # Set up an invalid job ARN
        job_arn = "invalid-arn"

        # Call the _job_ui_url method
        result = self.scheduler._job_ui_url(job_arn)

        # Assert that the returned value is None
        self.assertIsNone(result)

    def test_job_ui_url_with_no_match(self) -> None:
        # Set up a job ARN that does not match the regex pattern
        job_arn = "arn:aws:sagemaker:us-east-1:123456789012:training-job"

        # Call the _job_ui_url method
        result = self.scheduler._job_ui_url(job_arn)

        # Assert that the returned value is None
        self.assertIsNone(result)

    def test_parse_args(self) -> None:
        # Set up the role_args with no match
        role_args = ["arg1", "arg2", "arg3"]

        # Call the _parse_entrypoint_and_source_dir method
        with self.assertRaises(ValueError):
            self.scheduler._parse_args(role_args)

    def test_parse_args_with_overrides(self) -> None:
        # Set up the args
        test_args = [
            "--",
            "--config-path",
            "test-path/test-config",
            "--config-name",
            "config.yaml",
            "--overrides",
            "key1=value1",
        ]

        # Call the _parse_arguments method
        result = self.scheduler._parse_args(test_args)

        # Assert the returned values
        expected_args = OrderedDict(
            [
                ("config-path", "test-path/test-config"),
                ("config-name", "config.yaml"),
                ("overrides", "key1=value1"),
            ]
        )
        self.assertEqual(result, ("--", expected_args))

    def test_parse_args_without_overrides(self) -> None:
        # Set up the args
        test_args = [
            "--",
            "--config-path",
            "test-path/test-config",
            "--config-name",
            "config.yaml",
        ]

        # Call the _parse_arguments method
        result = self.scheduler._parse_args(test_args)

        # Assert the returned values
        expected_args = OrderedDict(
            [
                ("config-path", "test-path/test-config"),
                ("config-name", "config.yaml"),
            ]
        )
        self.assertEqual(result, ("--", expected_args))

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

    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler(session_name="test-sm")
        self.assertIsInstance(scheduler, AWSSageMakerScheduler)


if __name__ == "__main__":
    unittest.main()
