# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import threading
import unittest
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Iterable
from unittest import TestCase
from unittest.mock import MagicMock, patch

from torchx.schedulers.aws_sagemaker_scheduler import (
    _build_model_trainer,
    _local_session,
    AWSSageMakerJob,
    AWSSageMakerScheduler,
    create_scheduler,
    JOB_STATE,
    Opts,
)
from torchx.specs.api import AppDef, AppDryRunInfo, CfgVal, Role, runopts

ENV_TORCHX_ROLE_NAME = "TORCHX_ROLE_NAME"
MODULE = "torchx.schedulers.aws_sagemaker_scheduler"


def to_millis_since_epoch(ts: datetime) -> int:
    # datetime's timestamp returns seconds since epoch
    return int(round(ts.timestamp() * 1000))


class AWSSageMakerOptsTest(TestCase):
    def setUp(self) -> None:
        self.test_opts: Opts = Opts(
            role="test-arn",
            instance_type="ml.m5.large",
            subnets=["subnet-1", "subnet-2"],
            security_group_ids=["sg-1", "sg-2"],
        )

    def test_role(self) -> None:
        self.assertEqual(self.test_opts.role, "test-arn")
        self.assertIsInstance(self.test_opts.role, str)

    def test_subnets(self) -> None:
        self.assertEqual(self.test_opts.subnets, ["subnet-1", "subnet-2"])
        self.assertIsInstance(self.test_opts.subnets, list)

    def test_security_group_ids(self) -> None:
        self.assertEqual(self.test_opts.security_group_ids, ["sg-1", "sg-2"])
        self.assertIsInstance(self.test_opts.security_group_ids, list)


@contextmanager
def mock_rand() -> Generator[None, None, None]:
    with patch(f"{MODULE}.make_unique") as make_unique_ctx:
        make_unique_ctx.return_value = "app-name-42"
        yield


boto3Response = dict[str, object]  # boto3 responses are JSON


class MockPaginator:
    """
    Used for mocking ``boto3.client("<SERVICE>").get_paginator("<API>")`` calls.
    """

    def __init__(self, **op_to_pages: Iterable[boto3Response]) -> None:
        # boto3 paginators return an iterable of API responses
        self.op_to_pages: dict[str, Iterable[boto3Response]] = op_to_pages
        self.op_name: str | None = None

    def __call__(self, op_name: str) -> "MockPaginator":
        self.op_name = op_name
        return self

    def paginate(self, *_1: object, **_2: object) -> Iterable[dict[str, object]]:
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
                "training_image": "some_image_uri",
                "role": "some_role_arn",
                "source_code": {
                    "entry_script": "some_entry_point",
                    "source_dir": "/some/dir",
                },
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

    @patch(f"{MODULE}._build_model_trainer")
    def test_schedule(self, mock_build_trainer: MagicMock) -> None:
        mock_trainer = MagicMock()
        mock_training_job = MagicMock()
        mock_training_job.get_name.return_value = "test-name-2026-02-12"
        mock_trainer._latest_training_job = mock_training_job
        mock_build_trainer.return_value = mock_trainer

        returned_name = self.scheduler.schedule(self.dryrun_info)

        mock_build_trainer.assert_called_once_with(self.job.job_def)
        mock_trainer.train.assert_called_once_with(wait=False)
        # Should return the actual SageMaker job name, not req.job_name
        self.assertEqual(
            returned_name,
            "test-name-2026-02-12",
            "schedule() should return the actual SageMaker job name from trainer",
        )

    @patch(f"{MODULE}._build_model_trainer")
    def test_schedule_fallback_when_no_training_job(
        self, mock_build_trainer: MagicMock
    ) -> None:
        """When _latest_training_job is None, fall back to req.job_name."""
        mock_trainer = MagicMock()
        mock_trainer._latest_training_job = None
        mock_build_trainer.return_value = mock_trainer

        returned_name = self.scheduler.schedule(self.dryrun_info)

        self.assertEqual(
            returned_name,
            "test-name",
            "should fall back to req.job_name when _latest_training_job is None",
        )

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

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_tags_from_cfg(self, mock_make_unique: MagicMock) -> None:
        """Test that tags from cfg are properly converted and merged with app.metadata."""
        mock_make_unique.return_value = "test-job-42"

        # Create a minimal AppDef with metadata
        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(
            name="test-app",
            roles=[role],
            metadata={"app_key": "app_value"},
        )

        # Create cfg with tags as dict[str, str] (the CLI-parseable format)
        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "tags": {"env": "prod", "team": "ml"},
        }

        # Call submit_dryrun (the public API that resolves cfg)
        # pyre-ignore[6]: Testing with raw cfg dict, not AWSSageMakerOpts
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)

        # Verify tags are converted to SageMaker v3 format and merged
        job_def = dryrun_info.request.job_def
        tags = job_def["tags"]

        # Should have 3 tags: 2 from cfg + 1 from app.metadata
        self.assertEqual(len(tags), 3)

        # Verify cfg tags use lowercase key/value (sagemaker v3 Tag format)
        self.assertIn({"key": "env", "value": "prod"}, tags)
        self.assertIn({"key": "team", "value": "ml"}, tags)

        # Verify app.metadata is also included
        self.assertIn({"key": "app_key", "value": "app_value"}, tags)

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_tags_empty_cfg(self, mock_make_unique: MagicMock) -> None:
        """Test that when tags is empty dict, only app.metadata tags are included."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(
            name="test-app",
            roles=[role],
            metadata={"meta_key": "meta_value"},
        )

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            # tags defaults to {} via runopts
        }

        # pyre-ignore[6]: Testing with raw cfg dict, not AWSSageMakerOpts
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)

        job_def = dryrun_info.request.job_def
        tags = job_def["tags"]

        # Should only have the app.metadata tag
        self.assertEqual(len(tags), 1)
        self.assertIn({"key": "meta_key", "value": "meta_value"}, tags)

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_job_def_structure(self, mock_make_unique: MagicMock) -> None:
        """Test that _submit_dryrun produces correct structured job_def for ModelTrainer."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 2,
        }

        # pyre-ignore[6]: Testing with raw cfg dict, not AWSSageMakerOpts
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        # Verify structured fields
        self.assertEqual(job_def["training_image"], "test-image:latest")
        self.assertTrue(job_def["distributed"])
        self.assertEqual(job_def["role"], "arn:aws:iam::123456789:role/test-role")

        # Verify source_code structure
        source_code = job_def["source_code"]
        self.assertEqual(source_code["entry_script"], "-m train")
        self.assertIn("source_dir", source_code)

        # Verify compute structure
        compute = job_def["compute"]
        self.assertEqual(compute["instance_type"], "ml.p3.2xlarge")
        self.assertEqual(compute["instance_count"], 2)

        # Verify hyperparameters were parsed
        self.assertIn("epochs", job_def["hyperparameters"])
        self.assertEqual(job_def["hyperparameters"]["epochs"], "10")

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_base_job_name_default(
        self, mock_make_unique: MagicMock
    ) -> None:
        """Verify base_job_name defaults to TorchX-generated job_name."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        self.assertEqual(
            job_def["base_job_name"],
            "test-job-42",
            "base_job_name should default to TorchX job_name when not set in cfg",
        )

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_base_job_name_from_cfg(
        self, mock_make_unique: MagicMock
    ) -> None:
        """Verify user-provided base_job_name from cfg takes precedence."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "base_job_name": "user-custom-name",
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        self.assertEqual(
            job_def["base_job_name"],
            "user-custom-name",
            "base_job_name from cfg should take precedence over TorchX job_name",
        )

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_networking_config(self, mock_make_unique: MagicMock) -> None:
        """Verify networking config is correctly structured in job_def."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "subnets": ["subnet-a", "subnet-b"],
            "security_group_ids": ["sg-1"],
            "enable_network_isolation": True,
            "encrypt_inter_container_traffic": True,
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        networking = job_def["networking"]
        self.assertEqual(networking["subnets"], ["subnet-a", "subnet-b"])
        self.assertEqual(networking["security_group_ids"], ["sg-1"])
        self.assertTrue(networking["enable_network_isolation"])
        self.assertTrue(networking["enable_inter_container_traffic_encryption"])

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_stopping_condition(
        self, mock_make_unique: MagicMock
    ) -> None:
        """Verify stopping condition config is correctly structured."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "max_run": 86400,
            "max_wait": 172800,
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        stopping = job_def["stopping_condition"]
        self.assertEqual(stopping["max_runtime_in_seconds"], 86400)
        self.assertEqual(stopping["max_wait_time_in_seconds"], 172800)

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_output_data_config(
        self, mock_make_unique: MagicMock
    ) -> None:
        """Verify output data config includes s3_output_path, kms, and compression."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "output_path": "s3://my-bucket/output",
            "output_kms_key": "arn:aws:kms:us-east-1:123:key/abc",
            "disable_output_compression": True,
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        output_data = job_def["output_data_config"]
        self.assertEqual(output_data["s3_output_path"], "s3://my-bucket/output")
        self.assertEqual(output_data["kms_key_id"], "arn:aws:kms:us-east-1:123:key/abc")
        self.assertEqual(output_data["compression_type"], "NONE")

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_output_data_no_path_warns(
        self, mock_make_unique: MagicMock
    ) -> None:
        """When output_kms_key is set without output_path, output config is skipped."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "output_kms_key": "arn:aws:kms:us-east-1:123:key/abc",
        }

        with self.assertLogs(
            "torchx.schedulers.aws_sagemaker_scheduler", level="WARNING"
        ) as cm:
            # pyre-ignore[6]: Testing with raw cfg dict
            dryrun_info = self.scheduler.submit_dryrun(app, cfg)

        job_def = dryrun_info.request.job_def
        self.assertNotIn(
            "output_data_config",
            job_def,
            "output_data_config should be skipped when output_path is missing",
        )
        self.assertTrue(
            any("output_path" in msg for msg in cm.output),
            "should warn about missing output_path",
        )

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_checkpoint_config(self, mock_make_unique: MagicMock) -> None:
        """Verify checkpoint config is correctly structured."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "checkpoint_s3_uri": "s3://my-bucket/checkpoints",
            "checkpoint_local_path": "/opt/ml/checkpoints",
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        job_def = dryrun_info.request.job_def

        ckpt = job_def["checkpoint_config"]
        self.assertEqual(ckpt["s3_uri"], "s3://my-bucket/checkpoints")
        self.assertEqual(ckpt["local_path"], "/opt/ml/checkpoints")

    @patch(f"{MODULE}.make_unique")
    def test_submit_dryrun_compute_optional_fields(
        self, mock_make_unique: MagicMock
    ) -> None:
        """Verify optional compute fields (volume_size, spot, etc.) are mapped."""
        mock_make_unique.return_value = "test-job-42"

        role = Role(
            name="trainer",
            image="test-image:latest",
            args=["-m", "train", "--epochs=10"],
        )
        app = AppDef(name="test-app", roles=[role])

        cfg: dict[str, CfgVal] = {
            "role": "arn:aws:iam::123456789:role/test-role",
            "instance_type": "ml.p3.2xlarge",
            "instance_count": 1,
            "volume_size": 100,
            "volume_kms_key": "arn:aws:kms:key",
            "keep_alive_period_in_seconds": 3600,
            "use_spot_instances": True,
        }

        # pyre-ignore[6]: Testing with raw cfg dict
        dryrun_info = self.scheduler.submit_dryrun(app, cfg)
        compute = dryrun_info.request.job_def["compute"]

        self.assertEqual(compute["volume_size_in_gb"], 100)
        self.assertEqual(compute["volume_kms_key_id"], "arn:aws:kms:key")
        self.assertEqual(compute["keep_alive_period_in_seconds"], 3600)
        self.assertTrue(compute["enable_managed_spot_training"])


try:
    # pyre-fixme[21]: Could not find module `sagemaker.train`.
    import sagemaker.train  # noqa: F401

    _HAS_SAGEMAKER_V3 = True
except ImportError:
    _HAS_SAGEMAKER_V3 = False


@unittest.skipUnless(_HAS_SAGEMAKER_V3, "requires sagemaker>=3.2")
class BuildModelTrainerTest(TestCase):
    """Tests for the _build_model_trainer function."""

    @patch("sagemaker.train.ModelTrainer")
    def test_minimal_job_def(self, mock_model_trainer_cls: MagicMock) -> None:
        """Test with only required fields."""
        mock_trainer = MagicMock()
        mock_model_trainer_cls.return_value = mock_trainer

        job_def = {
            "training_image": "my-image:latest",
            "role": "arn:aws:iam::123:role/test",
        }

        result = _build_model_trainer(job_def)

        mock_model_trainer_cls.assert_called_once()
        call_kwargs = mock_model_trainer_cls.call_args[1]
        self.assertEqual(call_kwargs["training_image"], "my-image:latest")
        self.assertEqual(call_kwargs["role"], "arn:aws:iam::123:role/test")
        self.assertIs(result, mock_trainer)

    @patch("sagemaker.train.ModelTrainer")
    def test_distributed_flag(self, mock_model_trainer_cls: MagicMock) -> None:
        """Test that distributed=True creates a Torchrun instance."""
        mock_trainer = MagicMock()
        mock_model_trainer_cls.return_value = mock_trainer

        job_def = {
            "training_image": "my-image:latest",
            "distributed": True,
        }

        _build_model_trainer(job_def)

        call_kwargs = mock_model_trainer_cls.call_args[1]
        # Verify Torchrun was passed (it's imported from sagemaker)
        # pyre-fixme[21]: Could not find module `sagemaker.train.distributed`.
        from sagemaker.train.distributed import Torchrun

        self.assertIsInstance(call_kwargs["distributed"], Torchrun)

    @patch("sagemaker.train.ModelTrainer")
    def test_tags_converted_to_tag_objects(
        self, mock_model_trainer_cls: MagicMock
    ) -> None:
        """Test that tag dicts are converted to Tag objects."""
        mock_trainer = MagicMock()
        mock_model_trainer_cls.return_value = mock_trainer

        job_def = {
            "training_image": "my-image:latest",
            "tags": [
                {"key": "env", "value": "prod"},
                {"key": "team", "value": "ml"},
            ],
        }

        _build_model_trainer(job_def)

        call_kwargs = mock_model_trainer_cls.call_args[1]
        tags = call_kwargs["tags"]
        self.assertEqual(len(tags), 2)

        # pyre-fixme[21]: Could not find module `sagemaker.core.shapes.shapes`.
        from sagemaker.core.shapes.shapes import Tag

        self.assertIsInstance(tags[0], Tag)
        self.assertEqual(tags[0].key, "env")
        self.assertEqual(tags[0].value, "prod")

    @patch("sagemaker.train.ModelTrainer")
    def test_metric_definitions_builder(
        self, mock_model_trainer_cls: MagicMock
    ) -> None:
        """Test that metric_definitions uses the builder method."""
        mock_trainer = MagicMock()
        mock_trainer.with_metric_definitions.return_value = mock_trainer
        mock_model_trainer_cls.return_value = mock_trainer

        job_def = {
            "training_image": "my-image:latest",
            "metric_definitions": [("loss", "Loss: (\\S+)"), ("acc", "Acc: (\\S+)")],
        }

        result = _build_model_trainer(job_def)

        mock_trainer.with_metric_definitions.assert_called_once()
        self.assertIs(result, mock_trainer)

    @patch("sagemaker.train.ModelTrainer")
    def test_retry_strategy_builder(self, mock_model_trainer_cls: MagicMock) -> None:
        """Test that max_retry_attempts uses the builder method."""
        mock_trainer = MagicMock()
        mock_trainer.with_retry_strategy.return_value = mock_trainer
        mock_model_trainer_cls.return_value = mock_trainer

        job_def = {
            "training_image": "my-image:latest",
            "max_retry_attempts": 3,
        }

        _build_model_trainer(job_def)

        mock_trainer.with_retry_strategy.assert_called_once()

    @patch("sagemaker.train.ModelTrainer")
    def test_infra_check_builder(self, mock_model_trainer_cls: MagicMock) -> None:
        """Test that enable_infra_check uses the builder method."""
        mock_trainer = MagicMock()
        mock_trainer.with_infra_check_config.return_value = mock_trainer
        mock_model_trainer_cls.return_value = mock_trainer

        job_def = {
            "training_image": "my-image:latest",
            "enable_infra_check": True,
        }

        _build_model_trainer(job_def)

        mock_trainer.with_infra_check_config.assert_called_once()


if __name__ == "__main__":
    unittest.main()
