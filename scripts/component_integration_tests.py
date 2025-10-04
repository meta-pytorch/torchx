#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Kubernetes integration tests.
"""
import argparse
import logging
import os
from uuid import uuid4

import example_app_defs as examples_app_defs_providers
import torchx.components.integration_tests.component_provider as component_provider
from integ_test_utils import build_images, BuildInfo, push_images
from torchx.cli.colors import BLUE, ENDC, GRAY
from torchx.components.integration_tests.integ_tests import IntegComponentTest
from torchx.schedulers import get_scheduler_factories


logging.basicConfig(
    level=logging.INFO,
    format=f"{GRAY}%(asctime)s{ENDC} {BLUE}%(name)-12s{ENDC} %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def build_and_push_image(container_repo: str) -> BuildInfo:
    build = build_images()
    push_images(build, container_repo=container_repo)
    return build


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TorchX integration tests.")
    choices = list(get_scheduler_factories().keys())
    parser.add_argument("--scheduler", required=True, choices=choices)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--container_repo", type=str)
    return parser


def main() -> None:
    args = argparser().parse_args()
    scheduler = args.scheduler

    print("Starting components integration tests")
    torchx_image = "dummy_image"
    dryrun = False

    if args.mock:
        if scheduler == "aws_batch":
            _mock_aws_batch()
        else:
            raise ValueError(f"mocking is not supported for {scheduler}")

    if scheduler in (
        "kubernetes",
        "kubernetes_mcad",
        "local_docker",
        "aws_batch",
        "lsf",
    ):
        build = build_and_push_image(args.container_repo)
        torchx_image = build.torchx_image

    if args.container_repo == "" and scheduler == "aws_batch" and not args.mock:
        dryrun = True
        print("Skip running tests, executed only docker build step")

    run_parameters = {
        "kubernetes": {
            "providers": [
                component_provider,
                examples_app_defs_providers,
            ],
            "image": torchx_image,
            "cfg": {
                "namespace": "torchx-dev",
                "queue": "default",
            },
        },
        "kubernetes_mcad": {
            "providers": [
                component_provider,
                examples_app_defs_providers,
            ],
            "image": torchx_image,
            "cfg": {
                "namespace": "torchx-dev",
            },
        },
        "local_cwd": {
            "providers": [
                component_provider,
            ],
            "image": os.getcwd(),
            "cfg": {},
        },
        "local_docker": {
            "providers": [
                component_provider,
                examples_app_defs_providers,
            ],
            "image": torchx_image,
            "cfg": {},
        },
        "aws_batch": {
            "providers": [
                component_provider,
            ],
            "image": torchx_image,
            "cfg": {
                "queue": "torchx",
            },
        },
        "lsf": {
            "providers": [
                component_provider,
            ],
            "image": torchx_image,
            "cfg": {
                "runtime": "docker",
                "jobdir": "/mnt/data/torchx",
                "host_network": True,
            },
        },
    }

    params = run_parameters[scheduler]
    test_suite: IntegComponentTest = IntegComponentTest()
    for provider in params["providers"]:
        test_suite.run_components(
            module=provider,
            scheduler=scheduler,
            image=params["image"],
            cfg=params["cfg"],
            dryrun=dryrun,
            workspace=params.get("workspace"),
        )


def _mock_aws_batch() -> None:
    """
    This sets up a mock AWS batch backend that uses Docker to execute the jobs
    locally.
    """
    # setup the docker network so DNS works correctly
    from torchx.schedulers.docker_scheduler import ensure_network, NETWORK

    ensure_network()
    os.environ.setdefault("MOTO_DOCKER_NETWORK_NAME", NETWORK)

    from moto import mock_aws

    mock_aws().__enter__()

    import boto3.session

    session = boto3.session.Session()
    batch_client = session.client("batch")
    iam_client = session.client("iam")
    ec2_client = session.client("ec2")

    # Setup code is from:
    # https://github.com/getmoto/moto/blob/master/tests/test_batch/__init__.py

    # setup ec2
    resp = ec2_client.create_vpc(CidrBlock="172.30.0.0/24")
    vpc_id = resp["Vpc"]["VpcId"]
    resp = ec2_client.create_subnet(
        AvailabilityZone="eu-central-1a", CidrBlock="172.30.0.0/25", VpcId=vpc_id
    )
    subnet_id = resp["Subnet"]["SubnetId"]
    resp = ec2_client.create_security_group(
        Description="test_sg_desc", GroupName=str(uuid4())[0:6], VpcId=vpc_id
    )
    sg_id = resp["GroupId"]

    # setup IAM
    role_name = f"{str(uuid4())[0:6]}"
    resp = iam_client.create_role(
        RoleName=role_name, AssumeRolePolicyDocument="some_policy"
    )
    iam_arn = resp["Role"]["Arn"]
    iam_client.create_instance_profile(InstanceProfileName=role_name)
    iam_client.add_role_to_instance_profile(
        InstanceProfileName=role_name, RoleName=role_name
    )

    resp = batch_client.create_compute_environment(
        computeEnvironmentName="torchx",
        type="UNMANAGED",
        state="ENABLED",
        serviceRole=iam_arn,
    )
    arn = resp["computeEnvironmentArn"]

    resp = batch_client.create_job_queue(
        jobQueueName="torchx",
        state="ENABLED",
        priority=123,
        computeEnvironmentOrder=[{"order": 123, "computeEnvironment": arn}],
    )


if __name__ == "__main__":
    main()
