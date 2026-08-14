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

import example_app_defs as examples_app_defs_providers
import torchx.components.integration_tests.component_provider as component_provider
from integ_test_utils import build_images, BuildInfo, push_images
from torchx.components.integration_tests.integ_tests import IntegComponentTest
from torchx.schedulers import get_scheduler_factories
from torchx.util.colors import BLUE, ENDC, GRAY

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
    parser.add_argument("--container_repo", type=str)
    return parser


def main() -> None:
    args = argparser().parse_args()
    scheduler = args.scheduler

    print("Starting components integration tests")
    torchx_image = "dummy_image"

    if scheduler in (
        "kubernetes",
        "local_docker",
    ):
        build = build_and_push_image(args.container_repo)
        torchx_image = build.torchx_image

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
    }

    params = run_parameters[scheduler]
    test_suite: IntegComponentTest = IntegComponentTest()
    for provider in params["providers"]:
        test_suite.run_components(
            module=provider,
            scheduler=scheduler,
            image=params["image"],
            # pyrefly: ignore [bad-argument-type]
            cfg=params["cfg"],
            dryrun=False,
            # pyrefly: ignore [bad-argument-type]
            workspace=params.get("workspace"),
        )


if __name__ == "__main__":
    main()
