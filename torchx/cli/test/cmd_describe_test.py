#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import unittest
from unittest.mock import patch

from torchx.cli.cmd_describe import CmdDescribe
from torchx.specs.api import AppDef, Resource, Role


class CmdDescribeTest(unittest.TestCase):
    def get_test_app(self) -> AppDef:
        resource = Resource(cpu=2, gpu=0, memMB=256)
        trainer = Role(
            name="trainer",
            image="trainer_fbpkg",
            entrypoint="trainer.par",
            args=["--arg1", "foo"],
            resource=resource,
            num_replicas=2,
        )
        return AppDef("my_train_job", roles=[trainer])

    def test_run(self) -> None:
        parser = argparse.ArgumentParser()
        cmd_describe = CmdDescribe()
        cmd_describe.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app"])

        for app in [None, self.get_test_app()]:
            with self.subTest(app=app):
                with patch(
                    "torchx.runner.api.Runner.describe",
                    return_value=app,
                ) as desc_mock:
                    try:
                        cmd_describe.run(args)
                        exit_code = None
                    except SystemExit as e:
                        exit_code = e.code

                    desc_mock.assert_called_once_with(args.app_handle)

                    if app is None:
                        self.assertEqual(exit_code, 1)
                    else:
                        self.assertIsNone(exit_code)
