#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os.path
import tempfile
import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

import requests
from torchx.apps.serve.serve import main


class ServeTest(unittest.TestCase):
    @patch("requests.post")
    def test_serve(self, post: MagicMock) -> None:
        test_body: str = "test model"

        def request(
            url: str, params: Dict[str, str], timeout: int
        ) -> requests.Response:
            self.assertEqual(timeout, 60)
            self.assertEqual(url, "http://localhost:1234/models")
            self.assertEqual(params["model_name"], "modelname")
            self.assertEqual(params["initial_workers"], "2")
            self.assertNotIn("response_timeout", params)

            # download model
            r = requests.get(params["url"])
            r.raise_for_status()
            self.assertEqual(r.text, test_body)

            resp = requests.Response()
            resp.status_code = 200
            return resp

        post.side_effect = request

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.mar")
            with open(model_path, "w") as f:
                f.write(test_body)

            main(
                [
                    "--model_path",
                    model_path,
                    "--management_api",
                    "http://localhost:1234",
                    "--model_name",
                    "modelname",
                    "--initial_workers",
                    "2",
                    # use ephemeral port to avoid stress test collisions
                    "--port",
                    "0",
                ]
            )
