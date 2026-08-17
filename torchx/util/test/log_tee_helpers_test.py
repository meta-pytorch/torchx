# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest
from unittest.mock import MagicMock, patch

from torchx.schedulers.api import Stream
from torchx.util.log_tee_helpers import tee_logs


class TeeLogsTest(unittest.TestCase):
    def test_tee_logs_forwards_params(self) -> None:
        dst = io.StringIO()
        runner = MagicMock()
        with patch(
            "torchx.util.log_tee_helpers._start_threads_to_monitor_role_replicas"
        ) as monitor_mock:
            thread = tee_logs(
                dst=dst,
                app_handle="local://test_session/test_app",
                regex="foo.*",
                runner=runner,
                should_tail=False,
                streams=Stream.STDERR,
                colorize=True,
            )
            thread.start()
            thread.join()

        monitor_mock.assert_called_once_with(
            dst=dst,
            runner=runner,
            app_handle="local://test_session/test_app",
            regex="foo.*",
            should_tail=False,
            streams=Stream.STDERR,
            colorize=True,
        )
