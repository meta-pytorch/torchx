# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torchx.components.utils as utils
from torchx.components.component_test_base import ComponentTestCase
from torchx.specs import AppState


class UtilsComponentTest(ComponentTestCase):
    def test_sh(self) -> None:
        self.validate(utils, "sh")

    def test_python(self) -> None:
        self.validate(utils, "python")

    def test_binary(self) -> None:
        self.validate(utils, "binary")

    def test_touch(self) -> None:
        self.validate(utils, "touch")

    def test_echo(self) -> None:
        self.validate(utils, "echo")

    def test_copy(self) -> None:
        self.validate(utils, "copy")

    def test_booth(self) -> None:
        self.validate(utils, "booth")

    def test_run_sh(self) -> None:
        result = self.run_component(
            utils.echo, {"msg": "from test"}, scheduler_params={"cache_size": 1}
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.state, AppState.SUCCEEDED)

    def test_run_sh_scheduler_factory_failure(self) -> None:
        self.assertRaises(
            ValueError,
            lambda: self.run_component(
                utils.echo, {"msg": "from test"}, scheduler_params={"cache_size": -1}
            ),
        )
