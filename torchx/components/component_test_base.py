# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
You can unit test the component definitions as you would normal Python code
since they are valid Python definitions.

We do recommend using :py:class:`ComponentTestCase` to ensure that your
component can be parsed by the TorchX CLI. The CLI requires stricter formatting
on the doc string than pure Python as the doc string is used for parsing CLI
args.
"""

import os
import shutil
import tempfile
import time
import unittest
from types import ModuleType
from typing import Any, Callable

from torchx.runner import get_runner
from torchx.specs import AppDef, AppStatus
from torchx.specs.builders import _create_args_parser
from torchx.specs.finder import get_component


class ComponentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp("torchx_component_test")

        self.old_cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(__file__)))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)
        os.chdir(self.old_cwd)

    """
    ComponentTestCase is an extension of TestCase with helper methods for use
    with testing component definitions.

    .. doctest:: [component_test_case]

     import unittest
     from torchx.components.component_test_base import ComponentTestCase
     from torchx.components import utils
     class MyComponentTest(ComponentTestCase):
         def test_my_comp(self):
             self.validate(utils, "copy")
     MyComponentTest("test_my_comp").run()

    """

    def validate(self, module: ModuleType, function_name: str) -> None:
        """
        Validates the component by effectively running:

        .. code-block:: shell-session

         $ torchx run COMPONENT.py:FN --help

        """

        # make it the same as a custom component (e.g. /abs/path/to/component.py:train)
        module_path = module.__file__
        assert module_path, f"module must have __file__: {module_path}"
        module_path = os.path.abspath(module_path)
        component_id = f"{module_path}:{function_name}"
        component_def = get_component(component_id)

        # on `--help` argparse will print the help message and exit 0
        # just make sure that happens; if the component has errors then
        # this will raise an exception and the test will fail
        with self.assertRaises(SystemExit):
            _ = _create_args_parser(component_def.fn).parse_args(["--help"])

    def run_component(
        self,
        component: Callable[..., AppDef],
        args: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        scheduler: str = "local_cwd",
        interval: float = 0.1,
        timeout: float = 1,
    ) -> AppStatus | None:
        """
        Helper function that hides complexity of setting up the runner and polling results.
        Note: method is blocking until either scheduler exits or timeout is reached (for non-blocking schedulers).

        Args:
                components: component function, factory for AppDef
                args: optional component factory arguments
                scheduler_params: optional parameters for scheduler factory method
                scheduler: scheduler name
                interval: scheduler comppletion polling interval
                timeout: max time for scheduler to complete

        """

        app_def = component(**args)
        if scheduler_params:
            runner = get_runner(name=None, component_defaults=None, **scheduler_params)
        else:
            runner = get_runner(name=None, component_defaults=None)

        app_handle = runner.run(app_def, scheduler)

        elapsed = 0
        status = runner.status(app_handle)
        while (status and not status.is_terminal) or elapsed < timeout:
            time.sleep(interval)
            elapsed += interval
            status = runner.status(app_handle)

        return status
