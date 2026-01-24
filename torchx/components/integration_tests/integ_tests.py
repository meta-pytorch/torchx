# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import logging
import sys
from dataclasses import asdict
from json import dumps
from types import ModuleType
from typing import Callable, cast

from torchx.cli.cmd_log import get_logs
from torchx.components.integration_tests.component_provider import ComponentProvider
from torchx.runner import get_runner
from torchx.specs import AppHandle, AppState, AppStatus, CfgVal
from torchx.util.types import none_throws


log: logging.Logger = logging.getLogger(__name__)


class IntegComponentTest:
    """
    Class contains common methods for executing e2e integration tests of components on different schedulers.
    The providers of `AppDef` must implement `torchx.components.integration_tests.component_provider.ComponentProvider`.

    >>> import torchx.specs as specs
    >>> import torchx.components.integration_tests.integ_tests as integ_tests
    >>> from torchx.components.integration_tests.component_provider import ComponentProvider
    >>> class MyComponentProvider(ComponentProvider):
    ...     def get_app_def(self):
    ...         return specs.AppDef(...)
    >>> IntegComponentTest(...).run_component(my_module)

    #TODO(aivanou): implement `run_component` method that allows running a single component provider
    """

    def run_components(
        self,
        module: ModuleType,
        image: str,
        scheduler: str,
        cfg: dict[str, CfgVal],
        dryrun: bool = False,
        workspace: str | None = None,
    ) -> None:
        component_providers = [
            cast(Callable[..., ComponentProvider], cls)(scheduler, image)
            for cls in self._get_component_providers(module)
        ]

        # get all the appdefs from providers
        app_defs = []
        for provider in component_providers:
            provider.setUp()
            try:
                app_defs.append(provider.get_app_def())
            finally:
                provider.tearDown()

        # submit all the jobs asynchronously
        jobs: dict[AppHandle, AppStatus] = {}
        runner = get_runner("test-runner")

        for app_def in app_defs:
            log.info(f"Submitting AppDef... (dryrun={dryrun})")
            # get the dryrun info to log the scheduler request
            # then use the schedule (intead of the run API) for job submission
            dryrun_info = runner.dryrun(
                app_def, scheduler, cfg=cfg, workspace=workspace
            )
            log.info(f"\nAppDef:\n{dumps(asdict(app_def), indent=4)}")
            log.info(f"\nScheduler Request:\n{dryrun_info}")

            if not dryrun:
                app_handle = runner.schedule(dryrun_info)
                status = runner.status(app_handle)
                jobs[app_handle] = none_throws(status)
                log.info(f"Submitted Application ({app_handle})")
            else:
                log.info(
                    f"Dryrun, not submitting application to scheduler=`{scheduler}`"
                )

        # batch wait for the states
        for app_handle, status in jobs.items():
            log.info(f"Waiting for {app_handle} to finish...")
            status = none_throws(runner.wait(app_handle))
            log.info(
                f"App ({app_handle}) finished with state=`{status.state}` and msg=`{status.msg}` (see application log lines below)"
            )
            jobs[app_handle] = status

        # print the logs for all jobs
        for app_handle, status in jobs.items():
            print(f"=== BEGIN LOG {app_handle} ({status.state})===")
            # pass the same runner used to run the component so that log iter works with local scheduler (which is locally stateful)
            get_logs(sys.stdout, app_handle, regex=None, runner=runner)
            print(f"=== END LOG {app_handle} ({status.state}) ===")

        # log summary of states
        group_by_state: dict[AppState, list[AppHandle]] = {}
        for app_handle, status in jobs.items():
            handles = group_by_state.setdefault(status.state, [])
            handles.append(app_handle)

        log.info(f"\n{'*'*40}{dumps(group_by_state, indent=4)}\n{'*' * 40}")

        # assert that all jobs have been successful (jobs not in final state SUCCEEDED are considered a failure)
        num_total = len(jobs)
        num_failed = num_total - len(group_by_state.get(AppState.SUCCEEDED, []))
        if num_failed > 0:
            raise AssertionError(f"{num_failed}/{num_total} test jobs failed")

    def _get_component_providers(
        self, module: ModuleType
    ) -> list[type[ComponentProvider]]:
        providers = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                inspect.isclass(attr)
                and issubclass(attr, ComponentProvider)
                and not inspect.isabstract(attr)
            ):
                providers.append(attr)
        return providers
