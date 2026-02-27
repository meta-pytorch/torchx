# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import os
import time
import warnings
from datetime import datetime
from types import TracebackType
from typing import Any, Iterable, Mapping, Type, TYPE_CHECKING, TypeVar

from torchx.runner.events import log_event
from torchx.schedulers import get_scheduler_factories, SchedulerFactory
from torchx.schedulers.api import ListAppResponse, Scheduler, Stream
from torchx.specs import (
    AppDef,
    AppDryRunInfo,
    AppHandle,
    AppStatus,
    CfgVal,
    macros,
    make_app_handle,
    materialize_appdef,
    parse_app_handle,
    runopts,
    UnknownAppException,
    Workspace,
)
from torchx.specs.finder import get_component
from torchx.tracker.api import (
    ENV_TORCHX_JOB_ID,
    ENV_TORCHX_PARENT_RUN_ID,
    ENV_TORCHX_TRACKERS,
    tracker_config_env_var_name,
)
from torchx.util.session import get_session_id_or_create_new, TORCHX_INTERNAL_SESSION_ID
from torchx.util.types import none_throws
from torchx.workspace import WorkspaceMixin

if TYPE_CHECKING:
    from typing_extensions import Self

from .config import get_config, get_configs

logger: logging.Logger = logging.getLogger(__name__)


NONE: str = "<NONE>"
S = TypeVar("S")
T = TypeVar("T")


def get_configured_trackers() -> dict[str, str | None]:
    tracker_names = list(get_configs(prefix="torchx", name="tracker").keys())
    if ENV_TORCHX_TRACKERS in os.environ:
        logger.info(f"Using TORCHX_TRACKERS={tracker_names} as tracker names")
        tracker_names = os.environ[ENV_TORCHX_TRACKERS].split(",")

    tracker_names_with_config = {}
    for tracker_name in tracker_names:
        config_value = get_config(prefix="tracker", name=tracker_name, key="config")

        config_env_name = tracker_config_env_var_name(tracker_name)
        if config_env_name in os.environ:
            config_value = os.environ[config_env_name]
            logger.info(
                f"Using {config_env_name}={config_value} for `{tracker_name}` tracker"
            )

        tracker_names_with_config[tracker_name] = config_value
    logger.info(f"Tracker configurations: {tracker_names_with_config}")
    return tracker_names_with_config


class Runner:
    """Submits, monitors, and manages :py:class:`~torchx.specs.AppDef` jobs.

    Use :py:func:`get_runner` to create an instance with all registered schedulers.

    .. doctest::

        >>> from torchx.runner import get_runner
        >>> runner = get_runner()
        >>> runner.scheduler_backends()  # doctest: +SKIP
        ['local_cwd', 'local_docker', 'slurm', 'kubernetes', ...]

    """

    def __init__(
        self,
        name: str = "",  # session names can be empty
        scheduler_factories: dict[str, SchedulerFactory] | None = None,
        component_defaults: dict[str, dict[str, str]] | None = None,
        scheduler_params: dict[str, object] | None = None,
    ) -> None:
        self._name: str = name
        self._scheduler_factories: dict[str, SchedulerFactory] = (
            scheduler_factories or {}
        )
        self._scheduler_params: dict[str, Any] = {
            **(self._get_scheduler_params_from_env()),
            **(scheduler_params or {}),
        }
        # pyre-fixme[24]: SchedulerOpts is a generic, and we don't have access to the corresponding type
        self._scheduler_instances: dict[str, Scheduler] = {}
        self._apps: dict[AppHandle, AppDef] = {}

        # component_name -> map of component_fn_param_name -> user-specified default val encoded as str
        self._component_defaults: dict[str, dict[str, str]] = component_defaults or {}

    def _get_scheduler_params_from_env(self) -> dict[str, str]:
        scheduler_params = {}
        for key, value in os.environ.items():
            key = key.lower()
            if key.startswith("torchx_"):
                scheduler_params[key.removeprefix("torchx_")] = value
        return scheduler_params

    def __enter__(self) -> "Self":
        return self

    def __exit__(
        self,
        type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        # This method returns False so that if an error is raise within the
        # ``with`` statement, it is reraised properly
        # see: https://docs.python.org/3/reference/compound_stmts.html#with
        # see also: torchx/runner/test/api_test.py#test_context_manager_with_error
        #
        self.close()
        return False

    def close(self) -> None:
        """Closes the runner and all scheduler instances. Safe to call multiple times."""

        for scheduler in self._scheduler_instances.values():
            scheduler.close()

    def run_component(
        self,
        component: str,
        component_args: list[str] | dict[str, Any],
        scheduler: str,
        cfg: Mapping[str, CfgVal] | None = None,
        workspace: Workspace | str | None = None,
        parent_run_id: str | None = None,
    ) -> AppHandle:
        """Resolves and runs a named component.

        ``component`` resolution order (high → low):

        1. User-registered ``torchx.components`` entry points
        2. Builtins relative to ``torchx.components`` (e.g. ``"dist.ddp"``)
        3. File-based ``path/to/file.py:function_name``
        """

        with log_event("run_component") as ctx:
            dryrun_info = self.dryrun_component(
                component,
                component_args,
                scheduler,
                cfg=cfg,
                workspace=workspace,
                parent_run_id=parent_run_id,
            )
            handle = self.schedule(dryrun_info)
            app = none_throws(dryrun_info._app)

            ctx._torchx_event.workspace = str(workspace)
            ctx._torchx_event.scheduler = none_throws(dryrun_info._scheduler)
            ctx._torchx_event.app_image = app.roles[0].image
            ctx._torchx_event.app_id = parse_app_handle(handle)[2]
            ctx._torchx_event.app_metadata = app.metadata
            return handle

    def dryrun_component(
        self,
        component: str,
        component_args: list[str] | dict[str, Any],
        scheduler: str,
        cfg: Mapping[str, CfgVal] | None = None,
        workspace: Workspace | str | None = None,
        parent_run_id: str | None = None,
    ) -> AppDryRunInfo:
        """Like :py:meth:`run_component` but returns the request without submitting."""
        component_def = get_component(component)
        args_from_cli = component_args if isinstance(component_args, list) else []
        args_from_json = component_args if isinstance(component_args, dict) else {}
        app = materialize_appdef(
            component_def.fn,
            args_from_cli,
            self._component_defaults.get(component, None),
            args_from_json,
        )
        return self.dryrun(
            app,
            scheduler,
            cfg=cfg,
            workspace=workspace,
            parent_run_id=parent_run_id,
        )

    def run(
        self,
        app: AppDef,
        scheduler: str,
        cfg: Mapping[str, CfgVal] | None = None,
        workspace: Workspace | str | None = None,
        parent_run_id: str | None = None,
    ) -> AppHandle:
        """Submits an :py:class:`~torchx.specs.AppDef` and returns its :py:data:`~torchx.specs.AppHandle`."""

        with log_event(api="run") as ctx:
            dryrun_info = self.dryrun(
                app,
                scheduler,
                cfg=cfg,
                workspace=workspace,
                parent_run_id=parent_run_id,
            )
            handle = self.schedule(dryrun_info)

            event = ctx._torchx_event
            event.scheduler = scheduler
            event.runcfg = json.dumps(cfg) if cfg else None
            event.workspace = str(workspace)
            event.app_id = parse_app_handle(handle)[2]
            event.app_image = none_throws(dryrun_info._app).roles[0].image
            event.app_metadata = app.metadata

            return handle

    def schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        """Submits a previously dry-run request, allowing request mutation.

        .. code-block:: python

            dryrun_info = runner.dryrun(app, scheduler="kubernetes", cfg)
            dryrun_info.request.foo = "bar"  # mutate the raw request
            app_handle = runner.schedule(dryrun_info)

        .. warning:: Use sparingly. Overwriting many raw scheduler fields may
                     cause your usage to diverge from TorchX's supported API.
        """
        scheduler = none_throws(dryrun_info._scheduler)
        cfg = dryrun_info._cfg
        with log_event("schedule") as ctx:
            sched = self._scheduler(scheduler)
            app_id = sched.schedule(dryrun_info)
            app_handle = make_app_handle(scheduler, self._name, app_id)

            app = none_throws(dryrun_info._app)
            self._apps[app_handle] = app

            event = ctx._torchx_event
            event.scheduler = scheduler
            event.runcfg = json.dumps(cfg) if cfg else None
            event.app_id = app_id
            event.app_image = none_throws(dryrun_info._app).roles[0].image
            event.app_metadata = app.metadata

            return app_handle

    def name(self) -> str:
        return self._name

    def dryrun(
        self,
        app: AppDef,
        scheduler: str,
        cfg: Mapping[str, CfgVal] | None = None,
        workspace: Workspace | str | None = None,
        parent_run_id: str | None = None,
    ) -> AppDryRunInfo:
        """Returns what *would* be submitted without actually submitting.

        The returned :py:class:`~torchx.specs.AppDryRunInfo` can be
        ``print()``-ed for inspection or passed to :py:meth:`schedule`.
        """
        # input validation
        if not app.roles:
            raise ValueError(
                f"No roles for app: {app.name}. Did you forget to add roles to AppDef?"
            )

        if ENV_TORCHX_PARENT_RUN_ID in os.environ:
            parent_run_id = os.environ[ENV_TORCHX_PARENT_RUN_ID]
            logger.info(
                f"Using {ENV_TORCHX_PARENT_RUN_ID}={parent_run_id} env variable as tracker parent run id"
            )

        configured_trackers = get_configured_trackers()

        for role in app.roles:
            if not role.entrypoint:
                raise ValueError(
                    f"No entrypoint for role: {role.name}."
                    f" Did you forget to call role.runs(entrypoint, args, env)?"
                )
            if role.num_replicas <= 0:
                raise ValueError(
                    f"Non-positive replicas for role: {role.name}."
                    f" Did you forget to set role.num_replicas?"
                )
            # Setup tracking
            # 1. Inject parent identifier
            # 2. Inject this run's job ID
            # 3. Get the list of backends to support from .torchconfig
            #    - inject it as TORCHX_TRACKERS=names (it is expected that entrypoints are defined)
            #    - for each backend check configuration file, if exists:
            #        - inject it as TORCHX_TRACKER_<name>_CONFIGFILE=filename
            role.env[ENV_TORCHX_JOB_ID] = make_app_handle(
                scheduler, self._name, macros.app_id
            )
            role.env[TORCHX_INTERNAL_SESSION_ID] = get_session_id_or_create_new()

            if parent_run_id:
                role.env[ENV_TORCHX_PARENT_RUN_ID] = parent_run_id

            if configured_trackers:
                role.env[ENV_TORCHX_TRACKERS] = ",".join(configured_trackers.keys())

            for name, config in configured_trackers.items():
                if config:
                    role.env[tracker_config_env_var_name(name)] = config

        cfg = cfg or dict()
        with log_event(
            "dryrun",
            scheduler,
            runcfg=json.dumps(cfg) if cfg else None,
            workspace=str(workspace),
        ) as ctx:
            sched = self._scheduler(scheduler)
            resolved_cfg = sched.run_opts().resolve(cfg)

            sched._pre_build_validate(app, scheduler, resolved_cfg)

            if isinstance(sched, WorkspaceMixin):
                if workspace:
                    # NOTE: torchx originally took workspace as a runner arg and only applied the workspace to role[0]
                    # later, torchx added support for the workspace attr in Role
                    # for BC, give precedence to the workspace argument over the workspace attr for role[0]
                    if app.roles[0].workspace:
                        logger.info(
                            "Overriding role[%d] (%s) workspace to `%s`"
                            "To use the role's workspace attr pass: --workspace='' from CLI or workspace=None programmatically.",
                            0,
                            role.name,
                            str(app.roles[0].workspace),
                        )
                    app.roles[0].workspace = (
                        Workspace.from_str(workspace)
                        if isinstance(workspace, str)
                        else workspace
                    )

                sched.build_workspaces(app.roles, resolved_cfg)

            sched._validate(app, scheduler, resolved_cfg)
            dryrun_info = sched.submit_dryrun(app, resolved_cfg)
            dryrun_info._scheduler = scheduler

            event = ctx._torchx_event
            event.scheduler = scheduler
            event.runcfg = json.dumps(cfg) if cfg else None
            event.app_id = app.name
            event.app_image = none_throws(dryrun_info._app).roles[0].image
            event.app_metadata = app.metadata

            return dryrun_info

    def scheduler_run_opts(self, scheduler: str) -> runopts:
        """Returns the :py:class:`~torchx.specs.runopts` for the given scheduler."""
        return self._scheduler(scheduler).run_opts()

    def cfg_from_str(self, scheduler: str, *cfg_literal: str) -> Mapping[str, CfgVal]:
        """
        Convenience function around the scheduler's ``runopts.cfg_from_str()`` method.

        Usage:

        .. doctest::

            from torchx.runner import get_runner

            runner = get_runner()
            cfg = runner.cfg_from_str("local_cwd", "log_dir=/tmp/foobar", "prepend_cwd=True")
            assert cfg == {"log_dir": "/tmp/foobar", "prepend_cwd": True, "auto_set_cuda_visible_devices": False}
        """

        opts = self._scheduler(scheduler).run_opts()
        cfg = {}
        for cfg_str in cfg_literal:
            cfg.update(opts.cfg_from_str(cfg_str))
        return cfg

    def scheduler_backends(self) -> list[str]:
        """Returns all registered scheduler backend names."""
        return list(self._scheduler_factories.keys())

    def status(self, app_handle: AppHandle) -> AppStatus | None:
        """Returns app status, or ``None`` if the app no longer exists."""
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        with log_event("status", scheduler_backend, app_id):
            desc = scheduler.describe(app_id)
            if not desc:
                # app does not exist on the scheduler
                # remove it from apps cache if it exists
                # effectively removes this app from the list() API
                self._apps.pop(app_handle, None)
                return None

            app_status = AppStatus(
                desc.state,
                desc.num_restarts,
                msg=desc.msg,
                structured_error_msg=desc.structured_error_msg,
                roles=desc.roles_statuses,
            )
            if app_status:
                app_status.ui_url = desc.ui_url
            return app_status

    def wait(
        self, app_handle: AppHandle, wait_interval: float = 10
    ) -> AppStatus | None:
        """Blocks until the app reaches a terminal state.

        Args:
            wait_interval: seconds between status polls
        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        with log_event("wait", scheduler_backend, app_id):
            while True:
                app_status = self.status(app_handle)

                if not app_status:
                    return None
                if app_status.is_terminal():
                    return app_status
                else:
                    time.sleep(wait_interval)

    def cancel(self, app_handle: AppHandle) -> None:
        """Requests cancellation. The app transitions to ``CANCELLED`` asynchronously."""
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(app_handle)
        with log_event("cancel", scheduler_backend, app_id):
            status = self.status(app_handle)
            if status is not None and not status.is_terminal():
                scheduler.cancel(app_id)

    def delete(self, app_handle: AppHandle) -> None:
        """Deletes the app from the scheduler."""
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(app_handle)
        with log_event("delete", scheduler_backend, app_id):
            status = self.status(app_handle)
            if status is not None:
                scheduler.delete(app_id)

    def stop(self, app_handle: AppHandle) -> None:
        """.. deprecated:: Use :py:meth:`cancel` instead."""
        warnings.warn(
            "This method will be deprecated in the future, please use `cancel` instead.",
            PendingDeprecationWarning,
        )
        self.cancel(app_handle)

    def describe(self, app_handle: AppHandle) -> AppDef | None:
        """Reconstructs the :py:class:`~torchx.specs.AppDef` from the scheduler.

        Completeness is scheduler-dependent. Returns ``None`` if the app no longer exists.
        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )

        with log_event("describe", scheduler_backend, app_id):
            # if the app is in the apps list, then short circuit everything and return it
            app = self._apps.get(app_handle, None)
            if not app:
                desc = scheduler.describe(app_id)
                if desc:
                    app = AppDef(name=app_id, roles=desc.roles, metadata=desc.metadata)
            return app

    def log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        should_tail: bool = False,
        streams: Stream | None = None,
    ) -> Iterable[str]:
        """Returns an iterator over log lines for the k-th replica of a role.

        .. important:: ``k`` is the **node** (host) id, NOT the worker rank.

        .. warning:: Completeness is scheduler-dependent. Lines may be
                     partial or missing if logs have been purged. Do not use
                     this for programmatic output parsing.

        Lines include trailing whitespace (``\\n``). Use ``print(line, end="")``
        to avoid double newlines.

        Args:
            k: replica (node) index
            regex: optional filter pattern
            since: start cursor (scheduler-dependent)
            until: end cursor (scheduler-dependent)
        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        with log_event("log_lines", scheduler_backend, app_id):
            if not self.status(app_handle):
                raise UnknownAppException(app_handle)
            log_iter = scheduler.log_iter(
                app_id,
                role_name,
                k,
                regex,
                since,
                until,
                should_tail,
                streams=streams,
            )
            return log_iter

    def list(
        self,
        scheduler: str,
        cfg: Mapping[str, CfgVal] | None = None,
    ) -> list[ListAppResponse]:
        """Lists jobs on the scheduler.

        Args:
            cfg: scheduler config, used by some schedulers for backend routing.
        """
        with log_event("list", scheduler):
            sched = self._scheduler(scheduler)
            apps = sched.list(cfg)
            for app in apps:
                app.app_handle = make_app_handle(scheduler, self._name, app.app_id)
            return apps

    # pyre-fixme[24]: SchedulerOpts is a generic, and we don't have access to the corresponding type
    def _scheduler(self, scheduler: str) -> Scheduler:
        sched = self._scheduler_instances.get(scheduler)
        if not sched:
            factory = self._scheduler_factories.get(scheduler)
            if factory:
                sched = factory(self._name, **self._scheduler_params)
                self._scheduler_instances[scheduler] = sched
        if not sched:
            raise KeyError(
                f"Undefined scheduler backend: {scheduler}. Use one of: {self._scheduler_factories.keys()}"
            )
        return sched

    def _scheduler_app_id(
        self,
        app_handle: AppHandle,
        check_session: bool = True,
        # pyre-fixme[24]: SchedulerOpts is a generic, and we don't have access to the corresponding type
    ) -> tuple[Scheduler, str, str]:

        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        scheduler = self._scheduler(scheduler_backend)
        return scheduler, scheduler_backend, app_id

    def __repr__(self) -> str:
        return f"Runner(name={self._name}, schedulers={self._scheduler_factories}, apps={self._apps})"


def get_runner(
    name: str | None = None,
    component_defaults: dict[str, dict[str, str]] | None = None,
    **scheduler_params: Any,
) -> Runner:
    """Creates a :py:class:`Runner` with all registered schedulers.

    .. code-block:: python

        with get_runner() as runner:
            app_handle = runner.run(app, scheduler="kubernetes", cfg=cfg)
            print(runner.status(app_handle))

    Args:
        scheduler_params: extra kwargs passed to all scheduler constructors.
    """
    if name:
        warnings.warn(
            f"Custom session names are deprecated (detected explicitly set session name={name}). \
            To prevent this warning from showing again call `get_runner()` without the `name` param. \
            As an alternative, you can prefix the app name with the session name.",
            FutureWarning,
        )

    if not name:
        name = "torchx"

    scheduler_factories = get_scheduler_factories()
    return Runner(
        name, scheduler_factories, component_defaults, scheduler_params=scheduler_params
    )
