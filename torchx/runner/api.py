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
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

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


def get_configured_trackers() -> Dict[str, Optional[str]]:
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
    """
    TorchX individual component runner. Has the methods for the user to
    act upon ``AppDefs``. The ``Runner`` will cache information about the
    launched apps if they were launched locally otherwise it's up to the
    specific scheduler implementation.
    """

    def __init__(
        self,
        name: str,
        scheduler_factories: Dict[str, SchedulerFactory],
        component_defaults: Optional[Dict[str, Dict[str, str]]] = None,
        scheduler_params: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Creates a new runner instance.

        Args:
            name: the human readable name for this session. Jobs launched will
                inherit this name.
            schedulers: a list of schedulers the runner can use.
        """
        self._name: str = name
        self._scheduler_factories = scheduler_factories
        self._scheduler_params: Dict[str, Any] = {
            **(self._get_scheduler_params_from_env()),
            **(scheduler_params or {}),
        }
        # pyre-fixme[24]: SchedulerOpts is a generic, and we don't have access to the corresponding type
        self._scheduler_instances: Dict[str, Scheduler] = {}
        self._apps: Dict[AppHandle, AppDef] = {}

        # component_name -> map of component_fn_param_name -> user-specified default val encoded as str
        self._component_defaults: Dict[str, Dict[str, str]] = component_defaults or {}

    def _get_scheduler_params_from_env(self) -> Dict[str, str]:
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
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # This method returns False so that if an error is raise within the
        # ``with`` statement, it is reraised properly
        # see: https://docs.python.org/3/reference/compound_stmts.html#with
        # see also: torchx/runner/test/api_test.py#test_context_manager_with_error
        #
        self.close()
        return False

    def close(self) -> None:
        """
        Closes this runner and frees/cleans up any allocated resources.
        Transitively calls the ``close()`` method on all the schedulers.
        Once this method is called on the runner, the runner object is deemed
        invalid and any methods called on the runner object as well as
        the schedulers associated with this runner have undefined behavior.
        It is ok to call this method multiple times on the same runner object.
        """

        for scheduler in self._scheduler_instances.values():
            scheduler.close()

    def run_component(
        self,
        component: str,
        component_args: Union[list[str], dict[str, Any]],
        scheduler: str,
        cfg: Optional[Mapping[str, CfgVal]] = None,
        workspace: Optional[Union[Workspace, str]] = None,
        parent_run_id: Optional[str] = None,
    ) -> AppHandle:
        """
        Runs a component.

        ``component`` has the following resolution order(high to low):
            * User-registered components. Users can register components via
                https://packaging.python.org/specifications/entry-points/. Method looks for
                entrypoints in the group ``torchx.components``.
            * Builtin components relative to `torchx.components`. The path to the component should
                be module name relative to `torchx.components` and function name in a format:
                ``$module.$function``.
            * File-based components in format: ``$FILE_PATH:FUNCTION_NAME``. Both relative and
                absolute paths supported.

        Usage:

        .. code-block:: python

         # resolved to torchx.components.distributed.ddp()
         runner.run_component("distributed.ddp", ...)

         # resolved to my_component() function in ~/home/components.py
         runner.run_component("~/home/components.py:my_component", ...)


        Returns:
            An application handle that is used to call other action APIs on the app

        Raises:
            ComponentValidationException: if component is invalid.
            ComponentNotFoundException: if the ``component_path`` is failed to resolve.
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
        component_args: Union[list[str], dict[str, Any]],
        scheduler: str,
        cfg: Optional[Mapping[str, CfgVal]] = None,
        workspace: Optional[Union[Workspace, str]] = None,
        parent_run_id: Optional[str] = None,
    ) -> AppDryRunInfo:
        """
        Dryrun version of :py:func:`run_component`. Will not actually run the
        component, but just returns what "would" have run.
        """
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
        cfg: Optional[Mapping[str, CfgVal]] = None,
        workspace: Optional[Union[Workspace, str]] = None,
        parent_run_id: Optional[str] = None,
    ) -> AppHandle:
        """
        Runs the given application in the specified mode.

        .. note:: sub-classes of ``Runner`` should implement ``schedule`` method
                  rather than overriding this method directly.

        Returns:
            An application handle that is used to call other action APIs on the app.
        """

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
        """
        Actually runs the application from the given dryrun info.
        Useful when one needs to overwrite a parameter in the scheduler
        request that is not configurable from one of the object APIs.

        .. warning:: Use sparingly since abusing this method to overwrite
                     many parameters in the raw scheduler request may
                     lead to your usage of TorchX going out of compliance
                     in the long term. This method is intended to
                     unblock the user from experimenting with certain
                     scheduler-specific features in the short term without
                     having to wait until TorchX exposes scheduler features
                     in its APIs.

        .. note:: It is recommended that sub-classes of ``Session`` implement
                  this method instead of directly implementing the ``run`` method.

        Usage:

        ::

         dryrun_info = session.dryrun(app, scheduler="default", cfg)

         # overwrite parameter "foo" to "bar"
         dryrun_info.request.foo = "bar"

         app_handle = session.submit(dryrun_info)

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
        cfg: Optional[Mapping[str, CfgVal]] = None,
        workspace: Optional[Union[Workspace, str]] = None,
        parent_run_id: Optional[str] = None,
    ) -> AppDryRunInfo:
        """
        Dry runs an app on the given scheduler with the provided run configs.
        Does not actually submit the app but rather returns what would have been
        submitted. The returned ``AppDryRunInfo`` is pretty formatted and can
        be printed or logged directly.

        Usage:

        ::

         dryrun_info = session.dryrun(app, scheduler="local", cfg)
         print(dryrun_info)

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
        """
        Returns the ``runopts`` for the supported scheduler backends.

        Usage:

        ::

         local_runopts = session.scheduler_run_opts("local_cwd")
         print("local scheduler run options: {local_runopts}")

        Returns:
            The ``runopts`` for the specified scheduler type.
        """
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

    def scheduler_backends(self) -> List[str]:
        """
        Returns a list of all supported scheduler backends.
        """
        return list(self._scheduler_factories.keys())

    def status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        """
        Returns:
            The status of the application, or ``None`` if the app does not exist anymore
            (e.g. was stopped in the past and removed from the scheduler's backend).
        """
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
    ) -> Optional[AppStatus]:
        """
        Block waits (indefinitely) for the application to complete.
        Possible implementation:

        ::

         while(True):
             app_status = status(app)
             if app_status.is_terminal():
                 return
             sleep(10)

        Args:
            app_handle: the app handle to wait for completion
            wait_interval: the minimum interval to wait before polling for status

        Returns:
            The terminal status of the application, or ``None`` if the app does not exist anymore
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
        """
        Stops the application, effectively directing the scheduler to cancel
        the job. Does nothing if the app does not exist.

        .. note:: This method returns as soon as the cancel request has been
                  submitted to the scheduler. The application will be in a
                  ``RUNNING`` state until the scheduler actually terminates
                  the job. If the scheduler successfully interrupts the job
                  and terminates it the final state will be ``CANCELLED``
                  otherwise it will be ``FAILED``.

        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(app_handle)
        with log_event("cancel", scheduler_backend, app_id):
            status = self.status(app_handle)
            if status is not None and not status.is_terminal():
                scheduler.cancel(app_id)

    def stop(self, app_handle: AppHandle) -> None:
        """
        See method ``cancel``.

        .. warning:: This method will be deprecated in the future. It has been
                    replaced with ``cancel`` which provides the same functionality.
                    The change is to be consistent with the CLI and scheduler API.
        """
        warnings.warn(
            "This method will be deprecated in the future, please use `cancel` instead.",
            PendingDeprecationWarning,
        )
        self.cancel(app_handle)

    def describe(self, app_handle: AppHandle) -> Optional[AppDef]:
        """
        Reconstructs the application (to the best extent) given the app handle.
        Note that the reconstructed application may not be the complete app as
        it was submitted via the run API. How much of the app can be reconstructed
        is scheduler dependent.

        Returns:
            AppDef or None if the app does not exist anymore or if the
            scheduler does not support describing the app handle
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
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        """
        Returns an iterator over the log lines of the specified job container.

        .. note:: #. ``k`` is the node (host) id NOT the ``rank``.
                  #. ``since`` and ``until`` need not always be honored (depends on scheduler).

        .. warning:: The semantics and guarantees of the returned iterator is highly
                     scheduler dependent. See ``torchx.specs.api.Scheduler.log_iter``
                     for the high-level semantics of this log iterator. For this reason
                     it is HIGHLY DISCOURAGED to use this method for generating output
                     to pass to downstream functions/dependencies. This method
                     DOES NOT guarantee that 100% of the log lines are returned.
                     It is totally valid for this method to return no or partial log lines
                     if the scheduler has already totally or partially purged log records
                     for the application.

        Return lines will include whitespace characters such as ``\\n`` or
        ``\\r``. When outputting the lines you should make sure to avoid adding
        extra newline characters.

        Usage:

        .. code:: python

            app_handle = session.run(app, scheduler="local", cfg=Dict[str, ConfigValue]())

            print("== trainer node 0 logs ==")
            for line in session.log_lines(app_handle, "trainer", k=0):
               # for prints newlines will already be present in the line
               print(line, end="")

               # when writing to a file nothing extra is necessary
               f.write(line)

        Discouraged anti-pattern:

        .. code:: python

            # DO NOT DO THIS!
            # parses accuracy metric from log and reports it for this experiment run
            accuracy = -1
            for line in session.log_lines(app_handle, "trainer", k=0):
               if matches_regex(line, "final model_accuracy:[0-9]*"):
                   accuracy = parse_accuracy(line)
                   break
            report(experiment_name, accuracy)

        Args:
            app_handle: application handle
            role_name: role within the app (e.g. trainer)
            k: k-th replica of the role to fetch the logs for
            regex: optional regex filter, returns all lines if left empty
            since: datetime based start cursor. If left empty begins from the
                    first log line (start of job).
            until: datetime based end cursor. If left empty, follows the log output
                    until the job completes and all log lines have been consumed.

        Returns:
             An iterator over the role k-th replica of the specified application.

        Raise:
            UnknownAppException: if the app does not exist in the scheduler

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
    ) -> List[ListAppResponse]:
        """
        For apps launched on the scheduler, this API returns a list of ListAppResponse
        objects each of which have app id, app handle and its status.
        Note: This API is in prototype phase and is subject to change.
        """
        with log_event("list", scheduler):
            sched = self._scheduler(scheduler)
            apps = sched.list()
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
    ) -> Tuple[Scheduler, str, str]:
        """
        Returns the scheduler and app_id from the app_handle.
        Set ``check_session`` to validate that the session name in the app handle
        is the same as this session.

        Raises:
            ValueError: if ``check_session=True`` and the session in the app handle
                         does not match this session's name
            KeyError: if no such scheduler backend exists
        """

        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        scheduler = self._scheduler(scheduler_backend)
        return scheduler, scheduler_backend, app_id

    def __repr__(self) -> str:
        return f"Runner(name={self._name}, schedulers={self._scheduler_factories}, apps={self._apps})"


def get_runner(
    name: Optional[str] = None,
    component_defaults: Optional[Dict[str, Dict[str, str]]] = None,
    **scheduler_params: Any,
) -> Runner:
    """
    Convenience method to construct and get a Runner object. Usage:

    .. code-block:: python

      with get_runner() as runner:
        app_handle = runner.run(component(args), scheduler="kubernetes", runcfg)
        print(runner.status(app_handle))

    Alternatively,

    .. code-block:: python

     runner = get_runner()
     try:
        app_handle = runner.run(component(args), scheduler="kubernetes", runcfg)
        print(runner.status(app_handle))
     finally:
        runner.close()

    Args:
        name: human readable name that will be included as part of all launched
            jobs.
        scheduler_params: extra arguments that will be passed to the constructor
            of all available schedulers.


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
