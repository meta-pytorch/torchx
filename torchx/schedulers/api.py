# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Generic, Iterable, List, Optional, TypeVar, Union

from torchx.specs import (
    AppDef,
    AppState,
    NONE,
    NULL_RESOURCE,
    Role,
    RoleStatus,
    runopts,
    Workspace,
)
from torchx.workspace import WorkspaceMixin


DAYS_IN_2_WEEKS = 14


class Stream(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    COMBINED = "combined"


@dataclass
class DescribeAppResponse:
    """
    Response object returned by ``Scheduler.describe(app)`` API. Contains
    the status and description of the application as known by the scheduler.
    For some schedulers implementations this response object has necessary
    and sufficient information to recreate an ``AppDef`` object. For these types
    of schedulers, the user can re-``run()`` the recreated application. Otherwise
    the user can only call non-creating methods (e.g. ``wait()``, ``status()``,
    etc).

    Since this class is a data class and contains many member variables we
    keep the usage simple and provide a no-args constructor and chose to
    access the member vars directly rather than provide accessors.

    If scheduler returns arbitrary message, the ``msg`` field should be populated.
    If scheduler returns a structured json, the ``structured_error_msg`` field should be populated.
    """

    app_id: str = "<NOT_SET>"
    state: AppState = AppState.UNSUBMITTED
    num_restarts: int = -1
    msg: str = NONE
    structured_error_msg: str = NONE
    ui_url: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)

    roles_statuses: List[RoleStatus] = field(default_factory=list)
    roles: List[Role] = field(default_factory=list)


@dataclass
class ListAppResponse:
    """
    Response object returned by ``scheduler.list()`` and ``runner.list()`` APIs.
    Contains the app_id, app_handle and status of the application.
    App ID : The unique identifier that identifies apps submitted on the scheduler
    App handle: Identifier for apps run with torchx in a url format like
    {scheduler_backend}://{session_name}/{app_id}, which is created by the runner
    when it submits a job on a scheduler. Handle info in ListAppResponse is filled
    in by ``runner.list()``. This handle can be used to further describe the app
    with torchx CLI or a torchx runner instance.

    Since this class is a data class with some member variables we keep the usage
    simple and chose to access the member vars directly rather than provide accessors.
    """

    app_id: str
    state: AppState
    app_handle: str = "<NOT_SET>"
    name: str = ""

    # Implementing __hash__() makes ListAppResponse hashable which makes
    # it easier to check if a ListAppResponse object exists in a list of
    # objects for testing purposes.
    def __hash__(self) -> int:
        return hash((self.app_id, self.app_handle, self.state))


T = TypeVar("T")
A = TypeVar("A")
D = TypeVar("D")


class Scheduler(abc.ABC, Generic[T, A, D]):
    """
    An interface abstracting functionalities of a scheduler.
    Implementers need only implement those methods annotated with
    ``@abc.abstractmethod``.
    """

    def __init__(self, backend: str, session_name: str) -> None:
        self.backend = backend
        self.session_name = session_name

    def close(self) -> None:
        """
        Only for schedulers that have local state! Closes the scheduler
        freeing any allocated resources. Once closed, the scheduler object
        is deemed to no longer be valid and any method called on the object
        results in undefined behavior.

        This method should not raise exceptions and is allowed to be called
        multiple times on the same object.

        .. note:: Override only for scheduler implementations that have local state
                  (``torchx/schedulers/local_scheduler.py``).
                  Schedulers simply wrapping a remote scheduler's client need not
                  implement this method.
        """
        pass

    def submit(
        self,
        app: A,
        cfg: T,
        workspace: str | Workspace | None = None,
    ) -> str:
        """
        Submits the application to be run by the scheduler.

        WARNING: Mostly used for tests. Users should prefer to use the TorchX runner instead.

        Returns:
            The application id that uniquely identifies the submitted app.
        """
        # pyre-fixme: Generic cfg type passed to resolve
        resolved_cfg = self.run_opts().resolve(cfg)
        if workspace:
            assert isinstance(self, WorkspaceMixin)

            if isinstance(workspace, str):
                workspace = Workspace.from_str(workspace)

            app.roles[0].workspace = workspace
            self.build_workspaces(app.roles, resolved_cfg)

        # pyre-fixme: submit_dryrun takes Generic type for resolved_cfg
        dryrun_info = self.submit_dryrun(app, resolved_cfg)
        return self.schedule(dryrun_info)

    @abc.abstractmethod
    def schedule(self, dryrun_info: D) -> str:
        """
        Same as ``submit`` except that it takes an ``AppDryRunInfo``.
        Implementers are encouraged to implement this method rather than
        directly implementing ``submit`` since ``submit`` can be trivially
        implemented by:

        ::

         dryrun_info = self.submit_dryrun(app, cfg)
         return schedule(dryrun_info)

        """

        raise NotImplementedError()

    def submit_dryrun(self, app: A, cfg: T) -> D:
        """
        Rather than submitting the request to run the app, returns the
        request object that would have been submitted to the underlying
        service. The type of the request object is scheduler dependent.
        This method can be used to dry-run an application. Please refer
        to the scheduler implementation's documentation regarding
        the actual return type.
        """
        # pyre-fixme: Generic cfg type passed to resolve
        resolved_cfg = self.run_opts().resolve(cfg)
        # pyre-fixme: _submit_dryrun takes Generic type for resolved_cfg
        dryrun_info = self._submit_dryrun(app, resolved_cfg)

        if isinstance(app, AppDef):
            for role in app.roles:
                dryrun_info = role.pre_proc(self.backend, dryrun_info)
        dryrun_info._app = app
        dryrun_info._cfg = resolved_cfg
        return dryrun_info

    @abc.abstractmethod
    def _submit_dryrun(self, app: A, cfg: T) -> D:
        raise NotImplementedError()

    def run_opts(self) -> runopts:
        """
        Returns the run configuration options expected by the scheduler.
        Basically a ``--help`` for the ``run`` API.
        """
        opts = self._run_opts()
        if isinstance(self, WorkspaceMixin):
            opts.update(self.workspace_opts())
        return opts

    def _run_opts(self) -> runopts:
        return runopts()

    @abc.abstractmethod
    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        """
        Describes the specified application.

        Returns:
            AppDef description or ``None`` if the app does not exist.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list(self) -> List[ListAppResponse]:
        """
        For apps launched on the scheduler, this API returns a list of ListAppResponse
        objects each of which have app id and its status.
        Note: This API is in prototype phase and is subject to change.
        """
        raise NotImplementedError()

    def exists(self, app_id: str) -> bool:
        """
        Returns:
            ``True`` if the app exists (was submitted), ``False`` otherwise
        """
        desc = self.describe(app_id)
        return desc is not None

    @abc.abstractmethod
    def _cancel_existing(self, app_id: str) -> None:
        """
        Kills the application. This method will only be called on an
        application that exists.
        """
        raise NotImplementedError()

    def cancel(self, app_id: str) -> None:
        """
        Cancels/kills the application. This method is idempotent within the same
        thread and is safe to call on the same application multiple times.
        However when called from multiple threads/processes on the same app
        the exact semantics of this method depends on the idempotency guarantees
        of the underlying scheduler API.

        .. note:: This method does not block for the application to reach a
                  cancelled state. To ensure that the application reaches a
                  terminal state use the ``wait`` API.
        """
        if self.exists(app_id):
            self._cancel_existing(app_id)
        else:
            # do nothing if the app does not exist
            return

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        """
        Returns an iterator to the log lines of the ``k``th replica of the ``role``.
        The iterator ends when all qualifying log lines have been read.

        If the scheduler supports time-based cursors fetching log lines
        for custom time ranges, then the ``since``, ``until`` fields are
        honored, otherwise they are ignored. Not specifying ``since`` and ``until``
        is equivalent to getting all available log lines. If the ``until`` is
        empty, then the iterator behaves like ``tail -f``, following the log output
        until the job reaches a terminal state.

        The exact definition of what constitutes a log is scheduler specific. Some
        schedulers may consider stderr or stdout as the log, others may read the logs
        from a log file.

        Behaviors and assumptions:

        1. Produces an undefined-behavior if called on an app that does not exist
           The caller should check that the app exists using ``exists(app_id)``
           prior to calling this method.

        2. Is not stateful, calling this method twice with same parameters
           returns a new iterator. Prior iteration
           progress is lost.

        3. Does not always support log-tailing. Not all schedulers support live
           log iteration (e.g. tailing logs while the app is running). Refer to
           the specific scheduler's documentation for the iterator's behavior.

        3.1 If the scheduler supports log-tailing, it should be controlled
            by ``should_tail`` parameter.

        4. Does not guarantee log retention. It is possible that by the time this
           method is called, the underlying scheduler may have purged the log records
           for this application. If so this method raises an arbitrary exception.

        5. If ``should_tail`` is True, the method only raises a ``StopIteration`` exception
           when the accessible log lines have been fully exhausted and the app has reached
           a final state. For instance, if the app gets stuck and does not produce any log lines,
           then the iterator blocks until the app eventually gets killed (either via
           timeout or manually) at which point it raises a ``StopIteration``.

           If ``should_tail`` is False, the method raises ``StopIteration``
           when there are no more logs.

        6. Need not be supported by all schedulers.

        7. Some schedulers may support line cursors by supporting ``__getitem__``
           (e.g. ``iter[50]`` seeks to the 50th log line).

        8. Whitespace is preserved, each new line should include ``\\n``. To
            support interactive progress bars the returned lines don't need to
            include ``\\n`` but should then be printed without a newline to
            correctly handle ``\\r`` carriage returns.

        Args:
            streams: The IO output streams to select.
                One of: combined, stdout, stderr.
                If the selected stream isn't supported by the scheduler it will
                throw an ValueError.

        Returns:
            An ``Iterator`` over log lines of the specified role replica

        Raises:
            NotImplementedError: if the scheduler does not support log iteration
        """
        raise NotImplementedError(
            f"{self.__class__.__qualname__} does not support application log iteration"
        )

    def _pre_build_validate(self, app: AppDef, scheduler: str, cfg: T) -> None:
        """
        validates before workspace build whether application is consistent with the scheduler.

        Raises error if application is not compatible with scheduler
        """
        pass

    def _validate(self, app: A, scheduler: str, cfg: T) -> None:
        """
        Validates after workspace build whether application is consistent with the scheduler.

        Raises error if application is not compatible with scheduler
        """
        if not isinstance(app, AppDef):
            return

        for role in app.roles:
            if role.resource == NULL_RESOURCE:
                raise ValueError(
                    f"No resource for role: {role.image}. Did you forget to attach resource to the role"
                )


def filter_regex(regex: str, data: Iterable[str]) -> Iterable[str]:
    """
    filter_regex takes a string iterator and returns an iterator that only has
    values that match the regex.
    """

    r = re.compile(regex)
    return filter(lambda datum: r.search(datum), data)


def split_lines(text: str) -> List[str]:
    """
    split_lines splits the string by new lines and keeps the new line characters.
    """
    lines = []
    while len(text) > 0:
        idx = text.find("\n")
        if idx >= 0:
            lines.append(text[: idx + 1])
            text = text[idx + 1 :]
        else:
            lines.append(text)
            break
    return lines


def split_lines_iterator(chunks: Iterable[str]) -> Iterable[str]:
    """
    split_lines_iterator splits each chunk in the iterator by new lines and
    returns them.
    """
    for chunk in chunks:
        lines = split_lines(chunk)
        for line in lines:
            yield line
