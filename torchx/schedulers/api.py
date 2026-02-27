# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import abc
import inspect
import re
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields, MISSING
from datetime import datetime
from enum import Enum
from typing import (
    Generic,
    get_args,
    get_origin,
    get_type_hints,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

from torchx.specs import (
    AppDef,
    AppDryRunInfo,
    AppState,
    cases,
    CfgVal,
    NONE,
    NULL_RESOURCE,
    Role,
    RoleStatus,
    runopts,
    Workspace,
)
from torchx.workspace import WorkspaceMixin
from typing_extensions import Self


DAYS_IN_2_WEEKS = 14


# =============================================================================
# STRUCTURED OPTIONS BASE CLASS
# =============================================================================


class StructuredOpts(Mapping[str, CfgVal]):
    """Base class for typed scheduler configuration options.

    Provides a type-safe way to define scheduler run options as dataclass fields
    instead of manually building :py:class:`~torchx.specs.runopts`. Subclasses
    should be ``@dataclass`` decorated with fields representing config options.

    Features:
        - Auto-generates ``runopts`` from dataclass fields via :py:meth:`as_runopts`
        - Parses raw config dicts into typed instances via :py:meth:`from_cfg`
        - Supports snake_case field names with camelCase aliases
        - Extracts help text from field docstrings

    Example:
        .. doctest::

            >>> from dataclasses import dataclass
            >>> from torchx.schedulers.api import StructuredOpts
            >>>
            >>> @dataclass
            ... class MyOpts(StructuredOpts):
            ...     cluster_name: str
            ...     '''Name of the cluster to submit to.'''
            ...
            ...     num_retries: int = 3
            ...     '''Number of retry attempts.'''
            ...
            >>> # Use in scheduler:
            >>> # def _run_opts(self) -> runopts:
            >>> #     return MyOpts.as_runopts()
            >>> #
            >>> # def _submit_dryrun(self, app, cfg):
            >>> #     opts = MyOpts.from_cfg(cfg)
            >>> #     # opts.cluster_name, opts.num_retries are typed

    """

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, CfgVal]) -> Self:
        """Create an instance from a raw config dict.

        Fields are snake_case but also accept camelCase aliases (e.g.,
        ``hpc_identity`` can be set via ``hpcIdentity``).
        """
        kwargs = {}
        for f in fields(cls):
            name = f.name
            # Check for snake_case key first, then camelCase alias
            if name in cfg:
                kwargs[name] = cfg[name]
            else:
                camel_case = cases.snake_to_camel(name)
                if camel_case in cfg:
                    kwargs[name] = cfg[camel_case]
        return cls(**kwargs)

    # -------------------------------------------------------------------------
    # Mapping Protocol Methods (for backwards compatibility)
    #
    # These methods allow StructuredOpts instances to be used in places that
    # expect a dict-like interface (e.g., plugins that do cfg.get("key") or
    # cfg["key"]). Once all plugins are migrated to use typed field access
    # (e.g., cfg.field_name), these methods can be removed.
    #
    # TODO(T252193642): Remove these methods after migrating plugins to use
    # StructuredOpts field access instead of dict-like access.
    # -------------------------------------------------------------------------

    # pyre-fixme[14]: Inconsistent override - Mapping.get accepts a default parameter
    def get(self, key: str) -> CfgVal:
        try:
            return self[key]
        except KeyError:
            return None

    def __getitem__(self, key: str) -> CfgVal:
        snake_key = cases.camel_to_snake(key)
        if hasattr(self, snake_key):
            return getattr(self, snake_key)
        raise KeyError(key) from None

    def __len__(self) -> int:
        return len(fields(self))

    def __iter__(self) -> Iterator[str]:
        for f in fields(self):
            yield f.name

    # pyre-fixme[14]: Inconsistent override - Mapping uses PyreReadOnly[object]
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self[key]
        except KeyError:
            return False
        return True

    @classmethod
    def get_docstrings(cls) -> dict[str, str]:
        # Parses source to extract attribute docstrings for help text.
        docstrings: dict[str, str] = {}
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            return docstrings

        # Pattern to match: field_name: type = value (optional) followed by docstring
        # Captures: field name, then the triple-quoted docstring on the next line
        pattern = re.compile(
            r'^\s+(\w+):\s*[^\n]+\n\s+"""([^"]+)"""',
            re.MULTILINE,
        )
        for match in pattern.finditer(source):
            field_name = match.group(1)
            docstring = match.group(2).strip()
            docstrings[field_name] = docstring

        return docstrings

    @classmethod
    def as_runopts(cls) -> runopts:
        """Build :py:class:`~torchx.specs.runopts` from dataclass fields."""
        opts = runopts()

        # Get resolved type hints (handles string annotations from __future__.annotations)
        type_hints = get_type_hints(cls)
        # Get field docstrings parsed from source code
        docstrings = cls.get_docstrings()

        for f in fields(cls):
            name = f.name

            # Get help text from field docstring
            help_text = docstrings.get(name, name)

            # Get type: extract base type from Union (e.g., int | None -> int)
            field_type = type_hints.get(name, str)
            origin = get_origin(field_type)
            # Handle Union types to extract the base type for runopts.
            # This logic handles field declarations like:
            #   * foo: str | None = None  (types.UnionType without __future__.annotations)
            #   * foo: Union[str, None] = None
            #   * foo: Optional[str] = None  (equivalent to Union[str, None])
            # Note: With `from __future__ import annotations`, get_type_hints() returns
            # typing.Union for all syntaxes. The UnionType check handles the case
            # without __future__.annotations where `str | None` creates types.UnionType.
            if origin is Union or isinstance(field_type, types.UnionType):
                args = [a for a in get_args(field_type) if a is not type(None)]
                field_type = args[0] if args else str
            type_ = field_type

            # Get default value
            has_default = f.default is not MISSING
            has_default_factory = f.default_factory is not MISSING
            if has_default:
                default = f.default
            elif has_default_factory:
                default = None  # Don't call factory, just indicate no default
            else:
                default = None

            # Determine if required (no default value)
            required = not has_default and not has_default_factory

            # Add the option
            opts.add(
                name,
                type_=type_,
                default=default,
                required=required,
                help=help_text,
            )

        return opts

    # pyre-fixme[15]: Inconsistent override - __or__ returns dict, not UnionType
    def __or__(self, other: StructuredOpts) -> dict[str, CfgVal]:
        """Merge two StructuredOpts instances into a cfg dict.

        Example:
            .. doctest::

                >>> from dataclasses import dataclass
                >>> from torchx.schedulers.api import StructuredOpts
                >>> @dataclass
                ... class OptsA(StructuredOpts):
                ...     foo: str = "a"
                >>> @dataclass
                ... class OptsB(StructuredOpts):
                ...     bar: int = 1
                >>> cfg = OptsA(foo="x") | OptsB(bar=2)
                >>> cfg["foo"], cfg["bar"]
                ('x', 2)
        """
        merged: dict[str, CfgVal] = {}
        for key in self:
            merged[key] = self[key]
        for key in other:
            merged[key] = other[key]
        return merged


# =============================================================================
# STREAM AND RESPONSE TYPES
# =============================================================================


class Stream(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"
    COMBINED = "combined"


@dataclass
class DescribeAppResponse:
    """Response from :py:meth:`Scheduler.describe`. Contains status, roles, and metadata."""

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
    """Response from :py:meth:`Scheduler.list` / :py:meth:`~torchx.runner.api.Runner.list`."""

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


class Scheduler(abc.ABC, Generic[T]):
    """Abstract base class for job schedulers.

    Implementors must override all ``@abc.abstractmethod`` methods.
    See :py:class:`StructuredOpts` for typed config and
    :py:mod:`torchx.schedulers` for built-in implementations.
    """

    def __init__(self, backend: str, session_name: str) -> None:
        self.backend = backend
        self.session_name = session_name

    def close(self) -> None:
        """Releases local resources. Safe to call multiple times.

        Only override for schedulers with local state (e.g. ``local_scheduler``).
        """
        pass

    def submit(
        self,
        app: AppDef,
        cfg: T,
        workspace: str | Workspace | None = None,
    ) -> str:
        """Submits an app directly. Prefer :py:meth:`~torchx.runner.api.Runner.run` for production use."""
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
    def schedule(self, dryrun_info: AppDryRunInfo) -> str:
        """Submits a previously dry-run request. Returns the app_id."""
        raise NotImplementedError()

    def submit_dryrun(self, app: AppDef, cfg: T) -> AppDryRunInfo:
        """Returns the scheduler request without submitting."""
        # pyre-fixme: Generic cfg type passed to resolve
        resolved_cfg = self.run_opts().resolve(cfg)
        # pyre-fixme: _submit_dryrun takes Generic type for resolved_cfg
        dryrun_info = self._submit_dryrun(app, resolved_cfg)

        for role in app.roles:
            dryrun_info = role.pre_proc(self.backend, dryrun_info)

        dryrun_info._app = app
        dryrun_info._cfg = resolved_cfg
        return dryrun_info

    @abc.abstractmethod
    def _submit_dryrun(self, app: AppDef, cfg: T) -> AppDryRunInfo:
        raise NotImplementedError()

    def run_opts(self) -> runopts:
        """Returns accepted run configuration options (``torchx runopts <scheduler>``)."""
        opts = self._run_opts()
        if isinstance(self, WorkspaceMixin):
            opts.update(self.workspace_opts())
        return opts

    def _run_opts(self) -> runopts:
        return runopts()

    @abc.abstractmethod
    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        """Returns app description, or ``None`` if it no longer exists."""
        raise NotImplementedError()

    @abc.abstractmethod
    def list(self, cfg: Mapping[str, CfgVal] | None = None) -> List[ListAppResponse]:
        """Lists jobs on this scheduler."""
        raise NotImplementedError()

    def exists(self, app_id: str) -> bool:
        desc = self.describe(app_id)
        return desc is not None

    @abc.abstractmethod
    def _cancel_existing(self, app_id: str) -> None:
        raise NotImplementedError()

    def cancel(self, app_id: str) -> None:
        """Cancels the app. Idempotent — safe to call multiple times.

        Does not block. Use :py:meth:`~torchx.runner.api.Runner.wait` to
        await the terminal state.
        """
        if self.exists(app_id):
            self._cancel_existing(app_id)
        else:
            # do nothing if the app does not exist
            return

    def delete(self, app_id: str) -> None:
        """Deletes the job definition from the scheduler's data-plane.

        On schedulers with persistent job definitions (e.g. Kubernetes, AWS Batch),
        this purges the definition. On others (e.g. Slurm), this is equivalent to
        :py:meth:`cancel`. Calling on a live job cancels it first.
        """
        if self.exists(app_id):
            self._delete_existing(app_id)

    def _delete_existing(self, app_id: str) -> None:
        self._cancel_existing(app_id)

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
        """Returns an iterator over log lines for the ``k``-th replica of ``role_name``.

        .. important:: Not all schedulers support log iteration, tailing, or
                       time-based cursors. Check the specific scheduler docs.

        Lines include trailing whitespace (``\\n``). When ``should_tail=True``,
        the iterator blocks until the app reaches a terminal state.

        Args:
            k: replica (node) index
            regex: optional filter pattern
            since: start cursor (scheduler-dependent)
            until: end cursor (scheduler-dependent)
            should_tail: if ``True``, follow output like ``tail -f``
            streams: ``stdout``, ``stderr``, or ``combined``

        Raises:
            NotImplementedError: if the scheduler does not support log iteration
        """
        raise NotImplementedError(
            f"{self.__class__.__qualname__} does not support application log iteration"
        )

    def _pre_build_validate(self, app: AppDef, scheduler: str, cfg: T) -> None:
        # Hook for pre-workspace-build validation. Override to add checks.
        pass

    def _validate(self, app: AppDef, scheduler: str, cfg: T) -> None:
        # Hook for post-workspace-build validation.
        for role in app.roles:
            if role.resource == NULL_RESOURCE:
                raise ValueError(
                    f"No resource for role: {role.image}. Did you forget to attach resource to the role"
                )


def filter_regex(regex: str, data: Iterable[str]) -> Iterable[str]:
    """Filters an iterable of strings, yielding only lines matching ``regex``."""

    r = re.compile(regex)
    return filter(lambda datum: r.search(datum), data)


def split_lines(text: str) -> List[str]:
    """Splits ``text`` by newlines, preserving the ``\\n`` characters."""
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
    """Splits each chunk in the iterable by newlines, yielding individual lines."""
    for chunk in chunks:
        lines = split_lines(chunk)
        for line in lines:
            yield line
