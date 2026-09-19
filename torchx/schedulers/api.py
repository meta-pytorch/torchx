# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import abc
import inspect
import re
import types
from collections.abc import Iterator, Mapping
from dataclasses import MISSING, Field, dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import Self

from torchx.specs import (
    NONE,
    NULL_RESOURCE,
    AppDef,
    AppDryRunInfo,
    AppState,
    CfgVal,
    InvalidRunConfigException,
    Role,
    RoleStatus,
    Workspace,
    cases,
    runopts,
)
from torchx.workspace import WorkspaceMixin

DAYS_IN_2_WEEKS = 14


# =============================================================================
# STRUCTURED OPTIONS BASE CLASS
# =============================================================================


def _unwrap_optional(tp: type) -> type:
    """Strip ``None`` from union types (e.g. ``str | None`` -> ``str``)."""
    args = [a for a in get_args(tp) if a is not types.NoneType]
    if args and len(args) < len(get_args(tp)):
        # pyrefly: ignore [not-a-type]
        return args[0] if len(args) == 1 else Union[tuple(args)]
    return tp


def _is_structured_opts(tp: type) -> bool:
    """Return True if *tp* is a concrete ``StructuredOpts`` subclass."""
    try:
        return (
            isinstance(tp, type)
            and issubclass(tp, StructuredOpts)
            and tp is not StructuredOpts
        )
    except TypeError:
        # Generic aliases like list[str] or dict[str, str] can pass
        # isinstance(tp, type) on some Python versions but fail issubclass().
        return False


def _cfg_fields(cls_or_instance: type[StructuredOpts] | StructuredOpts) -> list[Field]:
    """Dataclass fields that are run options; underscored ones are private."""
    return [f for f in fields(cls_or_instance) if not f.name.startswith("_")]


@dataclass
class StructuredOpts(Mapping[str, CfgVal]):
    """Base class for typed scheduler configuration options.

    Provides a type-safe way to define scheduler run options as dataclass fields
    instead of manually building :py:class:`~torchx.specs.runopts`. Subclasses
    should be ``@dataclass`` decorated with fields representing config options.

    Features:
        - Auto-generates ``runopts`` from dataclass fields via :py:meth:`as_runopts`
        - Parses raw config dicts into typed instances via :py:meth:`from_cfg`
        - Supports snake_case field names with camelCase aliases
        - Field metadata ``cfg_key`` overrides the external config key for
          names that cannot be field names (e.g. hyphenated ``mail-user``):
          ``mail_user: str | None = field(default=None, metadata={"cfg_key": "mail-user"})``
        - Extracts help text from field docstrings
        - Supports nested ``StructuredOpts`` fields, flattened with dot-prefixed
          keys (e.g., ``k8s.context``)

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

    _extra_cfg: dict[str, CfgVal] = field(
        default_factory=dict, repr=False, compare=False, kw_only=True
    )
    """Options :py:meth:`from_cfg` was given that this dataclass does not
    declare, kept so the round trip is lossless. A scheduler's resolved config
    also carries the options of any mixin it inherits (e.g.
    :py:meth:`~torchx.workspace.api.WorkspaceMixin.workspace_opts`), and those
    stay reachable through the ``Mapping`` interface but are not fields."""

    @classmethod
    # pyrefly: ignore [not-a-type]
    def from_cfg(cls, cfg: Mapping[str, CfgVal]) -> Self:
        """Create an instance from a raw config dict.

        Fields are snake_case but also accept camelCase aliases (e.g.,
        ``hpc_identity`` can be set via ``hpcIdentity``). A field passed under
        two spellings with conflicting values raises
        :py:class:`~torchx.specs.InvalidRunConfigException`; equal values
        collapse into one.
        Nested :py:class:`StructuredOpts` fields are reconstructed from
        dot-prefixed keys (e.g., ``k8s.context``).
        """
        type_hints = get_type_hints(cls)
        kwargs = {}
        for f in _cfg_fields(cls):
            name = f.name
            field_type = _unwrap_optional(type_hints.get(name, str))

            if _is_structured_opts(field_type):
                prefix = f"{name}."
                nested_cfg = {
                    k[len(prefix) :]: v for k, v in cfg.items() if k.startswith(prefix)
                }
                if nested_cfg:
                    kwargs[name] = field_type.from_cfg(nested_cfg)
                elif f.default is MISSING and f.default_factory is MISSING:
                    # Required nested group — construct so its own validation runs.
                    kwargs[name] = field_type.from_cfg({})
                continue

            cfg_key = f.metadata.get("cfg_key", name)
            spellings = [
                k
                for k in dict.fromkeys((cfg_key, name, cases.snake_to_camel(name)))
                if k in cfg
            ]
            if not spellings:
                continue
            val = cfg[spellings[0]]
            for other in spellings[1:]:
                if cfg[other] != val:
                    raise InvalidRunConfigException(
                        f"Run option `{cfg_key}` was passed under two spellings"
                        f" (`{spellings[0]}` and `{other}`) with conflicting"
                        f" values. Pass it once, as `{cfg_key}`",
                        cfg_key,
                        cfg,
                    )
            kwargs[name] = val
        opts = cls(**kwargs)
        extra = {k: v for k, v in cfg.items() if k not in opts}
        if extra:
            opts._extra_cfg = extra
        return opts

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

    def __getitem__(self, key: str) -> CfgVal:
        if "." in key:
            prefix, rest = key.split(".", 1)
            prefix = cases.camel_to_snake(prefix)
            nested = getattr(self, prefix, None)
            if isinstance(nested, StructuredOpts):
                return nested[rest]
            raise KeyError(key) from None
        snake_key = cases.camel_to_snake(key)
        # only dataclass fields are cfg keys; a plain hasattr() check would
        # also resolve methods (e.g. opts["get"] -> bound method)
        if snake_key in {f.name for f in _cfg_fields(self)}:
            return getattr(self, snake_key)
        # pyrefly: ignore [bad-argument-type]
        for f in _cfg_fields(self):
            if f.metadata.get("cfg_key") == key:
                return getattr(self, f.name)
        if key in self._extra_cfg:
            return self._extra_cfg[key]
        raise KeyError(key) from None

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[str]:
        type_hints = get_type_hints(type(self))
        for f in _cfg_fields(self):
            field_type = _unwrap_optional(type_hints.get(f.name, str))
            if _is_structured_opts(field_type):
                nested = getattr(self, f.name)
                if nested is not None:
                    for nested_key in nested:
                        yield f"{f.name}.{nested_key}"
            else:
                yield f.metadata.get("cfg_key", f.name)
        yield from self._extra_cfg

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

        # Match: field_name: type...\n    """docstring"""
        # (non-greedy body so docstrings may contain single/double quotes)
        pattern = re.compile(
            r'^\s+(\w+):\s*[^\n]+\n\s+"""(.+?)"""',
            re.MULTILINE | re.DOTALL,
        )
        for match in pattern.finditer(source):
            field_name = match.group(1)
            docstring = match.group(2).strip()
            docstrings[field_name] = docstring

        type_hints = get_type_hints(cls)
        for f in _cfg_fields(cls):
            field_type = _unwrap_optional(type_hints.get(f.name, str))
            if _is_structured_opts(field_type):
                for key, doc in field_type.get_docstrings().items():
                    docstrings[f"{f.name}.{key}"] = doc

        return docstrings

    @classmethod
    def as_runopts(cls) -> runopts:
        """Build :py:class:`~torchx.specs.runopts` from dataclass fields.

        Nested :py:class:`StructuredOpts` fields are flattened with
        dot-prefixed keys (e.g., field ``k8s: K8sOpts`` with sub-field
        ``context`` becomes ``k8s.context``).
        """
        opts = runopts()

        type_hints = get_type_hints(cls)
        docstrings = cls.get_docstrings()

        for f in _cfg_fields(cls):
            name = f.name
            field_type = _unwrap_optional(type_hints.get(name, str))

            if _is_structured_opts(field_type):
                nested_opts = field_type.as_runopts()
                for nested_key, nested_runopt in nested_opts:
                    opts.add(
                        f"{name}.{nested_key}",
                        type_=nested_runopt.opt_type,
                        default=nested_runopt.default,
                        required=nested_runopt.is_required,
                        help=nested_runopt.help,
                    )
                continue

            help_text = docstrings.get(name, name)
            type_ = field_type

            has_default = f.default is not MISSING
            has_default_factory = f.default_factory is not MISSING
            if has_default:
                default = f.default
            elif has_default_factory:
                default = None  # Don't call factory, just indicate no default
            else:
                default = None

            required = not has_default and not has_default_factory

            opts.add(
                f.metadata.get("cfg_key", name),
                type_=type_,
                # pyrefly: ignore [bad-argument-type]
                default=default,
                required=required,
                help=help_text,
            )

        return opts

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


T = TypeVar("T", bound=Mapping[str, CfgVal])


class Scheduler(abc.ABC, Generic[T]):
    """Abstract base class for job schedulers.

    Implementors must override all ``@abc.abstractmethod`` methods.
    See :py:class:`StructuredOpts` for typed config and
    :py:mod:`torchx.schedulers` for built-in implementations.

    The type argument is the scheduler's config type. Name a
    :py:class:`StructuredOpts` subclass there and every ``cfg`` this scheduler
    is handed arrives as that dataclass, already converted.
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
        """Submits an app directly. Prefer :py:meth:`~torchx.runner.api.Runner.run` for production use.

        **Mutates** *app*: unlike :py:meth:`~torchx.runner.api.Runner.run`, no copy
        is taken, so every write lands on the caller's own object. Passing
        *workspace* sets ``app.roles[0].workspace`` and repoints
        ``app.roles[*].image`` at the build; without it nothing is built, and a
        role carrying its own :py:attr:`~torchx.specs.Role.workspace` reaches
        :py:meth:`_submit_dryrun` unbuilt.

        Raises:
            ValueError: *workspace* was passed but this scheduler is not a
                :py:class:`~torchx.workspace.WorkspaceMixin`.
        """
        resolved_cfg = self.run_opts().resolve(cfg)
        if workspace:
            if not isinstance(self, WorkspaceMixin):
                raise ValueError(
                    f"scheduler `{self.backend}` does not support workspaces"
                )

            if isinstance(workspace, str):
                workspace = Workspace.from_str(workspace)

            app.roles[0].workspace = workspace
            self.build_workspaces(app.roles, resolved_cfg)

        dryrun_info = self.submit_dryrun(app, resolved_cfg)
        return self.schedule(dryrun_info)

    @abc.abstractmethod
    def schedule(self, dryrun_info: AppDryRunInfo) -> str:
        """Submits a previously dry-run request. Returns the app_id."""
        raise NotImplementedError()

    @classmethod
    def _opts_type(cls) -> type[StructuredOpts] | None:
        """The :py:class:`StructuredOpts` subclass named in ``Scheduler[...]``, if any."""
        for klass in cls.__mro__:
            for base in getattr(klass, "__orig_bases__", ()):
                if get_origin(base) is Scheduler:
                    (arg,) = get_args(base)
                    return arg if _is_structured_opts(arg) else None
        return None

    def _resolve_cfg(self, cfg: T | Mapping[str, CfgVal]) -> T:
        """Applies :py:meth:`run_opts` defaults, then converts to the config type."""
        resolved = self.run_opts().resolve(cfg)
        opts_type = type(self)._opts_type()
        if opts_type is None:
            return cast(T, resolved)
        return cast(T, opts_type.from_cfg(resolved))

    def submit_dryrun(self, app: AppDef, cfg: T) -> AppDryRunInfo[Any, T]:
        """Returns the scheduler request without submitting.

        **No copy is taken**: the returned
        :py:attr:`~torchx.specs.AppDryRunInfo.app` *is* the *app* passed in, and
        :py:meth:`_submit_dryrun` and each role's ``pre_proc`` may mutate it.
        Callers wanting their :py:class:`~torchx.specs.AppDef` left alone should
        go through :py:meth:`~torchx.runner.api.Runner.dryrun`, which copies
        first.

        :py:attr:`~torchx.specs.AppDryRunInfo.cfg` is *cfg* with this
        scheduler's :py:meth:`run_opts` defaults applied and converted to the
        scheduler's config type, so it is not necessarily what was passed in.
        """
        resolved_cfg = self._resolve_cfg(cfg)
        dryrun_info = self._submit_dryrun(app, resolved_cfg)

        for role in app.roles:
            dryrun_info = role.pre_proc(self.backend, dryrun_info)

        dryrun_info.app = app
        dryrun_info.cfg = resolved_cfg
        return dryrun_info

    @abc.abstractmethod
    def _submit_dryrun(self, app: AppDef, cfg: T) -> AppDryRunInfo[Any, T]:
        """Renders *app* into this backend's submit request.

        Implementations receive *cfg* already resolved against
        :py:meth:`run_opts` and read ``role.image`` as final -- building a
        role's workspace is never their job. Reached through
        :py:meth:`~torchx.runner.api.Runner.dryrun`, or through
        :py:meth:`submit` with a *workspace*, that build has already happened
        and ``role.image`` points at its result; a caller who invokes
        :py:meth:`submit_dryrun` directly can still hand over roles whose
        ``workspace`` was never built.

        Mutating *app* is permitted (the caller owns no copy) but discouraged:
        prefer rendering into the returned
        :py:class:`~torchx.specs.AppDryRunInfo`. Do not set its ``app``/``cfg``
        attributes -- :py:meth:`submit_dryrun` does that.
        """
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

    def describe_native(self, app_id: str) -> Optional[AppDryRunInfo]:
        """Returns the scheduler-native request of an already-submitted app.

        The read-side twin of :py:meth:`submit_dryrun`: reads the request back
        from the scheduler, wrapped in the same
        :py:class:`~torchx.specs.AppDryRunInfo` (same scheduler-specific
        ``request`` type). Unlike :py:meth:`describe`, which lossily maps the
        job description onto :py:class:`~torchx.specs.AppDef`, the returned
        ``request`` preserves scheduler-specific fields with no ``AppDef``
        equivalent. The ``app``/``cfg`` back-references are populated only
        where the scheduler retains them.

        Returns ``None`` if this scheduler does not implement native read-back
        (the default) or if the app no longer exists.
        """
        return None

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

        On schedulers with persistent job definitions (e.g. Kubernetes),
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
                    f"No resource for role: {role.name} (image: {role.image})."
                    " Did you forget to attach resource to the role"
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
