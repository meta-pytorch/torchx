# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Core TorchX types for defining distributed applications.

The main types are :py:class:`AppDef`, :py:class:`Role`, and :py:class:`Resource`.
Components are functions that return an ``AppDef`` which can then be launched
via a :py:class:`~torchx.schedulers.api.Scheduler`.

.. doctest::

    >>> import torchx.specs as specs
    >>> app = specs.AppDef(
    ...     name="echo",
    ...     roles=[specs.Role(name="worker", image="/tmp", entrypoint="/bin/echo", args=["hello"])],
    ... )
    >>> app.name
    'echo'

"""

import difflib
import os
import threading
from typing import Callable, Iterator, KeysView, Mapping

from torchx import plugins
from torchx.specs.api import (  # noqa: F401
    ALL,
    AppDef,
    AppDryRunInfo,
    AppHandle,
    AppState,
    AppStatus,
    BindMount,
    cases,
    CfgVal,
    DeviceMount,
    get_type_name,
    InvalidRunConfigException,
    is_terminal,
    macros,
    MalformedAppHandleException,
    MISSING,
    NONE,
    NULL_RESOURCE,
    Package,
    PackageKind,
    parse_app_handle,
    ParsedAppHandle,
    ReplicaState,
    ReplicaStatus,
    Resource,
    RetryPolicy,
    Role,
    RoleStatus,
    runopt,
    runopts,
    TORCHX_HOME,
    UNKNOWN,
    UnknownAppException,
    UnknownSchedulerException,
    validate_packages,
    VolumeMount,
    Workspace,
)
from torchx.specs.builders import make_app_handle, materialize_appdef, parse_mounts
from torchx.specs.capabilities import CapabilityKey
from torchx.specs.metadata_keys import app_metadata, NA, TORCHX_CONTEXT_NAME
from torchx.util.modules import import_attr

GiB: int = 1024


ResourceFactory = Callable[[], Resource]


class _NamedResourcesLibrary:
    """Lazily-loaded named-resource lookup.

    ``import torchx.specs`` performs no resource-module import or plugin
    discovery — the AWS/generic/custom resource modules and the plugin
    registry are loaded on first lookup and cached.

    Discovery is single-flight: concurrent first lookups block until one
    scan populates the cache, and a re-entrant lookup (a plugin module
    looking up a named resource at import time, mid-scan) raises
    ``RuntimeError``, which the plugin scanner records as a load error for
    that module.
    """

    def __init__(self) -> None:
        self._factories: dict[str, ResourceFactory] | None = None
        self._lock = threading.RLock()
        self._loading = False

    def _load(self) -> dict[str, ResourceFactory]:
        factories = self._factories
        if factories is not None:
            return factories
        # double-checked: the lock serializes concurrent first lookups so
        # discovery runs exactly once; the RLock lets a re-entrant lookup on
        # the loading thread reach the sentinel check below instead of
        # deadlocking
        with self._lock:
            if self._factories is not None:
                return self._factories
            if self._loading:
                raise RuntimeError(
                    "re-entrant named-resource lookup: discovery is already in"
                    " progress on this thread. A plugin module is likely looking"
                    " up a named resource at import time; the registered set is"
                    " incomplete mid-scan, so defer the lookup into the factory"
                    " body."
                )
            self._loading = True
            try:
                aws: Mapping[str, ResourceFactory] = import_attr(
                    "torchx.specs.named_resources_aws", "NAMED_RESOURCES", default={}
                )
                generic: Mapping[str, ResourceFactory] = import_attr(
                    "torchx.specs.named_resources_generic",
                    "NAMED_RESOURCES",
                    default={},
                )
                try:
                    custom: Mapping[str, ResourceFactory] = import_attr(
                        os.environ.get(
                            "TORCHX_CUSTOM_NAMED_RESOURCES",
                            "torchx.specs.fb.named_resources",
                        ),
                        "NAMED_RESOURCES",
                        default={},
                    )
                except ModuleNotFoundError:
                    if "TORCHX_CUSTOM_NAMED_RESOURCES" in os.environ:
                        raise  # the user explicitly pointed at this module — surface its breakage
                    # the built-in fb default exists in a github checkout of fbsource but
                    # cannot resolve its fbcode-only deps there — treat as absent
                    custom = {}
                factories = {
                    **generic,
                    **aws,
                    **custom,
                    **plugins.registry().get(plugins.PluginType.NAMED_RESOURCE),
                }
                factories["NULL"] = lambda: NULL_RESOURCE
                factories["MISSING"] = lambda: NULL_RESOURCE
                self._factories = factories
            finally:
                self._loading = False
            return factories

    def reset(self) -> None:
        """Test hook: drop the cached factories so the next access re-discovers."""
        with self._lock:
            self._factories = None

    def __getitem__(self, key: str) -> Resource:
        factories = self._load()
        if key in factories:
            return factories[key]()
        else:
            matches = difflib.get_close_matches(
                key,
                factories.keys(),
                n=1,
            )
            if matches:
                msg = f"Did you mean `{matches[0]}`?"
            else:
                msg = f"Registered named resources: {list(factories.keys())}"

            raise KeyError(f"No named resource found for `{key}`. {msg}")

    def __contains__(self, key: str) -> bool:
        return key in self._load()

    def __iter__(self) -> Iterator[str]:
        """Iterates through the names of the registered named_resources.

        Usage:

        .. doctest::

            from torchx import specs

            for resource_name in specs.named_resources:
                resource = specs.resource(h=resource_name)
                assert isinstance(resource, specs.Resource)

        """
        yield from self._load()

    def keys(self) -> KeysView[str]:
        """The names of the registered named resources."""
        return self._load().keys()

    def items(self) -> Iterator[tuple[str, Resource]]:
        """Iterates ``(name, resource)`` pairs, materializing each resource.

        Usage:

        .. doctest::

            from torchx import specs

            for name, resource in specs.named_resources.items():
                assert isinstance(resource, specs.Resource)

        """
        for name, factory in self._load().items():
            yield name, factory()


named_resources: _NamedResourcesLibrary = _NamedResourcesLibrary()


def resource(
    cpu: int | None = None,
    gpu: int | None = None,
    memMB: int | None = None,
    h: str | None = None,
) -> Resource:
    """Creates a :py:class:`Resource` from raw specs or a named resource.

    When ``h`` is set, it takes precedence (raw specs are ignored). See
    :ref:`advanced:Registering Named Resources` for custom named resources.

    .. doctest::

        >>> from torchx.specs import resource
        >>> resource(cpu=4, gpu=1, memMB=8192)
        Resource(cpu=4, gpu=1, memMB=8192, capabilities={}, devices={}, tags={})

    """

    if h:
        return named_resources[h]
    else:
        # could make these defaults customizable via entrypoint
        # not doing that now since its not a requested feature and may just over complicate things
        # keeping these defaults method local so that no one else takes a dep on it
        DEFAULT_CPU = 2
        DEFAULT_GPU = 0
        DEFAULT_MEM_MB = 1024

        return Resource(
            cpu=cpu or DEFAULT_CPU,
            gpu=gpu or DEFAULT_GPU,
            memMB=memMB or DEFAULT_MEM_MB,
        )


def get_named_resources(res: str) -> Resource:
    """
    .. deprecated::
        Use :py:func:`resource(h=name) <resource>` instead.
    """
    import warnings

    warnings.warn(
        "`get_named_resources()` is deprecated, use `resource(h=name)` instead",
        FutureWarning,
        stacklevel=2,
    )
    return named_resources[res]


__all__ = [
    "app_metadata",
    "AppDef",
    "AppDryRunInfo",
    "AppHandle",
    "AppState",
    "AppStatus",
    "BindMount",
    "CapabilityKey",
    "CfgVal",
    "DeviceMount",
    "get_type_name",
    "is_terminal",
    "macros",
    "MISSING",
    "NA",
    "NONE",
    "NULL_RESOURCE",
    "Package",
    "PackageKind",
    "parse_app_handle",
    "ParsedAppHandle",
    "ReplicaState",
    "ReplicaStatus",
    "Resource",
    "RetryPolicy",
    "Role",
    "RoleStatus",
    "runopt",
    "runopts",
    "cases",
    "UNKNOWN",
    "UnknownAppException",
    "UnknownSchedulerException",
    "validate_packages",
    "InvalidRunConfigException",
    "MalformedAppHandleException",
    "VolumeMount",
    "resource",
    "get_named_resources",
    "named_resources",
    "make_app_handle",
    "materialize_appdef",
    "parse_mounts",
    "ALL",
    "TORCHX_CONTEXT_NAME",
    "TORCHX_HOME",
    "Workspace",
]
