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
from typing import Callable, Iterator, Mapping

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
    UnknownAppException,
    UnknownSchedulerException,
    VolumeMount,
    Workspace,
)
from torchx.specs.builders import make_app_handle, materialize_appdef, parse_mounts
from torchx.util.entrypoints import load_group
from torchx.util.modules import import_attr

GiB: int = 1024


ResourceFactory = Callable[[], Resource]

AWS_NAMED_RESOURCES: Mapping[str, ResourceFactory] = import_attr(
    "torchx.specs.named_resources_aws", "NAMED_RESOURCES", default={}
)
GENERIC_NAMED_RESOURCES: Mapping[str, ResourceFactory] = import_attr(
    "torchx.specs.named_resources_generic", "NAMED_RESOURCES", default={}
)
CUSTOM_NAMED_RESOURCES: Mapping[str, ResourceFactory] = import_attr(
    os.environ.get("TORCHX_CUSTOM_NAMED_RESOURCES", "torchx.specs.fb.named_resources"),
    "NAMED_RESOURCES",
    default={},
)


def _load_named_resources() -> dict[str, Callable[[], Resource]]:
    resource_methods = load_group("torchx.named_resources", default={})
    materialized_resources: dict[str, Callable[[], Resource]] = {}

    for name, resource in {
        **GENERIC_NAMED_RESOURCES,
        **AWS_NAMED_RESOURCES,
        **CUSTOM_NAMED_RESOURCES,
        **resource_methods,
    }.items():
        materialized_resources[name] = resource

    materialized_resources["NULL"] = lambda: NULL_RESOURCE
    materialized_resources["MISSING"] = lambda: NULL_RESOURCE
    return materialized_resources


_named_resource_factories: dict[str, Callable[[], Resource]] = _load_named_resources()


class _NamedResourcesLibrary:
    def __getitem__(self, key: str) -> Resource:
        if key in _named_resource_factories:
            return _named_resource_factories[key]()
        else:
            matches = difflib.get_close_matches(
                key,
                _named_resource_factories.keys(),
                n=1,
            )
            if matches:
                msg = f"Did you mean `{matches[0]}`?"
            else:
                msg = f"Registered named resources: {list(_named_resource_factories.keys())}"

            raise KeyError(f"No named resource found for `{key}`. {msg}")

    def __contains__(self, key: str) -> bool:
        return key in _named_resource_factories

    def __iter__(self) -> Iterator[str]:
        """Iterates through the names of the registered named_resources.

        Usage:

        .. doctest::

            from torchx import specs

            for resource_name in specs.named_resources:
                resource = specs.resource(h=resource_name)
                assert isinstance(resource, specs.Resource)

        """
        for key in _named_resource_factories:
            yield (key)


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
    "AppDef",
    "AppDryRunInfo",
    "AppHandle",
    "AppState",
    "AppStatus",
    "BindMount",
    "CfgVal",
    "DeviceMount",
    "get_type_name",
    "is_terminal",
    "macros",
    "MISSING",
    "NONE",
    "NULL_RESOURCE",
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
    "UnknownAppException",
    "UnknownSchedulerException",
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
    "TORCHX_HOME",
    "Workspace",
]
