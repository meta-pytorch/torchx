# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This contains the TorchX AppDef and related component definitions. These are
used by components to define the apps which can then be launched via a TorchX
scheduler or pipeline adapter.
"""
import difflib

import os
from typing import Callable, Dict, Iterator, Mapping, Optional

from torchx.specs.api import (
    ALL,
    AppDef,
    AppDryRunInfo,
    AppHandle,
    AppState,
    AppStatus,
    BindMount,
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


def _load_named_resources() -> Dict[str, Callable[[], Resource]]:
    resource_methods = load_group("torchx.named_resources", default={})
    materialized_resources: Dict[str, Callable[[], Resource]] = {}

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


_named_resource_factories: Dict[str, Callable[[], Resource]] = _load_named_resources()


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
    cpu: Optional[int] = None,
    gpu: Optional[int] = None,
    memMB: Optional[int] = None,
    h: Optional[str] = None,
) -> Resource:
    """
    Convenience method to create a ``Resource`` object from either the
    raw resource specs (cpu, gpu, memMB) or the registered named resource (``h``).
    Note that the (cpu, gpu, memMB) is mutually exclusive with ``h``
    taking predecence if specified.

    If ``h`` is specified then it is used to look up the
    resource specs from the list of registered named resources.
    See `registering named resource <https://meta-pytorch.org/torchx/latest/advanced.html#registering-named-resources>`_.

    Otherwise a ``Resource`` object is created from the raw resource specs.

    Example:

    .. code-block:: python

         resource(cpu=1) # returns Resource(cpu=1)
         resource(named_resource="foobar") # returns registered named resource "foo"
         resource(cpu=1, named_resource="foobar") # returns registered named resource "foo" (cpu=1 ignored)
         resource() # returns default resource values
         resource(cpu=None, gpu=None, memMB=None) # throws
    """

    if h:
        return get_named_resources(h)
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
    Get resource object based on the string definition registered via entrypoints.txt.

    TorchX implements ``named_resource`` registration mechanism, which consists of
    the following steps:

    1. Create a module and define your resource retrieval function:

    .. code-block:: python

     # my_module.resources
     from typing import Dict
     from torchx.specs import Resource

     def gpu_x_1() -> Dict[str, Resource]:
         return Resource(cpu=2, memMB=64 * 1024, gpu = 2)

    2. Register resource retrieval in the entrypoints section:

    ::

     [torchx.named_resources]
     gpu_x_1 = my_module.resources:gpu_x_1

    The ``gpu_x_1`` can be used as string argument to this function:

    ::

     from torchx.specs import named_resources
     resource = named_resources["gpu_x_1"]

    """
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
    "ReplicaState",
    "ReplicaStatus",
    "Resource",
    "RetryPolicy",
    "Role",
    "RoleStatus",
    "runopt",
    "runopts",
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
    "torchx_run_args_from_argparse",
    "torchx_run_args_from_json",
    "TorchXRunArgs",
    "ALL",
    "TORCHX_HOME",
    "Workspace",
]
