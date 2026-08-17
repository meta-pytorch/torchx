# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from importlib import metadata
from importlib.metadata import EntryPoint
from typing import Callable, cast, overload, TypeVar

T = TypeVar("T")

# sentinel distinguishing "no default given" from an explicit `default=None`
_NO_DEFAULT: object = object()


@overload
def load(group: str, name: str) -> object: ...


@overload
def load(group: str, name: str, default: T) -> T: ...


def load(group: str, name: str, default: T | object = _NO_DEFAULT) -> T | object:
    """
    Loads the entry point specified by

    ::

     [group]
     name1 = this.is:a_function
     -- or --
     name2 = this.is.a.module

    In case such an entry point is not found, an optional
    default is returned. If the default is not specified
    and the entry point is not found, then this method
    raises an error.
    """
    entrypoints = metadata.entry_points().select(group=group)

    if name not in entrypoints.names:
        if default is not _NO_DEFAULT:
            return default
        raise KeyError(f"entrypoint {group}.{name} not found")

    # the caller declares the expected type via `default`
    return cast(T, entrypoints[name].load())


def _defer_load_ep(ep: EntryPoint) -> Callable[..., object]:
    def run(*args: object, **kwargs: object) -> object:
        if ep.attr is None:  # this is a module
            return ep.load()
        else:
            return ep.load()(*args, **kwargs)

    return run


def load_group(group: str) -> dict[str, Callable[..., object]]:
    """
    Loads all the entry points specified by ``group`` and returns
    the entry points as a map of ``name (str) -> deferred_load_fn``.
    where the ``deferred_load_fn`` (as the name implies) defers the
    loading of the entrypoint (e.g. ``entrypoint.load()``) until the
    caller explicitly executes the funtion.

    For the following ``entry_point.txt``:

    ::

     [foo]
     bar = this.is:a_fn
     baz = this.is:b_fn

    1. ``load_group("foo")["bar"]("baz")`` -> equivalent to calling ``this.is.a_fn("baz")``
    1. ``load_group("food")`` -> ``{}``


    If the entrypoint is a module (versus a function as shown above), then calling the ``deferred_load_fn``
    simply loads the module and ignores any ``*args`` or ``**kwargs`` passed. For example:

    ::

     [foo]
     bar = this.is.a.module

    1. ``load_group("foo")["bar"]()`` -> loads ``this.is.a.module`` and returns a ``module`` type
    1. ``load_group("foo")["bar"]("baz", hello="world")`` -> same as above (ignores ``*args`` and ``**kwargs``)

    """
    return {
        ep.name: _defer_load_ep(ep)
        for ep in metadata.entry_points().select(group=group)
    }
