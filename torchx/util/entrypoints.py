# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3, 2, 16]

from importlib import metadata
from importlib.metadata import EntryPoint
from typing import Any, Dict, Optional


def load(group: str, name: str, default=None):
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

    # [note_on_entrypoints]
    # return type of importlib.metadata.entry_points() is different between python-3.9 and python-3.10
    # https://docs.python.org/3.9/library/importlib.metadata.html#importlib.metadata.entry_points
    # https://docs.python.org/3.10/library/importlib.metadata.html#importlib.metadata.entry_points
    if hasattr(metadata.entry_points(), "select"):
        # python>=3.10
        entrypoints = metadata.entry_points().select(group=group)

        if name not in entrypoints.names and default is not None:
            return default

        ep = entrypoints[name]
        return ep.load()

    else:
        # python<3.10 (e.g. 3.9)
        # metadata.entry_points() returns dict[str, tuple[EntryPoint]] (not EntryPoints) in python-3.9
        entrypoints = metadata.entry_points().get(group, ())

        for ep in entrypoints:
            if ep.name == name:
                return ep.load()

        # [group].name not found
        if default is not None:
            return default
        else:
            raise KeyError(f"entrypoint {group}.{name} not found")


def _defer_load_ep(ep: EntryPoint) -> object:
    def run(*args: object, **kwargs: object) -> object:
        if ep.attr is None:  # this is a module
            return ep.load()
        else:
            return ep.load()(*args, **kwargs)

    return run


def load_group(group: str, default: Optional[Dict[str, Any]] = None):
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
    1. ``load_group("food")`` -> ``None``
    1. ``load_group("food", default={"hello": this.is.c_fn})["hello"]("world")`` -> equivalent to calling ``this.is.c_fn("world")``


    If the entrypoint is a module (versus a function as shown above), then calling the ``deferred_load_fn``
    simply loads the module and ignores any ``*args`` or ``**kwargs`` passed. For example:

    ::

     [foo]
     bar = this.is.a.module

    1. ``load_group("foo")["bar"]()`` -> loads ``this.is.a.module`` and returns a ``module`` type
    1. ``load_group("foo")["bar"]("baz", hello="world")`` -> same as above (ignores ``*args`` and ``**kwargs``)

    """

    # see [note_on_entrypoints] above
    if hasattr(metadata.entry_points(), "select"):
        # python>=3.10
        entrypoints = metadata.entry_points().select(group=group)
    else:
        # python<3.10 (e.g. 3.9)
        entrypoints = metadata.entry_points().get(group, ())

    if len(entrypoints) == 0:
        return default

    eps = {}
    for ep in entrypoints:
        eps[ep.name] = _defer_load_ep(ep)
    return eps
