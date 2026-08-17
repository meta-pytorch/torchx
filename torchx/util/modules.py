# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
import logging
from types import ModuleType
from typing import Callable, TypeVar

logger: logging.Logger = logging.getLogger(__name__)


def load_module(path: str) -> ModuleType | Callable[..., object] | None:
    """
    Loads and returns the module/module attr represented by the ``path``: ``full.module.path:optional_attr``

    1. ``load_module("this.is.a_module:fn")`` -> equivalent to ``this.is.a_module.fn``
    1. ``load_module("this.is.a_module")`` -> equivalent to ``this.is.a_module``

    Always returns ``None`` on failure, but only the absence of the module
    (or one of its parent packages) is silent — a module that exists but is
    broken (e.g. fails importing one of its own dependencies) logs a warning
    naming the failure.
    """
    parts = path.split(":", 1)
    module_path, method = parts[0], parts[1] if len(parts) > 1 else None
    module = None
    i, n = -1, len(module_path)
    try:
        while i < n:
            i = module_path.find(".", i + 1)
            i = i if i >= 0 else n
            module = importlib.import_module(module_path[:i])
        return getattr(module, method) if method else module
    except Exception as e:
        missing = e.name if isinstance(e, ModuleNotFoundError) else None
        if missing is not None and (
            missing == module_path or module_path.startswith(missing + ".")
        ):
            # the requested module (or a parent package of it) is absent —
            # the expected miss for optional modules; stay silent
            return None
        # `module_path` exists but is broken (or the attr lookup failed)
        logger.warning("failed to load module `%s`: %s", path, e)
        return None


T = TypeVar("T")


def import_attr(name: str, attr: str, default: T) -> T:
    """
    Imports ``name.attr`` and returns it if the module is found.
    Otherwise, returns the specified ``default``.
    Useful when getting an attribute from an optional dependency.

    Note that the ``default`` parameter is intentionally not an optional
    since this function is intended to be used with modules that may not be
    installed as a dependency. Therefore the caller must ALWAYS provide a
    sensible default.

    Usage:

    .. code-block:: python

        aws_resources = import_attr("torchx.specs.named_resources_aws", "NAMED_RESOURCES", default={})
        all_resources.update(aws_resources)

    Raises:
        ModuleNotFoundError: If the module ``name`` exists but is broken —
            i.e. it failed importing one of its own dependencies. Only the
            absence of ``name`` itself (or one of its parent packages)
            returns ``default``.
        AttributeError: If the module exists (e.g. can be imported)
            but does not have an attribute with name ``attr``.
    """
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as e:
        missing = e.name
        if missing is not None and (missing == name or name.startswith(missing + ".")):
            # the requested module (or a parent package of it) is absent
            return default
        # `name` exists but is broken: it failed importing `missing`
        raise
    else:
        return getattr(mod, attr)
