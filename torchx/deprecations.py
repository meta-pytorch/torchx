# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Platform-agnostic deprecation utilities for TorchX.

Provides :func:`deprecated_module` for warning about moved import
paths and :func:`deprecated` for marking functions/classes as deprecated.

Both emit :py:class:`UserWarning` (not ``DeprecationWarning``) so that
warnings are always visible to end users — ``DeprecationWarning`` is
silenced by default outside ``__main__``.

For Meta-internal use (fbsource-specific path resolution and sed commands),
see :py:mod:`torchx.fb.deprecations` which layers on top of this module.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Iterable


# Entry-point groups that have ``torchx_plugins.*`` namespace-package
# alternatives.  Only these groups trigger a deprecation warning.
# Keep in sync with ``torchx.plugins._registry.PluginType``.
_PLUGIN_GROUPS: frozenset[str] = frozenset(
    {
        "torchx.schedulers",
        "torchx.named_resources",
        "torchx.tracker",
    }
)


def deprecated_entrypoint(
    group: str,
    ep_names: Iterable[str],
    *,
    stacklevel: int = 2,
) -> None:
    """Emit a deprecation warning for entry-point based plugins.

    Only warns for groups that have ``torchx_plugins.*`` namespace-package
    equivalents (i.e., groups listed in
    :py:class:`~torchx.plugins.PluginType`).  Groups without namespace
    alternatives (e.g., ``"torchx.schedulers.orchestrator"``,
    ``"torchx.components"``) are silently ignored.

    Args:
        group: The entry-point group name (e.g., ``"torchx.schedulers"``).
        ep_names: Names of the entry-point plugins that were loaded.
        stacklevel: Stack level for :py:func:`warnings.warn`.  Default ``2``
            points at the caller of this function.

    Example::

        >>> # In _registry._find():
        >>> deprecated_entrypoint("torchx.schedulers", ["mast_conda"])

    """
    if group not in _PLUGIN_GROUPS:
        return

    names = ", ".join(sorted(ep_names))
    namespace = f"torchx_plugins.{group.removeprefix('torchx.')}"
    warnings.warn(
        f"Entry-point plugins in group '{group}' are deprecated. "
        f"Migrate to the '{namespace}' namespace package using "
        f"the @register decorator. "
        f"Set TORCHX_NO_ENTRYPOINTS=1 to opt out early. "
        f"Deprecated entry-point plugins: {names}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated_module(
    old_import: str,
    new_import: str,
    *,
    stacklevel: int = 3,
) -> None:
    """Emit a :py:class:`UserWarning` for a moved module import.

    Call this from backwards-compatibility stub modules so users see a
    warning on first import:

    .. code-block:: python

        # torchx/old/path.py (BC stub)
        from torchx.deprecations import deprecated_module

        deprecated_module(
            old_import="torchx.old.path",
            new_import="torchx.new.path",
        )

        from torchx.new.path import *  # noqa: F401,F403

    Args:
        old_import: The deprecated import path.
        new_import: The replacement import path.
        stacklevel: Stack level for the warning.  Default ``3`` works when
            called from module-level code in a stub
            (caller -> stub -> this function -> ``warnings.warn``).
    """
    warnings.warn(
        f"[Deprecated] {old_import} has moved to {new_import} and will be"
        f" removed in a future release. Update your import:"
        f"\n\n  - from {old_import} import ..."
        f"\n  + from {new_import} import ...\n",
        UserWarning,
        stacklevel=stacklevel,
    )


_F = Callable[..., object]


def deprecated(
    *,
    replacement: str | None = None,
) -> Callable[[_F], _F]:
    """Mark a function or class as deprecated.

    .. code-block:: python

        from torchx.deprecations import deprecated

        @deprecated(replacement="new_func")
        def old_func():
            ...

    Args:
        replacement: Name or import path of the replacement, if any.

    Returns:
        A decorator that wraps the target to emit :py:class:`UserWarning`
        on each call.
    """

    def decorator(fn: _F) -> _F:
        parts = [f"[Deprecated] {fn.__qualname__} is deprecated"]
        if replacement:
            parts.append(f"-- use {replacement} instead")
        msg: str = " ".join(parts) + "."

        @functools.wraps(fn)
        # pyre-ignore[3]: Wrapper preserves fn's signature at runtime via wraps
        def wrapper(*args: Any, **kwargs: Any) -> object:
            warnings.warn(msg, UserWarning, stacklevel=2)
            return fn(*args, **kwargs)

        # pyre-ignore[7]: wrapper has same runtime signature as fn via wraps
        return wrapper

    return decorator
