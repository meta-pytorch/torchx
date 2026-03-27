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
from typing import Callable, TypeVar


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


_F = TypeVar("_F", bound=Callable[..., object])


def deprecated(
    *,
    replacement: str | None = None,
    # pyre-ignore[34]: _F is bound when the returned decorator is called
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
        def wrapper(*args: object, **kwargs: object) -> object:
            warnings.warn(msg, UserWarning, stacklevel=2)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
