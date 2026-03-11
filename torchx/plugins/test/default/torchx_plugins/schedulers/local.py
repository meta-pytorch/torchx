# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""Stub local scheduler plugin for testing namespace package discovery."""

from torchx.plugins import register


class LocalScheduler:
    """Stub local scheduler for testing."""

    __slots__ = ("session_name",)

    def __init__(self, session_name: str) -> None:
        self.session_name = session_name


@register.scheduler()
def local_cwd(session_name: str, **kwargs: object) -> LocalScheduler:
    """Create a :class:`LocalScheduler`."""
    return LocalScheduler(session_name)


@register.scheduler()
def local_docker(session_name: str, **kwargs: object) -> LocalScheduler:
    """Create a Docker-based :class:`LocalScheduler`."""
    return LocalScheduler(session_name)
