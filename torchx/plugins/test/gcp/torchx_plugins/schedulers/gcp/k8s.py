# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[56]

"""Stub GCP Kubernetes scheduler plugin for testing multi-package discovery."""

from torchx.plugins import register


class GcpKubernetesScheduler:
    """Stub GCP Kubernetes scheduler for testing."""

    __slots__ = ("session_name",)

    def __init__(self, session_name: str) -> None:
        self.session_name = session_name


@register.scheduler()
def gcp_k8s(session_name: str, **kwargs: object) -> GcpKubernetesScheduler:
    """Create a :class:`GcpKubernetesScheduler`."""
    return GcpKubernetesScheduler(session_name)
