# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Kueue scheduler plugin — wraps KubernetesScheduler with Kueue queue labels."""

from torchx.plugins import register
from torchx.schedulers.kubernetes_scheduler import KubernetesJob, KubernetesScheduler, Opts
from torchx.specs.api import AppDef, AppDryRunInfo

KUEUE_QUEUE_LABEL: str = "kueue.x-k8s.io/queue-name"


class KueueScheduler(KubernetesScheduler):
    """KubernetesScheduler variant that stamps ``kueue.x-k8s.io/queue-name`` labels."""

    def _submit_dryrun(self, app: AppDef, cfg: Opts) -> AppDryRunInfo[KubernetesJob]:
        dryrun_info = super()._submit_dryrun(app, cfg)
        queue = cfg.get("queue") or "default"
        assert isinstance(queue, str), "queue must be a str"
        dryrun_info.request.resource["metadata"].setdefault("labels", {})[KUEUE_QUEUE_LABEL] = queue
        return dryrun_info


@register.scheduler(name="kueue")
def kueue_scheduler(session_name: str, **kwargs: object) -> KueueScheduler:
    """Create a :class:`KueueScheduler`."""
    return KueueScheduler(session_name=session_name)
