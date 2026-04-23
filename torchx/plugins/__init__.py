# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3, 2, 16]

"""Plugin registration, discovery, and diagnostics for TorchX.

Decorate factory functions with :py:class:`register` and place them in
``torchx_plugins.*`` namespace packages for automatic discovery::

    # torchx_plugins/schedulers/my_scheduler.py
    from torchx.plugins import register

    @register.scheduler()
    def my_scheduler(session_name: str, **kwargs) -> Scheduler:
        ...

Discover plugins and print a diagnostic report::

    from torchx import plugins
    reg = plugins.registry()
    scheds = reg.get(plugins.PluginType.SCHEDULER)
    print(reg)

.. deprecated::
    Entry-point based registration (``[torchx.*]`` in ``pyproject.toml``)
    is deprecated.  Set ``TORCHX_PLUGINS_SOURCE=1`` (namespace package
    only) to opt out of entry-point discovery early.
"""

from torchx.plugins._registration import (
    EIGHTH,
    HALF,
    halve_mem_down_to,
    powers_of_two_gpus,
    QUARTER,
    register,
    resource_tags,
    SIXTEENTH,
    WHOLE,
)
from torchx.plugins._registry import (
    PluginRegistry,
    PluginType,
    RegistrationError,
    registry,
)


__all__ = [
    # _registration.py
    "register",
    "resource_tags",
    "powers_of_two_gpus",
    "halve_mem_down_to",
    "WHOLE",
    "HALF",
    "QUARTER",
    "EIGHTH",
    "SIXTEENTH",
    # _registry.py
    "RegistrationError",
    "PluginType",
    "PluginRegistry",
    "registry",
]
