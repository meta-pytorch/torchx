# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional

import torchx.specs as specs


def ddp(
    script: str,
    nnodes: int = 1,
    name: str = "ddp_app",
    role: str = "worker",
    env: Optional[Dict[str, str]] = None,
    *script_args: str,
) -> specs.AppDef:
    """Single role application.

    Single role application.

    Args:
        script: Script to execute.
        nnodes: Number of nodes to launch.
        name: Name of the application.
        role: Name of the role.
        env: Env variables.
        script_args: Script arguments.

    Returns:
        specs.AppDef: Torchx AppDef
    """
    app_env: Dict[str, str] = {}
    if env:
        app_env.update(env)
    entrypoint = os.path.join(specs.macros.img_root, script)
    ddp_role = (
        specs.Role(
            name=role,
            image="dummy_image",
            resource=specs.Resource(cpu=1, gpu=0, memMB=1),
        )
        .runs(entrypoint, *script_args, **app_env)
        .replicas(nnodes)
    )

    # get app name from cli or extract from fbpkg. Note that fbpkg name can has "."
    # but not allowed in app name.
    return specs.AppDef(name).of(ddp_role)
