# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
    app_env: Dict[str, str] = {}
    if env:
        app_env.update(env)
    entrypoint = os.path.join(specs.macros.img_root, script)
    ddp_role = specs.Role(
        name=role,
        image="dummy_image",
        entrypoint=entrypoint,
        args=list(script_args),
        env=app_env,
        num_replicas=nnodes,
        resource=specs.Resource(cpu=1, gpu=0, memMB=1),
    )

    # get app name from cli or derive it from the image. Note that image names
    # may contain "." which is not allowed in app names.
    return specs.AppDef(name, roles=[ddp_role])
