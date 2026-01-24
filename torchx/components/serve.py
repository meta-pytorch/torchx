# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
These components aim to make it easier to interact with inference and serving
tools such as `torchserve <https://pytorch.org/serve/>`_.
"""


import torchx
import torchx.specs as specs


def torchserve(
    model_path: str,
    management_api: str,
    image: str = torchx.IMAGE,
    params: dict[str, object] | None = None,
    dryrun: bool = False,
) -> specs.AppDef:
    """Deploys the provided model to the given torchserve management API
    endpoint.

    >>> from torchx.components.serve import torchserve
    >>> torchserve(
    ...     model_path="s3://your-bucket/your-model.pt",
    ...     management_api="http://torchserve:8081",
    ... )
    AppDef(name='torchx-torchserve', ...)

    Args:
        model_path: The fsspec path to the model archive file.
        management_api: The URL to the root of the torchserve management API.
        image: Container to use.
        params: torchserve parameters.
            See https://pytorch.org/serve/management_api.html#register-a-model
        dryrun: Start the app, but does not perform actual work

    Returns:
        specs.AppDef: the TorchX application definition
    """

    args = [
        "-m",
        "torchx.apps.serve.serve",
        "--model_path",
        model_path,
        "--management_api",
        management_api,
    ]
    if params is not None:
        for param, value in params.items():
            args += [
                f"--{param}",
                str(value),
            ]
    if dryrun:
        args.append("--dryrun")

    return specs.AppDef(
        name="torchx-torchserve",
        roles=[
            specs.Role(
                name="worker",
                image=image,
                entrypoint="python",
                args=args,
                port_map={"model-download": 8222},
                resource=specs.Resource(cpu=1, gpu=0, memMB=1024),
            ),
        ],
    )
