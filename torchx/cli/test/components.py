# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torchx.specs as specs


def touch(file: str) -> specs.AppDef:
    touch = specs.Role(
        name="touch",
        image="/tmp",
        entrypoint="touch",
        args=[f"{file}.test"],
        num_replicas=1,
    )

    return specs.AppDef(name="touch", roles=[touch])


def touch_v2(file: str) -> specs.AppDef:
    touch = specs.Role(
        name="touch",
        image="/tmp",
        entrypoint="touch",
        args=[f"{file}.testv2"],
        num_replicas=1,
    )

    return specs.AppDef(name="touch", roles=[touch])


def simple(
    num_trainers: int = 10, trainer_image: str = "pytorch/torchx:latest"
) -> specs.AppDef:
    trainer = specs.Role(
        name="trainer",
        image=trainer_image,
        entrypoint="train_main.py",
        args=["--epochs", "50"],
        env={"MY_ENV_VAR": "foobar"},
        num_replicas=num_trainers,
    )

    ps = specs.Role(
        name="parameter_server",
        image=trainer_image,
        entrypoint="ps_main.py",
        num_replicas=10,
    )

    reader = specs.Role(
        name="reader",
        image=trainer_image,
        entrypoint="reader_main.py",
        args=["--buffer_size", "1024"],
        num_replicas=1,
    )

    return specs.AppDef(name="my_train_job", roles=[trainer, ps, reader])


def echo_stderr(msg: str) -> specs.AppDef:
    touch = specs.Role(
        name="echo",
        image="/tmp",
        entrypoint="python3",
        args=[
            "-c",
            "import sys; sys.stderr.write(sys.argv[1])",
            msg,
        ],
        num_replicas=1,
    )

    return specs.AppDef(name="echo", roles=[touch])
