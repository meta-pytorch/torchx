# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This contains TorchX utility components that are `ready-to-use` out of the box. These are
components that simply execute well known binaries (e.g. ``cp``)
and are meant to be used as tutorial materials or glue operations between
meaningful stages in a workflow.
"""

import os
import shlex
from typing import Annotated

import torchx
import torchx.specs as specs


def echo(
    msg: str = "hello world", image: str = torchx.IMAGE, num_replicas: int = 1
) -> specs.AppDef:
    """
    Echos a message to stdout (calls echo)

    Args:
        msg: message to echo
        image: image to use
        num_replicas: number of replicas to run

    """
    return specs.AppDef(
        name="echo",
        roles=[
            specs.Role(
                name="echo",
                image=image,
                entrypoint="echo",
                args=[msg],
                num_replicas=num_replicas,
                resource=specs.Resource(cpu=1, gpu=0, memMB=1024),
            )
        ],
    )


def touch(file: str, image: str = torchx.IMAGE) -> specs.AppDef:
    """
    Touches a file (calls touch)

    Args:
        file: file to create
        image: the image to use

    """
    return specs.AppDef(
        name="touch",
        roles=[
            specs.Role(
                name="touch",
                image=image,
                entrypoint="touch",
                args=[file],
                num_replicas=1,
                resource=specs.Resource(cpu=1, gpu=0, memMB=1024),
            )
        ],
    )


def sh(
    *args: str,
    image: str = torchx.IMAGE,
    num_replicas: int = 1,
    cpu: int = 1,
    gpu: int = 0,
    memMB: int = 1024,
    h: str | None = None,
    env: dict[str, str] | None = None,
    max_retries: int = 0,
    mounts: list[str] | None = None,
    entrypoint: str | None = None,
) -> specs.AppDef:
    """
    Runs the provided command via sh. Currently sh does not support
    environment variable substitution.

    Args:
        args: bash arguments
        image: image to use
        num_replicas: number of replicas to run
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
        env: environment varibles to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)
        max_retries: the number of scheduler retries allowed
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
        entrypoint: the entrypoint to use for the command (defaults to sh)
    """

    escaped_args = [shlex.quote(arg) for arg in args]
    if env is None:
        env = {}
    env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "WARNING"))

    if entrypoint is not None:
        resolved_entrypoint = entrypoint
        resolved_args = escaped_args
    else:
        resolved_entrypoint = "sh"
        resolved_args = ["-c", " ".join(escaped_args)]

    return specs.AppDef(
        name="sh",
        roles=[
            specs.Role(
                name="sh",
                image=image,
                entrypoint=resolved_entrypoint,
                args=resolved_args,
                num_replicas=num_replicas,
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                env=env,
                max_retries=max_retries,
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
    )


def python(
    *args: str,
    m: str | None = None,
    c: str | None = None,
    script: str | None = None,
    image: str = torchx.IMAGE,
    name: str = "torchx_utils_python",
    cpu: int = 1,
    gpu: int = 0,
    memMB: int = 1024,
    h: str | None = None,
    num_replicas: int = 1,
    mounts: Optional[List[str]] = None,
) -> specs.AppDef:
    """
    Runs ``python`` with the specified module, command or script on the specified
    image and host. Use ``--`` to separate component args and program args
    (e.g. ``torchx run utils.python --m foo.main -- --args to --main``)

    Note: (cpu, gpu, memMB) parameters are mutually exclusive with ``h`` (named resource) where
          ``h`` takes precedence if specified for setting resource requirements.
          See `registering named resources <https://meta-pytorch.org/torchx/latest/advanced.html#registering-named-resources>`_.

    Args:
        args: arguments passed to the program in sys.argv[1:] (ignored with `--c`)
        m: run library module as a script
        c: program passed as string (may error if scheduler has a length limit on args)
        script: .py script to run
        image: image to run on
        name: name of the job
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
        num_replicas: number of copies to run (each on its own container)
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
    :return:
    """
    if sum([m is not None, c is not None, script is not None]) != 1:
        raise ValueError(
            "exactly one of `-m`, `-c` and `--script` needs to be specified"
        )

    if script:
        cmd = [script]
    elif m:
        cmd = ["-m", m]
    elif c:
        cmd = ["-c", c]
    else:
        raise ValueError("no program specified")

    return specs.AppDef(
        name=name,
        roles=[
            specs.Role(
                name="python",
                image=image,
                entrypoint="python",
                num_replicas=num_replicas,
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                args=[*cmd, *args],
                env={"HYDRA_MAIN_MODULE": m} if m else {},
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
    )


def binary(
    *args: str,
    entrypoint: str,
    name: str = "torchx_utils_binary",
    num_replicas: int = 1,
    cpu: int = 1,
    gpu: int = 0,
    memMB: int = 1024,
    h: str | None = None,
) -> specs.AppDef:
    """
    Test component

    Args:
        args: arguments passed to the program in sys.argv[1:] (ignored with `--c`)
        name: name of the job
        num_replicas: number of copies to run (each on its own container)
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
    :return:
    """
    return specs.AppDef(
        name=name,
        roles=[
            specs.Role(
                name="binary",
                image="<NONE>",
                entrypoint=entrypoint,
                num_replicas=num_replicas,
                args=[*args],
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
            )
        ],
    )


def copy(src: str, dst: str, image: str = torchx.IMAGE, mounts: Optional[List[str]] = None,) -> specs.AppDef:
    """
    copy copies the file from src to dst. src and dst can be any valid fsspec
    url.

    This does not support recursive copies or directories.

    Args:
        src: the source fsspec file location
        dst: the destination fsspec file location
        image: the image that contains the copy app
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
    """

    return specs.AppDef(
        name="torchx-utils-copy",
        roles=[
            specs.Role(
                name="torchx-utils-copy",
                image=image,
                entrypoint="python",
                args=[
                    "-m",
                    "torchx.apps.utils.copy_main",
                    "--src",
                    src,
                    "--dst",
                    dst,
                ],
                resource=specs.Resource(cpu=1, gpu=0, memMB=1024),
                mounts=specs.parse_mounts(mounts) if mounts else [],
            ),
        ],
    )


def booth(
    x1: float,
    x2: float,
    trial_idx: int = 0,
    tracker_base: str = "/tmp/torchx-util-booth",
    image: str = torchx.IMAGE,
) -> specs.AppDef:
    """
    Evaluates the booth function, ``f(x1, x2) = (x1 + 2*x2 - 7)^2 + (2*x1 + x2 - 5)^2``.
    Output result is accessible via ``FsspecResultTracker(outdir)[trial_idx]``

    Args:
        x1: x1
        x2: x2
        trial_idx: ignore if not running hpo
        tracker_base: URI of the tracker's base output directory (e.g. s3://foo/bar)
        image: the image that contains the booth app
    """
    return specs.AppDef(
        name="torchx-utils-booth",
        roles=[
            specs.Role(
                name="torchx-utils-booth",
                image=image,
                entrypoint="python",
                args=[
                    "-m",
                    "torchx.apps.utils.booth_main",
                    "--x1",
                    str(x1),
                    "--x2",
                    str(x2),
                    "--trial_idx",
                    str(trial_idx),
                    "--tracker_base",
                    tracker_base,
                ],
                resource=specs.Resource(cpu=1, gpu=0, memMB=1024),
            )
        ],
    )


def hydra(
    *overrides: str,
    config_name: Annotated[str, "-cn"],
    config_dir: Annotated[str, "-cd"] = ".torchx",
) -> specs.AppDef:
    """Build AppDef from Hydra configuration.

    Config should have an 'app' key with _target_: torchx.specs.AppDef.
    Other top-level keys (like 'role') can be used for config groups and interpolation.

    Example:
        defaults:
          - role: python

        app:
          _target_: torchx.specs.AppDef
          name: my_job
          roles:
            - ${role}

    Args:
        overrides: Hydra config overrides (e.g., role.num_replicas=2)
        config_name: Config file name in config_dir
        config_dir: Directory containing configs (default: .torchx)

    Returns:
        AppDef instantiated from configuration
    """
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Register TorchX resolvers - return escaped strings so they're not re-resolved
    # The backslash escape tells OmegaConf to store the literal ${...} string
    OmegaConf.register_new_resolver(
        "torchx.app_id", lambda: f"\\{specs.macros.app_id}", replace=True
    )
    OmegaConf.register_new_resolver(
        "torchx.replica_id", lambda: f"\\{specs.macros.replica_id}", replace=True
    )
    OmegaConf.register_new_resolver(
        "torchx.rank0_env", lambda: f"\\{specs.macros.rank0_env}", replace=True
    )
    OmegaConf.register_new_resolver(
        "torchx.img_root", lambda: f"\\{specs.macros.img_root}", replace=True
    )
    config_dir = (
        config_dir if os.path.isabs(config_dir) else os.path.abspath(config_dir)
    )
    initialize_config_dir(config_dir=config_dir, version_base="1.3")
    cfg = compose(config_name=config_name, overrides=list(overrides))

    if os.environ.get("TORCHX_DEBUG"):
        print("=" * 80)
        print("TORCHX DEBUG: Configuration")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

    return instantiate(cfg.app, _convert_="all")
