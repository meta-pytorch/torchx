# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
For distributed training, TorchX relies on the scheduler's gang scheduling
capabilities to schedule ``n`` copies of nodes. Once launched, the application
is expected to be written in a way that leverages this topology, for instance,
with PyTorch's
`DDP <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.
You can express a variety of node topologies with TorchX by specifying multiple
:py:class:`torchx.specs.Role` in your component's AppDef. Each role maps to
a homogeneous group of nodes that performs a "role" (function) in the overall
training. Scheduling-wise, TorchX launches each role as a sub-gang.


A DDP-style training job has a single role: trainers. Whereas a
training job that uses parameter servers will have two roles: parameter server, trainer.
You can specify different entrypoint (executable), num replicas, resource requirements,
and more for each role.


DDP Builtin
----------------

DDP-style trainers are common and easy to templetize since they are homogeneous
single role AppDefs, so there is a builtin: ``dist.ddp``. Assuming your DDP
training script is called ``main.py``, launch it as:

.. code:: shell-session

    # locally, 1 node x 4 workers
    $ torchx run -s local_cwd dist.ddp -j 1x4 --script main.py

    # locally, 2 node x 4 workers (8 total)
    $ torchx run -s local_cwd dist.ddp -j 2x4 --script main.py

    # remote (optionally pass --rdzv_port to use a different master port than the default 29500)
    $ torchx run -s kubernetes -cfg queue=default dist.ddp \\
        -j 2x4 \\
        --script main.py

    # remote -- elastic/autoscaling with 2 minimum and max 5 nodes with 8
    # workers each
    $ torchx run -s kubernetes dist.ddp -j 2:5x8 --script main.py


Note that the only difference compared to the local launch is the scheduler (``-s``).
The ``dist.ddp`` builtin uses ``torchelastic`` (more specifically ``torch.distributed.run``)
under the hood. Read more about torchelastic `here <https://pytorch.org/docs/stable/elastic/run.html>`_.

Components APIs
-----------------
"""
import os
import re
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torchx
import torchx.specs as specs
from torchx.components.structured_arg import StructuredJArgument, StructuredNameArgument
from torchx.specs import macros

_TORCH_DEBUG_FLAGS: Dict[str, str] = {
    "CUDA_LAUNCH_BLOCKING": "1",
    "NCCL_DESYNC_DEBUG": "1",
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
}
"""
These are commonly set environment variables to debug PyTorch execution.

* ``CUDA_LAUNCH_BLOCKING``: Read more `here <https://docs.nvidia.com/cuda/cuda-gdb/index.html#set-cuda-launch-blocking>`__.
* ``NCCL_DESYNC_DEBUG``
* ``TORCH_DISTRIBUTED_DEBUG``: Read more `here <https://pytorch.org/docs/stable/distributed.html#torch-distributed-debug>`__.
* ``TORCH_SHOW_CPP_STACKTRACES``: Read more `here <https://pytorch.org/docs/stable/distributed.html#torch-distributed-debug>`__.
"""


def spmd(
    *args: str,
    script: Optional[str] = None,
    m: Optional[str] = None,
    image: str = torchx.IMAGE,
    name: str = "/",
    h: str = "gpu.small",
    j: str = "1x1",
    env: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    max_retries: int = 0,
    mounts: Optional[List[str]] = None,
    debug: bool = False,
) -> specs.AppDef:
    """
    Usage (by script): torchx run spmd -j 2x8 -h aws_p4d.24xlarge --name my_experiment/trial_1 --script path/to/my/trainer.py -foo bar

    Usage (by module): torchx run spmd -j 2x8 -h aws_p4d.24xlarge --name my_experiment/trial_1 -m path.to.my.trainer -foo bar

    Usage (infer GPU count): torchx run spmd -j 2 -h p4d.24xlarge ... (same as -j 2x8)

    Creates a torchx.specs.AppDef (Job Definition) for a Single-Process-Multiple-Data (SPMD)
    style application. See: https://en.wikipedia.org/wiki/Single_program,_multiple_data.

    SPMD launches `n x m` (set via the `-j nxm` option) copies of the same program,
    where `n` is the number of nodes (hosts) and `m` is the number of processes on each node.

    If you have a distributed PyTorch script (DDP, FSDP, RPC) use this component to launch
    the distributed application. You can also use `-j 1x1` to launch a single process application
    which would be equivalent to launching with regular `python` except that your application
    can safely call `torch.distributed.init_process_group(backend)`.

    Note: For multi-node distributed runs, the hosts MUST have a network route to each other
          AND port 29500 should be open on all hosts. Please check your security group settings.


    Args:
        args: the arguments to the main module or script (e.g. my/trainer.py -foo bar)
            (for docker based runs) the script path must be relative to the WORKDIR of the image
        script:
        m: the main module name (e.g. my.module.trainer). When this option is used, the `script_args` are passed
           as the arguments to the main module). Invoking my module is useful when the relative/absolute path
           of the main script is unknown w.r.t the WORKDIR of the image. Use this option when it makes sense to
           invoke the main script via `python -m <MAIN.MODULE>`.
        image: the base docker image of the workspace, if workspace is disabled, then the image of the job
        name: ``{experimentname}/{runname}`` or ``{experimentname}/`` or ``/{runname}`` or ``{runname}``
        h: the type of host to run on (e.g. aws_p4d.24xlarge). Must be one of the registered named resources
        j: {nnodes}x{nproc_per_node}. For GPU hosts omitting nproc_per_node will infer it from the GPU count on the host
        env: environment variables to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)
        metadata: metadata to be passed to the scheduler (e.g. KEY1=v1,KEY2=v2,KEY3=v3)
        max_retries: the number of scheduler retries allowed
        mounts: (for docker based runs only) mounts to mount into the worker environment/container
                (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
        debug: whether to run with preset debug flags enabled

    """

    if env is None:
        env = {}

    return ddp(
        *args,
        script=script,
        m=m,
        image=image,
        name=name,
        h=h,
        j=str(StructuredJArgument.parse_from(h, j)),
        env=env,
        metadata=metadata,
        max_retries=max_retries,
        mounts=mounts,
        debug=debug,
    )


def ddp(
    *script_args: str,
    script: Optional[str] = None,
    m: Optional[str] = None,
    image: str = torchx.IMAGE,
    name: str = "/",
    h: Optional[str] = None,
    cpu: int = 2,
    gpu: int = 0,
    memMB: int = 1024,
    j: str = "1x2",
    env: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    max_retries: int = 0,
    rdzv_port: int = 29500,
    rdzv_backend: str = "c10d",
    rdzv_conf: Optional[str] = None,
    mounts: Optional[List[str]] = None,
    debug: bool = False,
    tee: int = 3,
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate PyTorch worker processes. Defaults to using ``c10d`` rendezvous backend
    on rendezvous_endpoint ``$rank_0_host:$rdzv_port``. Note that ``rdzv_port`` parameter is ignored
    when running on single node, and instead we use port 0 which instructs torchelastic to chose
    a free random port on the host.

    Note: (cpu, gpu, memMB) parameters are mutually exclusive with ``h`` (named resource) where
          ``h`` takes precedence if specified for setting resource requirements.
          See `registering named resources <https://meta-pytorch.org/torchx/latest/advanced.html#registering-named-resources>`_.

    Args:
        script_args: arguments to the main module
        script: script or binary to run within the image
        m: the python module path to run
        image: image (e.g. docker)
        name: job name override in the following format: ``{experimentname}/{runname}`` or ``{experimentname}/`` or ``/{runname}`` or ``{runname}``.
            Uses the script or module name if ``{runname}`` not specified.
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
        j: [{min_nnodes}:]{nnodes}x{nproc_per_node}, for gpu hosts, nproc_per_node must not exceed num gpus
        env: environment varibles to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)
        metadata: metadata to be passed to the scheduler (e.g. KEY1=v1,KEY2=v2,KEY3=v3)
        max_retries: the number of scheduler retries allowed
        rdzv_port: the port on rank0's host to use for hosting the c10d store used for rendezvous.
                   Only takes effect when running multi-node. When running single node, this parameter
                   is ignored and a random free port is chosen.
        rdzv_backend: the rendezvous backend to use. Only takes effect when running multi-node.
        rdzv_conf: the additional rendezvous configuration to use (ex. join_timeout=600,close_timeout=600,timeout=600).
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
        debug: whether to run with preset debug flags enabled
        tee: tees the specified std stream(s) to console + file. 0: none, 1: stdout, 2: stderr, 3: both
    """

    if (script is None) == (m is None):
        raise ValueError("exactly one of --script and -m must be specified")

    # nnodes: number of nodes or minimum nodes for elastic launch
    # max_nnodes: maximum number of nodes for elastic launch
    # nproc_per_node: number of processes on each node
    min_nnodes, max_nnodes, nproc_per_node, nnodes_rep = parse_nnodes(j)

    if max_nnodes == 1:
        # using port 0 makes elastic chose a free random port which is ok
        # for single-node jobs since all workers run under a single agent
        # When nnodes is 0 and max_nnodes is 1, it's stil a single node job
        # but pending until the resources become available
        rdzv_endpoint = "localhost:0"
    else:
        # for multi-node, rely on the rank0_env environment variable set by
        # the schedulers (see scheduler implementation for the actual env var this maps to)
        # some schedulers (e.g. aws batch) make the rank0's ip-addr available on all BUT on rank0
        # so default to "localhost" if the env var is not set or is empty
        # rdzv_endpoint bash resolves to something to the effect of
        # ${TORCHX_RANK0_HOST:=localhost}:29500
        # use $$ in the prefix to escape the '$' literal (rather than a string Template substitution argument)
        rdzv_endpoint = _noquote(f"$${{{macros.rank0_env}:=localhost}}:{rdzv_port}")

    env = env or {}
    metadata = metadata or {}

    argname = StructuredNameArgument.parse_from(
        name=name,
        m=m,
        script=script,
    )

    env["TORCHX_TRACKING_EXPERIMENT_NAME"] = argname.experiment_name
    env["TORCHX_TRACKING_RUN_NAME"] = argname.run_name

    env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "WARNING"))
    if debug:
        env.update(_TORCH_DEBUG_FLAGS)

    cmd = [
        "torchrun",
        "--rdzv_backend",
        rdzv_backend,
        *(["--rdzv_conf", rdzv_conf] if rdzv_conf is not None else []),
        "--rdzv_endpoint",
        rdzv_endpoint,
        "--rdzv_id",
        f"{macros.app_id}",
        "--nnodes",
        nnodes_rep,
        "--nproc_per_node",
        str(nproc_per_node),
        "--tee",
        str(tee),
        "--role",
        "",
    ]
    # TODO 'node_rank' is made optional as it currently does not work with the AWS Batch scheduler.
    # node_rank is only used when rdzv_backend is 'static'
    if rdzv_backend == "static":
        cmd += ["--node_rank", f"{macros.replica_id}"]
    if script is not None:
        cmd += [script]
    elif m is not None:
        cmd += ["-m", m]
    cmd += script_args
    return specs.AppDef(
        name=argname.run_name,
        roles=[
            specs.Role(
                name=get_role_name(script, m),
                image=image,
                min_replicas=min_nnodes,
                entrypoint="bash",
                num_replicas=int(max_nnodes),
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                args=["-c", _args_join(cmd)],
                env=env,
                port_map={
                    "c10d": rdzv_port,
                },
                max_retries=max_retries,
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
        metadata=metadata,
    )


def get_role_name(script: Optional[str], m: Optional[str]) -> str:
    if script:
        # script name/module no extension
        role_name = Path(script).stem
    elif m:
        role_name = m.rpartition(".")[2]
    else:
        raise ValueError("failed to compute role_name")
    return role_name


def _args_join(args: Iterable[str]) -> str:
    """
    _args_join is like shlex.join but if the argument is wrapped in _noquote
    it'll not quote that argument.
    """
    quoted = [arg if isinstance(arg, _noquote) else shlex.quote(arg) for arg in args]
    return " ".join(quoted)


class _noquote(str):
    """
    _noquote is a wrapper around str that indicates that the argument shouldn't
    be passed through shlex.quote.
    """

    pass


def parse_nnodes(j: str) -> Tuple[int, int, int, str]:
    """
    parse_nnodes converts a node and process string into the individual
    components. Format is ``[[<min_replicas>:]<replicas>x]<num processes>``.
    """
    if re.match("^\\d+:\\d+x\\d+$", j):  # match 2:4x1
        nnodes_rep, nproc_per_node = j.split("x")
        min_nnodes, max_nnodes = nnodes_rep.split(":")
    elif re.match("^\\d+x\\d+$", j):  # match 2x1
        min_nnodes, nproc_per_node = j.split("x")
        max_nnodes = min_nnodes
        nnodes_rep = min_nnodes
    elif re.match("^\\d+$", j):  # match 2
        min_nnodes = "1"
        max_nnodes = min_nnodes
        nnodes_rep = min_nnodes
        nproc_per_node = j
    else:
        raise ValueError(
            f"Invalid format for -j, usage example: 1:2x4 or 1x4 or 4. Given: {j}"
        )
    return int(min_nnodes), int(max_nnodes), int(nproc_per_node), nnodes_rep
