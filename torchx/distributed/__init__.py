# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convenience methods to use ``torch.distributed``.
"""

import logging
import os
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from torchx.util.cuda import has_cuda_devices
from typing_extensions import Literal

log: logging.Logger = logging.getLogger(__name__)


def local_rank() -> int:
    """
    Returns the local rank (aka rank within the node) of this process.
    Typically, the local rank is used to set the CUDA device on the node.

    .. warning::
        This function only works correctly if the invoker of the program sets ``LOCAL_RANK`` env var
        or invokes the program with ``torchrun`` (aka ``torch.distributed.run``) or ``torchx``.
        If ``LOCAL_RANK`` is not set or the process group is not initialized
        then this function assumes that the process is not distributed and trivially returns 0.

    """

    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:  # "LOCAL_RANK" not in os.environ
        if dist.is_initialized():
            warnings.warn(
                "\n"
                "==============================================================================================\n"
                " The default torch.distributed process group is initialized\n"
                " but the `LOCAL_RANK` environment variable is not set. Will trivially return 0 for local_rank.\n"
                " It is recommended to use torchrun/torchx to run your script or set the `LOCAL_RANK` manually.\n"
                " For additional details see:\n"
                "  1) https://meta-pytorch.org/torchx/latest/components/distributed.html\n"
                "  2) https://pytorch.org/docs/stable/elastic/run.html\n"
                "=============================================================================================="
            )
        return 0


def local_cuda_device() -> torch.device:
    """
    Returns the CUDA device (as a ``torch.device``) based on the local rank.

    .. note::
        For hardware agnostic code, prefer to use :py:func:`local_device`, which will
        return the correct device based on the backend the process group has been initialized with

    See Also: :py:func:`local_rank`.
    """
    return torch.device(f"cuda:{local_rank()}")


def local_device() -> torch.device:
    """
    Returns the device that the current process should be using for models and tensors
    based on the default process group.

    .. note:: If the process group has not been initialized
        then this method returns ``cuda`` if GPU is available on the machine, and ``cpu`` otherwise.

    Returns ``cuda:$LOCAL_RANK`` if the default process group's backend is ``nccl`` otherwise ``cpu``

    """

    if dist.is_initialized():
        default_pg = _get_default_group()
        return (
            local_cuda_device() if default_pg.name() == "nccl" else torch.device("cpu")
        )
    else:
        return torch.device("cuda") if has_cuda_devices() else torch.device("cpu")


def rank() -> int:
    """
    A non-distributed-safe get_rank call. Unlike ``torch.distributed.get_rank()``
    this method will not fail if being invoked from a non-distributed (e.g. process group not initialized)
    context. Therefore, this method is safe to use in internal methods that may be used
    in non-distributed contexts as well.

    Returns:
        If a process group has been initialized returns the value returned by ``torch.distributed.get_rank()``.
        Otherwise, returns the rank specified by the env var ``RANK`` or 0 (trivial rank) if no such env var exists.

    """
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.getenv("RANK", "0"))


def world_size() -> int:
    """
    A non-distributed-safe get_world_size call. Unlike ``torch.distributed.get_world_size()``,
    this method will not fail if being invoked from a non-distributed (e.g. process group not initialized)
    context. Therefore, this method is safe to use in internal mthods that may be used
    in non-distributed contexts as well.

    Returns:
        If a process group has been initialized returns the value returns by ``torch.distributed.get_world_size()``.
        Otherwise, returns the world size specified by the env var ``WORLD_SIZE`` or 1 (trivial world_size)
        if no such env var exists.

    """
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    """
    Returns ``True`` if the caller is rank 0 (in a distributed setting).
    If no process group has been initialized, then this method assumes
    that the caller is a single-process (aka not-distributed) and trivially returns ``True``.
    That is, for a non-distributed job there is only one process and hence that process
    is trivially rank 0.

    .. note::
      To initialize the process group prefer to use :py:func:init_process_group over
      ``torch.distributed.init_process_group()`` since the former can be called from
      both distributed and non-distributed scripts.

    """
    return rank() == 0


def is_local_rank0() -> bool:
    """
    Returns ``True`` if this process is local rank 0 and ``False`` otherwise.
    Used to perform an action just once per node. Example

    .. code-block:: python

     if is_local_rank0():
        # download a file just once per node
        download_file("s3://...")


    """
    return local_rank() == 0


Backend = Literal["nccl", "gloo", "auto"]


def init_pg(backend: Backend = "auto", **kwargs: Any) -> torch.device:
    """
    A convenience wrapper around ``torch.distributed.init_proces_group()``
    that makes initializing a trivial (single world_size) process group easy.

    Useful when you want to make your code portable across launching with
    simple python or with ``torchrun`` (aka ``torch.distributed.run``)

    Usage:


    .. doctest::

     >>> from torchx.distributed import init_pg
     >>> init_pg(backend="gloo") # or nccl # doctest: +SKIP
     device(type='cpu')

    The example above works to initialize a pytorch process group
    for the trivial (``world_size = 1``) and distributed (``world_size > 1``)
    cases without you having to write an explicit check with an if-else branch statement.

    You can pass ``backend="auto"`` to have this function select ``"nccl"``
    if there is a cuda device available, otherwise ``"gloo"`` (for CPU)


    .. doctest::

      >>> from torchx.distributed import init_pg
      >>> device = init_pg(backend="auto") # doctest: +SKIP


    In the code above, ``device`` will be ``cuda:{LOCAL_RANK}`` if the host has CUDA devices (GPUs)
    and ``cpu`` if not.

    Returns:
        The cuda device that this rank should be using or cpu device if ``backend="gloo"``
        or if ``backend="auto"`` and no GPUs are available on the host.

    """

    if backend == "auto":
        backend = (
            "nccl"
            if torch.cuda.is_available()  # returns True if gpu-torch was installed even on CPU host
            and (
                torch.cuda.device_count() > 0
            )  # so need to check for CUDA devices explicitly
            and dist.is_nccl_available()
            else "gloo"
        )

    # this means that the script was launched as a single python process
    # initialize a trivial process group
    if not dist.is_torchelastic_launched():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "0"  # port selection - selects free random port
        dist.init_process_group(backend=backend, rank=0, world_size=1, **kwargs)
    else:
        dist.init_process_group(backend=backend, **kwargs)

    if backend == "nccl":
        return local_cuda_device()
    else:
        return torch.device("cpu")


@contextmanager
def on_rank0_first() -> Iterator[None]:
    """
    Runs the piece of code that is wrapped in this context manager
    first on rank0 then on the rest of the ranks.

    Example:

    .. code-block:: python

        import time
        from torchx.distributed import on_rank0_first, rank

        with on_rank0_first():
            print(f"Running on rank {rank()} at {int(time.monotonic())}")
            time.sleep(10)


    Would print:

    .. code-block::

        Running on rank 0 at 12534774
        Running on rank 1 at 12534784 # at least +10 seconds on the other ranks
        Running on rank 2 at 12534784
        ...

    To run ONLY on rank0 use an if-statement as such:

    .. code-block:: python

        if is_rank0():
            print(f"Running on rank {dist.get_rank()}")

    The code above would only print once on rank 0.

    """
    if dist.is_initialized() and not is_rank0():
        dist.barrier()

    try:
        yield
    finally:
        if dist.is_initialized() and is_rank0():
            dist.barrier()


@contextmanager
def on_local_rank0_first() -> Iterator[None]:
    """
    Runs the piece of code that is wrapped in this context manager
    first on local rank 0 then on the rest of the ranks.

    The behavior is exactly the same as :py:func:`torchx.distributed.on_rank0_first`
    except that the barrier is on each local rank on each node (versus a global barrier on rank0).

    This is useful in situations there a node-local action (that would otherwise cause races)
    needs to be done first from a representative worker on each node. For instance,
    downloading a checkpoint file to a tmp dir on each node once, then having all the
    workers read off the downloaded file.

    .. note::
        For actions that need to be run first at a job level
        use :py:func:`torchx.distributed.on_rank0_first`

    """
    if dist.is_initialized() and not is_local_rank0():
        dist.barrier()

    try:
        yield
    finally:
        if dist.is_initialized() and is_local_rank0():
            dist.barrier()
