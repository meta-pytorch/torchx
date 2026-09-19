# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Test fixtures for code that runs under ``torch.distributed``.
"""

from typing import Callable, TypeVar

from torch.distributed.launcher import LaunchConfig, elastic_launch

from torchx.testing.fixtures import TestWithTmpDir

Ret = TypeVar("Ret")


class DistributedTestCase(TestWithTmpDir):
    """
    A ``unittest.TestCase`` that has utility methods to run tests that need to be run in the context
    of ``torch.distributed``.

    Usage:

    .. doctest::

        >>> from torchx.testing.distributed import DistributedTestCase
        >>> import torch.distributed as dist

        >>> class MyDistributedTest(DistributedTestCase):
        ...     @staticmethod
        ...     def run_test(arg1) -> str:
        ...         dist.init_process_group(backend="gloo")
        ...         # run whatever needs to be tested
        ...         return f"rank={dist.get_rank()}/{dist.get_world_size()} arg1={arg1}"
        ...     def test_foo(self) -> None:
        ...         ret = self.run_ddp(world_size=2, fn=MyDistributedTest.run_test)("hello-world")
        ...         self.assertDictEqual(
        ...           {
        ...             0: "rank=0/2 arg1=hello-world",
        ...             1: "rank=1/2 arg1=hello-world",
        ...           },
        ...           ret
        ...         )
        >>> MyDistributedTest().test_foo()
    """

    def run_ddp(
        self, world_size: int, fn: Callable[..., Ret]
    ) -> Callable[..., dict[int, Ret]]:
        """
        Runs ``world_size`` copies of ``fn`` (one on each sub-process) as a DDP job on the local host.

        .. note::
            You MUST initialize the default process group as ``ape.distributed.util.init_process_group()``
            in your ``fn`` before running any distributed/collective operations.

        See class docstring for usage example.
        """
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=world_size,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=0.01,
        )

        return elastic_launch(config, entrypoint=fn)
