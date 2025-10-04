# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from torchx.distributed import (
    init_pg,
    is_local_rank0,
    is_rank0,
    local_cuda_device,
    local_rank,
    on_local_rank0_first,
    on_rank0_first,
    rank,
    world_size,
)
from torchx.test.fixtures import DistributedTestCase, IS_CI, IS_MACOS

TORCHX_DIST_LOCAL_RANK = "torchx.distributed.local_rank"

TORCH_CUDA_IS_AVAILABLE = "torch.cuda.is_available"
TORCH_CUDA_DEVICE_COUNT = "torch.cuda.device_count"

DIST_GET_RANK = "torchx.distributed.dist.get_rank"
DIST_GET_WORLD_SIZE = "torchx.distributed.dist.get_world_size"
DIST_IS_INITIALIZED = "torchx.distributed.dist.is_initialized"
DIST_INIT_PROCESS_GROUP = "torchx.distributed.dist.init_process_group"
DIST_IS_NCCL_AVAILABLE = "torchx.distributed.dist.is_nccl_available"
DIST_GET_LOCAL_CUDA_DEVICE = "torchx.distributed.local_cuda_device"

WARNINGS_WARN = "warnings.warn"


def check_touch_file_rank0_first(filepath: Path) -> None:
    init_pg("gloo")
    with on_rank0_first():
        if is_rank0():
            # make sure it doesn't exist first
            assert (
                not filepath.exists()
            ), f"On rank: {rank()} {filepath} is not expected to exist"
            filepath.touch()

        assert filepath.exists(), f"{filepath} is expected to exist here"


def check_touch_file_local_rank0_first(filepath: Path) -> None:
    init_pg("gloo")
    with on_local_rank0_first():
        if is_local_rank0():
            # make sure it doesn't exist first
            assert (
                not filepath.exists()
            ), f"On rank: {local_rank()} {filepath} is not expected to exist"
            filepath.touch()

        assert filepath.exists(), f"{filepath} is expected to exist here"


class DistributedTest(DistributedTestCase):
    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

        super().tearDown()

    def test_is_rank0(self) -> None:
        # should return 0 even before the call to init_pg()
        self.assertTrue(is_rank0())

        # initializing process group from a single process
        # basically makes this process rank 0 (e.g. there are no other ranks)
        init_pg(backend="gloo")
        self.assertTrue(is_rank0())

        with mock.patch(DIST_GET_RANK, return_value=1):
            self.assertFalse(is_rank0())

    def test_is_local_rank0(self) -> None:
        with mock.patch(TORCHX_DIST_LOCAL_RANK, return_value=0):
            self.assertTrue(is_local_rank0())

        with mock.patch(TORCHX_DIST_LOCAL_RANK, return_value=2):
            self.assertFalse(is_local_rank0())

    def test_get_local_cuda_device(self) -> None:
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            self.assertEqual(local_cuda_device(), torch.device("cuda:2"))

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(local_cuda_device(), torch.device("cuda:0"))

    @mock.patch.dict(os.environ, {"LOCAL_RANK": "2"})
    def test_get_local_rank(self) -> None:
        # as long as the LOCAL_RANK env var is set
        # local_rank() should always read from it and should not
        # matter whether dist.is_initialized()

        with mock.patch(DIST_IS_INITIALIZED, return_value=True):
            self.assertEqual(2, local_rank())

        with mock.patch(DIST_IS_INITIALIZED, return_value=False):
            self.assertEqual(2, local_rank())

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_get_local_rank_trivial(self) -> None:
        # if LOCAL_RANK is not in env var
        # then make sure we display a warning if dist.is_initialized()
        # since this most likely means that the user's intention is to
        # use pytorch distributed but did not run the script with torchrun
        # in either case the return value is still trivially 0

        with mock.patch(DIST_IS_INITIALIZED, return_value=True), mock.patch(
            WARNINGS_WARN
        ) as mock_warn:
            self.assertEqual(0, local_rank())
            mock_warn.assert_called_once()

        with mock.patch(DIST_IS_INITIALIZED, return_value=False), mock.patch(
            WARNINGS_WARN
        ) as mock_warn:
            self.assertEqual(0, local_rank())
            mock_warn.assert_not_called()

    def test_init_process_group_trivial(self) -> None:
        self.assertFalse(dist.is_initialized())

        init_pg(backend="gloo")
        self.assertTrue(dist.is_initialized())
        self.assertEqual(0, dist.get_rank())
        self.assertEqual(1, dist.get_world_size())

    @unittest.skipIf(IS_CI and IS_MACOS, "no localhost on osx CI")  # pyre-ignore[56]
    def test_init_process_group_distributed(self) -> None:
        self.run_ddp(world_size=4, fn=DistributedTest.init_pg_and_check_rank)()

    @staticmethod
    def init_pg_and_check_rank() -> None:
        init_pg()
        assert int(os.environ["RANK"]) == dist.get_rank()
        assert int(os.environ["WORLD_SIZE"]) == dist.get_world_size()

    @mock.patch(DIST_INIT_PROCESS_GROUP)
    def test_init_process_group_backend_auto(self, init_pg_mock: MagicMock) -> None:
        with mock.patch(TORCH_CUDA_IS_AVAILABLE, return_value=True), mock.patch(
            TORCH_CUDA_DEVICE_COUNT, return_value=1
        ), mock.patch(DIST_IS_NCCL_AVAILABLE, return_value=True), mock.patch(
            DIST_GET_LOCAL_CUDA_DEVICE
        ) as mock_get_local_cuda_device:
            init_pg(backend="auto")
            mock_get_local_cuda_device.assert_called_once()
            init_pg_mock.assert_called_once_with(backend="nccl", rank=0, world_size=1)

        init_pg_mock.reset_mock()

        with mock.patch(TORCH_CUDA_IS_AVAILABLE, return_value=True), mock.patch(
            TORCH_CUDA_DEVICE_COUNT, return_value=0
        ), mock.patch(DIST_IS_NCCL_AVAILABLE, return_value=True):
            device = init_pg(backend="auto")
            self.assertEqual(torch.device("cpu"), device)
            init_pg_mock.assert_called_once_with(backend="gloo", rank=0, world_size=1)

        init_pg_mock.reset_mock()

        with mock.patch(DIST_IS_NCCL_AVAILABLE, return_value=False):
            device = init_pg(backend="auto")
            self.assertEqual(torch.device("cpu"), device)
            init_pg_mock.assert_called_once_with(backend="gloo", rank=0, world_size=1)

    @unittest.skipIf(IS_CI and IS_MACOS, "no localhost on osx CI")  # pyre-ignore[56]
    def test_on_rank0_first(self) -> None:
        self.run_ddp(world_size=8, fn=check_touch_file_rank0_first)(
            self.tmpdir / "sentinel"
        )

    @unittest.skipIf(IS_CI and IS_MACOS, "no localhost on osx CI")  # pyre-ignore[56]
    def test_on_local_rank0_first(self) -> None:
        self.run_ddp(world_size=8, fn=check_touch_file_local_rank0_first)(
            self.tmpdir / "sentinel"
        )

    def test_trivial_rank_and_world_size(self) -> None:
        self.assertFalse(dist.is_initialized())
        self.assertEqual(0, rank())
        self.assertEqual(0, local_rank())
        self.assertEqual(1, world_size())

    @mock.patch(DIST_GET_RANK, return_value=2)
    @mock.patch(DIST_IS_INITIALIZED, return_value=True)
    @mock.patch(DIST_GET_WORLD_SIZE, return_value=3)
    @mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}, clear=True)
    def test_non_trivial_rank_and_world_size(
        self, _0: MagicMock, _1: MagicMock, _2: MagicMock
    ) -> None:
        self.assertEqual(2, rank())
        self.assertEqual(1, local_rank())
        self.assertEqual(3, world_size())
