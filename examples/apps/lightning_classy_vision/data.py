# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer Datasets Example
========================

This is the datasets used for the training example. It's using stock Pytorch
Lightning + Classy Vision libraries.
"""

import os.path
import tarfile
from typing import Optional, Callable

import fsspec
import pytorch_lightning as pl
from classy_vision.dataset.classy_dataset import ClassyDataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
# This uses classy vision to define a dataset that we will then later use in our
# Pytorch Lightning data module.


class TinyImageNetDataset(ClassyDataset):
    """
    TinyImageNetDataset is a ClassyDataset for the tiny imagenet dataset.
    """

    def __init__(self, data_path: str, transform: Callable[[object], object]) -> None:
        batchsize_per_replica = 16
        shuffle = False
        num_samples = 1000
        dataset = datasets.ImageFolder(data_path)
        super().__init__(
            # pyre-fixme[6]
            dataset,
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
        )


# %%
# For easy of use, we define a lightning data module so we can reuse it across
# our trainer and other components that need to load data.

# pyre-fixme[13]: Attribute `test_ds` is never initialized.
# pyre-fixme[13]: Attribute `train_ds` is never initialized.
# pyre-fixme[13]: Attribute `val_ds` is never initialized.
class TinyImageNetDataModule(pl.LightningDataModule):
    """
    TinyImageNetDataModule is a pytorch LightningDataModule for the tiny
    imagenet dataset.
    """

    train_ds: TinyImageNetDataset
    val_ds: TinyImageNetDataset
    test_ds: TinyImageNetDataset

    def __init__(self, data_dir: str, batch_size: int = 16) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        # Setup data loader and transforms
        img_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        self.train_ds = TinyImageNetDataset(
            data_path=os.path.join(self.data_dir, "train"),
            transform=lambda x: (img_transform(x[0]), x[1]),
        )
        self.val_ds = TinyImageNetDataset(
            data_path=os.path.join(self.data_dir, "val"),
            transform=lambda x: (img_transform(x[0]), x[1]),
        )
        self.test_ds = TinyImageNetDataset(
            data_path=os.path.join(self.data_dir, "test"),
            transform=lambda x: (img_transform(x[0]), x[1]),
        )

    def train_dataloader(self) -> DataLoader:
        # pyre-fixme[6]
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        # pyre-fixme[6]:
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        # pyre-fixme[6]
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


# %%
# To pass data between the different components we use fsspec which allows us to
# read/write to cloud or local file storage.


def download_data(remote_path: str, tmpdir: str) -> str:
    """
    download_data downloads the training data from the specified remote path via
    fsspec and places it in the tmpdir unextracted.
    """
    tar_path = os.path.join(tmpdir, "data.tar.gz")
    print(f"downloading dataset from {remote_path} to {tar_path}...")
    fs, _, rpaths = fsspec.get_fs_token_paths(remote_path)
    assert len(rpaths) == 1, "must have single path"
    fs.get(rpaths[0], tar_path)

    data_path = os.path.join(tmpdir, "data")
    print(f"extracting {tar_path} to {data_path}...")
    with tarfile.open(tar_path, mode="r") as f:
        f.extractall(data_path)

    return data_path
