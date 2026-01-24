# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Trainer Datasets Example
========================

This is the datasets used for the training example. It's using PyTorch Lightning
libraries.
"""

import os.path
import tarfile
from typing import Callable

import fsspec
import numpy
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm


# %%
# This uses torchvision to define a dataset that we will then later use in our
# Pytorch Lightning data module.


class ImageFolderSamplesDataset(datasets.ImageFolder):
    """
    ImageFolderSamplesDataset is a wrapper around ImageFolder that allows you to
    limit the number of samples.
    """

    def __init__(
        self,
        root: str,
        transform: Callable[..., object] | None = None,
        num_samples: int | None = None,
        **kwargs: object,
    ) -> None:
        """
        Args:
            num_samples: optional. limits the size of the dataset
        """
        super().__init__(root, transform=transform)
        self.num_samples = num_samples

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        return super().__len__()


# %%
# For easy of use, we define a lightning data module so we can reuse it across
# our trainer and other components that need to load data.


class TinyImageNetDataModule(pl.LightningDataModule):
    """
    TinyImageNetDataModule is a pytorch LightningDataModule for the tiny
    imagenet dataset.
    """

    # pyre-fixme[13]: Attribute `test_ds` is never initialized.
    train_ds: ImageFolderSamplesDataset
    # pyre-fixme[13]: Attribute `train_ds` is never initialized.
    val_ds: ImageFolderSamplesDataset
    # pyre-fixme[13]: Attribute `val_ds` is never initialized.
    test_ds: ImageFolderSamplesDataset

    def __init__(
        self, data_dir: str, batch_size: int = 16, num_samples: int | None = None
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_samples = num_samples

    def setup(self, stage: str | None = None) -> None:
        # Setup data loader and transforms
        img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.train_ds = ImageFolderSamplesDataset(
            root=os.path.join(self.data_dir, "train"),
            transform=img_transform,
            num_samples=self.num_samples,
        )
        self.val_ds = ImageFolderSamplesDataset(
            root=os.path.join(self.data_dir, "val"),
            transform=img_transform,
            num_samples=self.num_samples,
        )
        self.test_ds = ImageFolderSamplesDataset(
            root=os.path.join(self.data_dir, "test"),
            transform=img_transform,
            num_samples=self.num_samples,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def teardown(self, stage: str | None = None) -> None:
        pass


# %%
# To pass data between the different components we use fsspec which allows us to
# read/write to cloud or local file storage.


def download_data(remote_path: str, tmpdir: str) -> str:
    """
    download_data downloads the training data from the specified remote path via
    fsspec and places it in the tmpdir unextracted.
    """
    if os.path.isdir(remote_path):
        print("dataset path is a directory, using as is")
        return remote_path

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


def create_random_data(output_path: str, num_images: int = 250) -> None:
    """
    Fills the given path with randomly generated 64x64 images.
    This can be used for quick testing of the workflow of the model.
    Does NOT pack the files into a tar, but does preprocess them.
    """
    train_path = os.path.join(output_path, "train")
    class1_train_path = os.path.join(train_path, "class1")
    class2_train_path = os.path.join(train_path, "class2")

    val_path = os.path.join(output_path, "val")
    class1_val_path = os.path.join(val_path, "class1")
    class2_val_path = os.path.join(val_path, "class2")

    test_path = os.path.join(output_path, "test")
    class1_test_path = os.path.join(test_path, "class1")
    class2_test_path = os.path.join(test_path, "class2")

    paths = [
        class1_train_path,
        class1_val_path,
        class1_test_path,
        class2_train_path,
        class2_val_path,
        class2_test_path,
    ]

    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        for i in range(num_images):
            pixels = numpy.random.rand(64, 64, 3) * 255
            im = Image.fromarray(pixels.astype("uint8")).convert("RGB")
            im.save(os.path.join(path, f"rand_image_{i}.jpeg"))

    process_images(output_path)


def process_images(img_root: str) -> None:
    print("transforming images...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.ToPILImage(),
        ]
    )

    image_files = []
    for root, _, fnames in os.walk(img_root):
        for fname in fnames:
            path = os.path.join(root, fname)
            if not is_image_file(path):
                continue
            image_files.append(path)
    for path in tqdm(image_files, miniters=int(len(image_files) / 2000)):
        f = Image.open(path)
        f = transform(f)
        f.save(path)


# sphinx_gallery_thumbnail_path = '_static/img/gallery-lib.png'
