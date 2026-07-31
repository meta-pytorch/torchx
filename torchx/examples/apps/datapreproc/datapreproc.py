#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Data Preprocessing App Example
====================================

This is a simple TorchX app that downloads some data via HTTP, normalizes the
images via torchvision and then reuploads it via fsspec.

Usage
---------

.. note:: The datapreproc app is a single process python program, hence for
          local runs you can run it as a regular python program: ``python ./datapreproc.py``.
          TorchX lets you run this app on a remote cluster.

To launch with TorchX locally (see note above) run:

.. code-block:: shell-session

  $ torchx run -s local_cwd utils.python \
      --script ./datapreproc/datapreproc.py \
      -- \
      --input_path="http://cs231n.stanford.edu/tiny-imagenet-200.zip" \
      --output_path=/tmp/torchx/datapreproc


To launch this app onto a remote cluster, simply specify a different scheduler
in the ``-s`` option.

.. code-block:: shell-session

  $ torchx run -s kubernetes -cfg queue=foo,namespace=bar utils.python \
      --script ./datapreproc/datapreproc.py \
      -- \
      --input_path="http://cs231n.stanford.edu/tiny-imagenet-200.zip" \
      --output_path=/tmp/torchx/datapreproc \


"""

import argparse
import os
import sys
import tarfile
import tempfile
import zipfile

import fsspec
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="example data preprocessing",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="dataset to download",
        default="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="remote path to save the .tar.gz data to",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="limit number of processed examples",
    )
    return parser.parse_args(argv)


def download_and_extract_zip_archive(url: str, path: str) -> None:
    with fsspec.open(url, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_ref:
            zip_ref.extractall(path)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"downloading {args.input_path} to {tmpdir}...")
        download_and_extract_zip_archive(args.input_path, tmpdir)

        img_root = os.path.join(
            tmpdir,
            os.path.splitext(os.path.basename(args.input_path))[0],
        )
        print(f"img_root={img_root}")

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

                if args.limit and len(image_files) > args.limit:
                    break
        for path in tqdm(image_files, miniters=int(len(image_files) / 2000)):
            f = Image.open(path)
            f = transform(f)
            f.save(path)

        tar_path = os.path.join(tmpdir, "out.tar.gz")
        print(f"packing images into {tar_path}...")
        with tarfile.open(tar_path, mode="w:gz") as f:
            f.add(img_root, arcname="")

        print(f"uploading dataset to {args.output_path}...")
        fs, _, rpaths = fsspec.get_fs_token_paths(args.output_path)
        assert len(rpaths) == 1, "must have single output path"
        if fs.exists(rpaths[0]):
            fs.rm(rpaths[0])
        fs.put(tar_path, rpaths[0])


if __name__ == "__main__" and "NOTEBOOK" not in globals():
    main(sys.argv[1:])


# sphinx_gallery_thumbnail_path = '_static/img/gallery-app.png'
