#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import os
import shutil
import sys

import fsspec


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="copies a file between fsspec locations"
    )
    parser.add_argument(
        "--src",
        type=str,
        help="fsspec location of the file to read from",
        required=True,
    )
    parser.add_argument(
        "--dst",
        type=str,
        help="fsspec location of where to copy the file to",
        required=True,
    )
    parser.add_argument(
        "--bufsize",
        type=int,
        help="bufsize to use for copying",
        default=64 * 1024,
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    print(f"copying from {args.src} to {args.dst}")

    src_fs, src_path = fsspec.core.url_to_fs(args.src)
    dst_fs, dst_path = fsspec.core.url_to_fs(args.dst)
    dst_dir_path = os.path.dirname(dst_path)
    try:
        dst_fs.mkdir(dst_dir_path, create_parents=True)
    except FileExistsError:
        # Some fs, e.g. memory://, raise this exception if the dir already exists.
        pass

    if src_fs == dst_fs:
        print("filesystems are the same, using fs.copy() method")
        src_fs.copy(src_path, dst_path)
    else:
        print("filesystems are different, using shutil.copyfileobj()")
        with src_fs.open(src_path, "rb") as src, dst_fs.open(dst_path, "wb") as dst:
            shutil.copyfileobj(src, dst, args.bufsize)


if __name__ == "__main__":
    main(sys.argv[1:])
