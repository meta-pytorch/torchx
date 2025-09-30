#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys
from datetime import date

from setuptools import find_packages, setup


def get_version():
    # get version string from version.py
    # TODO: ideally the version.py should be generated when setup is run
    version_file = os.path.join(os.path.dirname(__file__), "torchx/version.py")
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    with open(version_file, "r") as f:
        version = re.search(version_regex, f.read(), re.M).group(1)
        return version


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("python >= 3.10 required for torchx")

    name = "torchx"
    NAME_ARG = "--override-name"
    if NAME_ARG in sys.argv:
        idx = sys.argv.index(NAME_ARG)
        name = sys.argv.pop(idx + 1)
        sys.argv.pop(idx)
    is_nightly = "nightly" in name

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()

    with open("dev-requirements.txt") as f:
        dev_reqs = f.read()

    version = get_nightly_version() if is_nightly else get_version()
    print(f"-- {name} building version: {version}")

    setup(
        # Metadata
        name=name,
        version=version,
        author="TorchX Devs",
        author_email="torchx@fb.com",
        description="TorchX SDK and Components",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/meta-pytorch/torchx",
        license="BSD-3",
        keywords=["pytorch", "machine learning"],
        python_requires=">=3.7",
        install_requires=reqs.strip().split("\n"),
        include_package_data=True,
        packages=find_packages(exclude=("examples", "*.test", "aws*", "*.fb")),
        test_suite="torchx.test.suites.unittests",
        entry_points={
            "console_scripts": [
                "torchx=torchx.cli.main:main",
            ],
            "torchx.tracker": [
                "fsspec=torchx.tracker.backend.fsspec:create",
            ],
            "fsspec.specs": [
                "torchx_minio=torchx.test.minio.MinioFS",
            ],
        },
        extras_require={
            "aws_batch": ["boto3"],
            "kubernetes": ["kubernetes>=11"],
            "dev": dev_reqs,
        },
        # PyPI package information.
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
