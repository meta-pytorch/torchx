#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

if [ ! "$(lintrunner --version)" ]
then
    echo "Please install lintrunner."
    exit 1
fi

lintrunner --force-color
