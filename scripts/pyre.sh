#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
pyre --version
pyre --noninteractive --search-path "${SITE_PACKAGES}" check
