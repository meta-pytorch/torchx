#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

rm -r dist || true

# Temporarily change package name to torchx-nightly in pyproject.toml
sed -i 's/^name = "torchx"$/name = "torchx-nightly"/' pyproject.toml

# Build the wheel using uv
uv build --wheel

# Restore original package name
sed -i 's/^name = "torchx-nightly"$/name = "torchx"/' pyproject.toml

if [ -z "$PYPI_TOKEN" ]; then
    echo "must specify PYPI_TOKEN"
    exit 1
fi

uv run twine upload \
    --username __token__ \
    --password "$PYPI_TOKEN" \
    dist/torchx_nightly-*
