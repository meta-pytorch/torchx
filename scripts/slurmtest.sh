#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

REMOTE_WHEEL="$1"
VENV="$2"

BASE_DIR="$(dirname "$REMOTE_WHEEL")"
DIR="$BASE_DIR/project"
mkdir "$DIR"
cd "$DIR"

# shellcheck disable=SC1091
source /opt/slurm/etc/slurm.sh
sbatch --version
# shellcheck disable=SC1090
source "$VENV"/bin/activate
python --version

pip install "$REMOTE_WHEEL"
pip install numpy
pip install torch==1.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

cat <<EOT > .torchxconfig
[slurm]
partition=compute
time=10
comment=hello
nomem=true
EOT

cat <<EOT > main.py
print("hello world!")
EOT

APP_ID="$(torchx run --wait --log --scheduler slurm dist.ddp -j 2x1 --script main.py)"
torchx status "$APP_ID"
torchx describe "$APP_ID"
sacct -j "$(basename "$APP_ID")"
torchx log "$APP_ID"
LINES="$(torchx log "$APP_ID" | grep -c 'hello world')"

if [ "$LINES" -ne 2 ]
then
    echo "expected 2 log lines"
    exit 1
fi
