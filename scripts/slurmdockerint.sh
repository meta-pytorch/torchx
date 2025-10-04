#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

DIST=dist

rm -r $DIST || true
python setup.py bdist_wheel

WHEEL="$DIST/$(ls $DIST)"

JOB="$USER-$(uuidgen)"
DIR="/data/integ-tests/$JOB"
VENV="$DIR/venv"

function run_cmd {
  # shellcheck disable=SC2048,SC2086
  docker exec c1 $*
}

function run_scp {
  docker cp "$1" "c1:$2"
}

function cleanup {
  echo "Removing $DIR"
  run_cmd rm  -r "$DIR"
}
trap cleanup EXIT

REMOTE_WHEEL="$DIR/$(basename "$WHEEL")"

SCRIPT="scripts/slurmtest.sh"
REMOTE_SCRIPT="$DIR/$(basename "$SCRIPT")"

for host in c1 c2
do
  docker exec "$host" dnf install python3.11 -y
done

run_cmd mkdir -p "$DIR"
run_cmd python3.11 -m venv "$VENV"
run_scp "$WHEEL" "$REMOTE_WHEEL"
run_scp "$SCRIPT" "$REMOTE_SCRIPT"
run_cmd "$REMOTE_SCRIPT" "$REMOTE_WHEEL" "$VENV"
