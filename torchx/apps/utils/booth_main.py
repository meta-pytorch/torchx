#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import sys

from torchx.runtime.tracking import FsspecResultTracker


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="evaluates the booth function at (x1, x2)"
    )
    parser.add_argument(
        "--x1",
        type=float,
        help="x1",
        required=True,
    )
    parser.add_argument(
        "--x2",
        type=float,
        help="x2",
        required=True,
    )
    parser.add_argument(
        "--tracker_base",
        type=str,
        help="URI of output directory to use as the tracker's base dir",
        default="/tmp/torchx-utils-booth",
    )
    parser.add_argument(
        "--trial_idx",
        type=int,
        help="trial index (ignore if not running hpo)",
        default=0,
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    x1 = args.x1
    x2 = args.x2

    eval = (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

    tracker = FsspecResultTracker(args.tracker_base)
    tracker[args.trial_idx] = {"booth_eval": eval}


if __name__ == "__main__":
    main(sys.argv[1:])
