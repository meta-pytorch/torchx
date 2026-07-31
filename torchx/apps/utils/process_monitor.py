#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import subprocess
import sys
import time

import fsspec

TIMEOUT_EXIT_CODE = 34


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="monitors a process and terminates it when a condition is met"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="timeout in seconds to wait before killing the process",
    )
    parser.add_argument(
        "--start_on_file",
        type=str,
        help="monitor specified fsspec path for existence and terminate process when the file is created",
    )
    parser.add_argument(
        "--exit_on_file",
        type=str,
        help="monitor specified fsspec path for existence and terminate process when the file is created",
    )
    parser.add_argument(
        "--poll_rate",
        type=float,
        default=5,
        help="seconds between polling the process and file for termination",
    )
    parser.add_argument(
        "--kill_timeout",
        type=float,
        default=60,
        help="seconds to wait after terminating the process before calling kill",
    )
    parser.add_argument(
        "entrypoint",
        type=str,
        help="program entrypoint",
    )
    parser.add_argument(
        "args",
        type=str,
        help="program args",
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    start_time = time.time()

    if args.start_on_file:
        fs, path = fsspec.core.url_to_fs(args.start_on_file)
        while True:
            if fs.exists(path):
                print(f"{args.start_on_file} exists, starting process...")
                break
            if args.timeout:
                elapsed_time = time.time() - start_time
                if elapsed_time > args.timeout:
                    print("reached timeout before launching, terminating...")
                    sys.exit(TIMEOUT_EXIT_CODE)

            time.sleep(args.poll_rate)

    p = subprocess.Popen([args.entrypoint] + args.args)
    print(f"started process {p.pid}")

    while True:
        try:
            p.wait(args.poll_rate)

            print(f"process exited with exit code {p.returncode}")
            sys.exit(p.returncode)
        except subprocess.TimeoutExpired:
            if args.timeout:
                elapsed_time = time.time() - start_time
                if elapsed_time > args.timeout:
                    print("reached timeout, terminating...")
                    break

            if args.exit_on_file:
                fs, path = fsspec.core.url_to_fs(args.exit_on_file)
                if fs.exists(path):
                    print(f"{args.exit_on_file} exists, terminating...")
                    break

    p.terminate()
    print("issued terminate, waiting for exit...")
    try:
        p.wait(args.kill_timeout)
    except subprocess.TimeoutExpired:
        print("reached safe termination timeout, killing...")
        p.kill()

    p.wait()
    print(f"process exited with exit code {p.returncode}")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
