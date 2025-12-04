#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import struct


START_CANDIDATES: str = "bcdfghjklmnpqrstvwxz"
END_CANDIDATES: str = START_CANDIDATES + "012345679"


def make_unique(name: str, string_length: int = 0) -> str:
    """
    Appends a unique 64-bit string to the input argument.

    Note that the unique string pulls entropy from `/dev/urandom` hence is not
    affected by `random.seed()`

    Args:
        name: the name string to unique-ify
        string_length: max length of the unique 64-bit string to append to the ``name``.
          Default is 0, which returns the length of a randomly generated 64-bit string (typically 11-14 characters long).

    Returns:
        string in format ``{name}-{unique_suffix}`
    """
    max_length = None if string_length == 0 else string_length
    return f"{name}-{random_id(max_length)}"


def random_uint64() -> int:
    """
    random_uint64 returns an random unsigned 64 bit int.
    """
    return struct.unpack("!Q", os.urandom(8))[0]


def random_id(max_length: int | None = None) -> str:
    """
    Generates an alphanumeric string ID that matches the requirements from
    https://kubernetes.io/docs/concepts/overview/working-with-objects/names/

    Note that the unique string pulls entropy from `/dev/urandom` hence is not
    affected by `random.seed()`

    If ``max_length`` is provided, the returned ID will be at most that many characters long.

    """
    # If a max_length is provided and is non-positive, return empty string
    if max_length is not None and max_length <= 0:
        return ""

    out = ""
    v = random_uint64()

    while v > 0:
        if out == "":
            candidates = START_CANDIDATES
        else:
            candidates = END_CANDIDATES

        char = v % len(candidates)
        v = v // len(candidates)
        out += candidates[char]

        if max_length is not None and len(out) >= max_length:
            break

    # NOTE: statistically the length of `out` is typically between 12-14 characters long
    return out
