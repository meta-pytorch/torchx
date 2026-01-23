#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import uuid

TORCHX_INTERNAL_SESSION_ID = "TORCHX_INTERNAL_SESSION_ID"

CURRENT_SESSION_ID: str | None = None


def get_session_id_or_create_new() -> str:
    """
    Returns the current session ID, or creates a new one if none exists.
    The session ID remains the same as long as it is in the same process.
    Please DO NOT use this function out of torchx codebase.
    """
    global CURRENT_SESSION_ID
    if CURRENT_SESSION_ID:
        return CURRENT_SESSION_ID
    env_session_id = os.getenv(TORCHX_INTERNAL_SESSION_ID)
    if env_session_id:
        CURRENT_SESSION_ID = env_session_id
        return CURRENT_SESSION_ID
    session_id = str(uuid.uuid4())
    CURRENT_SESSION_ID = session_id
    return session_id


def get_torchx_session_id() -> str | None:
    """
    Returns the torchx session ID.
    Please use this function to get the session ID out of torchx codebase.
    """
    return CURRENT_SESSION_ID
