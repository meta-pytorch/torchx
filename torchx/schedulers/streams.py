#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import os
import threading
import time


class Tee:
    """
    Tee is a IO writer that allows writing to multiple writers. This uses pipe
    and file descriptors so it works with subprocess.

    .. note:: Tee creates a background thread so must be closed.

    The order of writes is not guaranteed when writing to both fileno() and
    directly via write().
    """

    def __init__(self, out: io.FileIO, *sources: str) -> None:
        assert len(sources) > 0, "must have at least one stream"
        self.streams_lock = threading.Lock()  # protects streams
        self.out = out
        self.streams: list[io.FileIO] = []
        for source in sources:
            r = io.open(source, "rb", buffering=0)
            os.set_blocking(r.fileno(), False)
            self.streams.append(r)

        self._closed = False

        self.thread = threading.Thread(
            target=self._start_thread,
        )
        self.thread.daemon = True
        self.thread.start()

    def _start_thread(self) -> None:
        BUFSIZE = 64000
        while True:
            read = False
            for r in self.streams:
                data = r.read(BUFSIZE)
                if data:
                    read = True
                    self.write(data)
            if not read:
                if self._closed:
                    break
                time.sleep(0.1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.thread.join()
        with self.streams_lock:
            for s in self.streams:
                s.close()
            self.out.close()

    def write(self, s: bytes) -> int:
        with self.streams_lock:
            return self.out.write(s)
