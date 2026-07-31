# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
In the actual implementation of `filelock`, the type `BaseFileLock` is marked
as abstract. And Pyre does not allow us to instantiate a variable of `Type[X]`
when `X` is abstract because we cannot be sure that the type is valid to
instantiate.

In reality, the `FileLock` type is always some concrete instantiatable type so
it is okay to use; this stub makes Pyre happy by declaring BaseFileLock as not
abstract.
"""

import types
import typing as t

class Timeout(TimeoutError):
    lock_file: str
    def __init__(self, lock_file: str) -> None: ...
    def __str__(self) -> str: ...

class BaseFileLock:
    def __init__(self, lock_file: str, timeout: float = -1) -> None: ...
    @property
    def lock_file(self) -> str: ...
    @property
    def timeout(self) -> float: ...
    @timeout.setter
    def timeout(self, value: float) -> None: ...
    @property
    def is_locked(self) -> bool: ...
    def acquire(
        self, timeout: t.Optional[float] = None, poll_intervall: float = 0.05
    ) -> BaseFileLock: ...
    def release(self, force: bool = False) -> None: ...
    def __enter__(self) -> BaseFileLock: ...
    def __exit__(
        self,
        exc_type: t.Optional[t.Type[BaseException]],
        exc_value: t.Optional[Exception],
        traceback: t.Optional[types.TracebackType],
    ) -> None: ...
    def __del__(self) -> None: ...

class WindowsFileLock(BaseFileLock): ...
class UnixFileLock(BaseFileLock): ...
class SoftFileLock(BaseFileLock): ...

FileLock: t.Type[BaseFileLock]
