# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Useful test fixtures (classes that you can subclass your python ``unittest.TestCase``)
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable, Dict, Iterable, List, TypeVar, Union

from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.launcher import elastic_launch, LaunchConfig
from torchx.schedulers.test import test_util

IS_CI: bool = os.getenv("CI", "false").lower() == "true"
IS_MACOS: bool = sys.platform == "darwin"


class TestWithTmpDir(unittest.TestCase):
    """
    A test fixture that creates and destroys (deletes) a temporary test directory for each
    test case by implementing a ``setUp()`` and ``tearDown()`` method. The temporary directory
    is made available via ``self.tmpdir`` parameter and is only valid for the duration of test case
    (not the whole test class).

    This test fixture also contains a few useful utility methods to manipulate
    temporary test files and directories (see usage below).

    Usage:

    .. code-block:: python

     from ape.test.utils import TestWithTmpDir

     class MyTest(TestWithTmpDir):

        def test_foo(self) -> None:
            self.tmpdir
    """

    def setUp(self) -> None:
        self.tmpdir: Path = Path(
            tempfile.mkdtemp(prefix=f"torchx-{self.__class__.__name__}-")
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def touch(self, filepath: str) -> Path:
        """
        Creates an empty file with the given name (equivalent to UNIX "touch") in the test's tmpdir
        returns a ``Path`` to the created file. The ``filepath`` can be a file name or a relative path.

        Usage:

        .. code-block:: python

         self.touch("foobar") # creates self.tmpdir / "foobar"
         self.touch("foo/bar") # creates self.tmpdir / "foo" / "bar"

        """

        f = self.tmpdir / filepath
        f.parent.mkdir(parents=True, exist_ok=True)

        f.touch()
        return f

    def mkdir(self, dirpath: str) -> Path:
        """
        Creates a directory in the test's tmpdir and returns the ``Path`` to the created directory
        """
        d = self.tmpdir / dirpath
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, filepath: str, content: Iterable[str]) -> Path:
        """
        Creates a file given the filepath (can be a file name or a relative path) in the test's tmpdir
        and writes the given content line-by-line into the file. Returns the filepath to the written file.

        Usage:

        .. code-block:: python

         self.write("foo.txt", content=["hello", "world"])
         self.write("foo/bar.txt", content=["hello", "world"])

        """

        f = self.touch(filepath)
        with open(f, "w") as fout:
            fout.writelines(content)
        return f

    def write_shell_script(self, script_path: str, content: List[str]) -> Path:
        """
        Writes a bash script with the body as provided by ``content`` in the test's tmp directory.
        The content should not include the shebang (``#!/bin/bash``) header as one is added automatically.
        The shell script is also made executable (``chmod 755``) so that it can be run from test methods.

        Usage:

        .. code-block:: python

         p = self.write_shell_script("echo.sh", ["echo hello", "echo world"])
         assert p == self.tmpdir / "echo.sh"

         p = self.write_shell_script("bin/echo.sh", ["echo hello", "echo world"])
         assert p == self.tmpdir / "bin" / "echo.sh")

        Returns: The path to the written shell script.

        """
        return Path(
            test_util.write_shell_script(str(self.tmpdir), script_path, content)
        )

    def read(self, filepath: Union[str, Path]) -> List[str]:
        """
        Reads the entire contents of the file into a list.
        The ``filepath`` is assumed to be relative to the test tmpdir


        Usage:

        .. code-block:: python

         f = self.write("foo/bar.txt", content=["hello", "world"])
         self.read(f) == ["hello", "world"]

        """

        with open(self.tmpdir / filepath, "r") as fin:
            return fin.readlines()

    def assertDirIsEmpty(self, directory: str | Path) -> None:
        """
        Asserts that the given directory exists but is empty.
        If ``dir`` is a relative path, it is assumed to be relative to this test's ``tmpdir``.
        """
        directory = Path(directory)
        d = directory if directory.is_absolute() else self.tmpdir / directory
        self.assertTrue(d.is_dir(), f"{d} is not a directory")
        self.assertEqual(0, len(list(d.iterdir())), f"{d} is not empty")

    def assertDirTree(self, root: str | Path, tree: dict[str, object]) -> None:
        """
        Asserts that the given ``root`` has the directory structure as specified by ``tree``.
        If ``root`` is a relative path, it is assumed to be relative to this test's ``tmpdir``

        Usage:

        .. code-block:: python
            self.assertDirTree(
                "out",
                {
                    "setup.py": "",
                    "torchx": {
                        "__init__.py": "",
                        "version.py": "0.8.0dev0",
                        "specs": {
                            "__init__.py": "",
                            "api.py": "",
                        },
                    },
                },
            )
        """
        root = Path(root)
        d = root if root.is_absolute() else self.tmpdir / root
        self.assertTrue(d.is_dir(), f"{d} is not a directory")

        def _assert_tree(current_dir: Path, subtree: dict[str, object]) -> None:
            # Check that the directory contains exactly the keys in subtree
            actual_entries = {p.name for p in current_dir.iterdir()}
            expected_entries = set(subtree.keys())
            self.assertSetEqual(
                expected_entries,
                actual_entries,
                f"contents of the dir `{current_dir}` do not match the expected",
            )
            for name, value in subtree.items():
                path = current_dir / name
                if isinstance(value, dict):
                    self.assertTrue(path.is_dir(), f"{path} is not a directory")
                    _assert_tree(path, value)
                else:
                    self.assertTrue(path.is_file(), f"{path} is not a file")
                    if value != "":
                        with open(path, "r") as f:
                            content = f.read().strip()
                        self.assertEqual(
                            content,
                            value,
                            f"file {path} content {content!r} does not match expected {value!r}",
                        )

        _assert_tree(d, tree)

    def create_dir_tree(self, root: str, tree: dict[str, object]) -> Path:
        """
        Creates the directory structure as specified by ``tree`` under ``self.tmpdir / root``

        Usage:

        .. code-block:: python
            self.createDirTree(
                "out",
                {
                    "README.md": "foobar",
                    "torchx": {
                        "__init__.py": "",
                        "version.py": "0.8.0dev0",
                        "specs": {
                            "__init__.py": "",
                            "api.py": "",
                        },
                    },
                },
            )
        """
        d = self.tmpdir / root
        d.mkdir(parents=True, exist_ok=True)

        def _create_tree(current_dir: Path, subtree: dict[str, object]) -> None:
            for name, value in subtree.items():
                path = current_dir / name
                if isinstance(value, dict):
                    path.mkdir(parents=True, exist_ok=True)
                    _create_tree(path, value)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        f.write(str(value))

        _create_tree(d, tree)
        return d


Ret = TypeVar("Ret")


class DistributedTestCase(TestWithTmpDir):
    """
    A ``unittest.TestCase`` that has utility methods to run tests that need to be run in the context
    of ``torch.distributed``.

    Usage:

    .. doctest::

        >>> from torchx.test.fixtures import DistributedTestCase
        >>> import torch.distributed as dist

        >>> class MyDistributedTest(DistributedTestCase):
        ...     @staticmethod
        ...     def run_test(arg1) -> str:
        ...         dist.init_process_group(backend="gloo")
        ...         # run whatever needs to be tested
        ...         return f"rank={dist.get_rank()}/{dist.get_world_size()} arg1={arg1}"
        ...     def test_foo(self) -> None:
        ...         ret = self.run_ddp(world_size=2, fn=MyDistributedTest.run_test)("hello-world")
        ...         self.assertDictEqual(
        ...           {
        ...             0: "rank=0/2 arg1=hello-world",
        ...             1: "rank=1/2 arg1=hello-world",
        ...           },
        ...           ret
        ...         )
        >>> MyDistributedTest().test_foo()
    """

    def run_ddp(
        self, world_size: int, fn: Callable[..., Ret]
    ) -> Callable[..., Dict[int, Ret]]:
        """
        Runs ``world_size`` copies of ``fn`` (one on each sub-process) as a DDP job on the local host.

        .. note::
            You MUST initialize the default process group as ``ape.distributed.util.init_process_group()``
            in your ``fn`` before running any distributed/collective operations.

        See class docstring for usage example.
        """
        config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=world_size,
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=0.01,
        )

        return elastic_launch(config, entrypoint=fn)
