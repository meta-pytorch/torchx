#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import posixpath
import unittest
from unittest.mock import MagicMock, patch

import fsspec
from IPython.testing.globalipapp import get_ipython


class VersionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ip = get_ipython()

    def test_get_workspace(self) -> None:
        from torchx.notebook import get_workspace

        self.assertEqual(get_workspace(), "memory://torchx-workspace/")

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_workspacefile(self, stdout: MagicMock) -> None:
        from torchx.notebook import get_workspace

        cell = "print('Arrakis')"
        file_path = "foo/bar.py"
        out = self.ip.run_cell_magic("workspacefile", file_path, cell)
        self.assertEqual(out, None)
        self.assertEqual(
            stdout.getvalue(),
            "Added foo/bar.py to workspace memory://torchx-workspace/\n",
        )
        fs, workspace_path = fsspec.core.url_to_fs(get_workspace())
        path = posixpath.join(workspace_path, file_path)
        with fs.open(path, "rt") as f:
            self.assertEqual(f.read(), cell)
