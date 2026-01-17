#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import List

import fsspec
from torchx.tracker.backend.fsspec import (
    _decode_torchx_run_id,
    _encode_torchx_run_id,
    _read_config,
    create,
    FsspecTracker,
    generate_filename,
)


class FsspecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "local://session/id1"
        self.parent_run_id = "local://session/id0"
        self.test_dir = tempfile.mkdtemp(prefix="torchx_tracker_backend_fsspec_test")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def _write_file(self, filename: str, content: List[str]) -> Path:
        """
        Writes the given contents (element per line) into
        the provided filename (relative to the temp test_dir)
        and returns the path to the written file.
        """
        filepath = Path(self.test_dir) / filename
        with open(filepath, mode="w") as fp:
            fp.writelines([f"{line}\n" for line in content])
        return filepath

    def test_generate_filename(self) -> None:
        filename = generate_filename()
        self.assertIsNotNone(filename)

    def test_encode_decode_torchx_run(self) -> None:
        expected_run_id = "mast://session_id/app_id"

        self.assertEqual(
            _decode_torchx_run_id(_encode_torchx_run_id(expected_run_id)),
            expected_run_id,
        )

    def test_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            # TODO use memory based fsspec, `exists` check on dirs are always falsy
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_artifact(self.run_id, "tf", "~/logs")

            self.assertEqual(tracker.artifacts(self.run_id)["tf"].path, "~/logs")

    def test_metadata(self) -> None:
        fs = fsspec.filesystem("file")
        tracker = FsspecTracker(fs, self.test_dir)
        tracker.add_metadata(self.run_id, k1="1", k2=2)

        self.assertEqual(tracker.metadata(self.run_id)["k1"], "1")
        self.assertEqual(tracker.metadata(self.run_id)["k2"], 2)

    def test_sources(self) -> None:
        fs = fsspec.filesystem("file")
        tracker = FsspecTracker(fs, self.test_dir)
        tracker.add_source(self.run_id, self.parent_run_id)

        sources = list(tracker.sources(self.run_id))
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_run_id, self.parent_run_id)

    def test_sources_with_artifact(self) -> None:
        fs = fsspec.filesystem("file")
        tracker = FsspecTracker(fs, self.test_dir)
        tracker.add_source(self.run_id, self.parent_run_id)
        parent_id_for_artifact = "local://session/parent"
        artifact = "some_propery_x"
        tracker.add_source(self.run_id, self.parent_run_id)
        tracker.add_source(self.run_id, parent_id_for_artifact, artifact)

        self.assertEqual(len(list(tracker.sources(self.run_id))), 2)

        sources = list(tracker.sources(self.run_id, artifact_name=artifact))
        self.assertEqual(len(sources), 1)
        self.assertEqual(
            sources[0].source_run_id,
            parent_id_for_artifact,
        )

    def test_run_ids(self) -> None:
        fs = fsspec.filesystem("file")
        tracker = FsspecTracker(fs, self.test_dir)
        tracker.add_artifact(self.parent_run_id, "tf", "path1")
        tracker.add_artifact(self.run_id, "tf", "path2", {"k": "v"})
        tracker.add_source(self.run_id, self.parent_run_id)

        run_ids = tracker.run_ids()
        self.assertEqual(set(run_ids), {self.parent_run_id, self.run_id})

    def test_run_ids_filter_by_parent(self) -> None:
        fs = fsspec.filesystem("file")
        tracker = FsspecTracker(fs, self.test_dir)
        tracker.add_artifact(self.parent_run_id, "tf", "path1")
        tracker.add_artifact(self.run_id, "tf", "path2", {"k": "v"})
        tracker.add_source(self.run_id, self.parent_run_id)

        run_ids = tracker.run_ids(parent_run_id=self.parent_run_id)
        self.assertEqual(run_ids, [self.run_id])

    def test_read_config(self) -> None:
        configfile = self._write_file(
            "config.conf",
            [
                "protocol=file",
                "root_path=/home/bob/torchx",
                "# this is a comment a=b",
                "client_kwargs.endpointUrl=http://myminio:9000",
                "client_kwargs.foo.bar=baz",
            ],
        )
        self.assertDictEqual(
            {
                "protocol": "file",
                "root_path": "/home/bob/torchx",
                "client_kwargs": {
                    "endpointUrl": "http://myminio:9000",
                    "foo": {
                        "bar": "baz",
                    },
                },
            },
            _read_config(str(configfile)),
        )

    def test_read_config_bad_key(self) -> None:
        configfile = self._write_file(
            "config.conf",
            [
                "protocol=file",
                "root_path=/home/bob/torchx",
                "client_kwargs.endpointUrl.=http://myminio:9000",
            ],
        )
        with self.assertRaises(ValueError):
            _read_config(str(configfile))

    def test_create(self) -> None:
        fs = fsspec.filesystem("file")
        tracker_root_path = str(Path(self.test_dir) / "project")
        fs.mkdirs(tracker_root_path)
        configfile = self._write_file(
            "config.conf",
            [
                "protocol=file",
                f"root_path={tracker_root_path}",
            ],
        )

        tracker = create(f"file://{str(configfile)}")
        self.assertEqual(
            tracker._path_builder.root_dir, tracker_root_path  # pyre-ignore
        )
