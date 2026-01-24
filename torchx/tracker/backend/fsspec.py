# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import os
import time
from base64 import b32decode, b32encode
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import fsspec
from torchx.tracker.api import Lineage, TrackerArtifact, TrackerBase, TrackerSource


def generate_filename() -> str:
    timestamp = time.time_ns()
    return str(timestamp)


def _encode_torchx_run_id(run_id: str) -> str:
    """
    Encodes run_id that can be POSIX compatible, case-aware filename

    Note: TorchX run_id scheme is: <scheduler_backend>://<session_name>/<app_id>
    """
    return b32encode(run_id.encode("ascii")).decode("utf-8")


def _decode_torchx_run_id(s: str) -> str:
    """
    Symmetric operation for `_encode_torchx_run_id` function to decode filename as a run_id
    """
    return b32decode(s).decode("utf-8")


@dataclass(frozen=True)
class _FsspecTrackerPathBuilder:
    """
    Encapsulation of the path logic used by `FsspecTracker`
    """

    root_dir: str

    def with_run_id(self, run_id: str) -> _FsspecTrackerPathBuilder:
        encoded_run_id = _encode_torchx_run_id(run_id)
        return self.with_subpath(encoded_run_id)

    def with_subpath(self, subpath: str) -> _FsspecTrackerPathBuilder:
        os.path.join(self.root_dir, subpath)
        return _FsspecTrackerPathBuilder(root_dir=os.path.join(self.root_dir, subpath))

    def with_unique_filename(self) -> _FsspecTrackerPathBuilder:
        filename = generate_filename()
        return _FsspecTrackerPathBuilder(root_dir=os.path.join(self.root_dir, filename))

    def path(self) -> str:
        return self.root_dir


class FsspecTracker(TrackerBase):
    """
    Implements `TrackerBase` using Fsspec abstraction and has an advantage of
    using various storage options for persisting the data.

    Important: `torchx.tracker.api` API is still experimental, hence there
    are no backwards compatibility gurantees with future releases yet.

    Each run will have a directory with subdirs for metadata, artifact, source and descendants data.
    """

    def __init__(self, fs: fsspec.AbstractFileSystem, root_dir: str) -> None:
        assert fs.exists(
            root_dir
        ), f"Expects FSSpec tracker directory '{root_dir}' to exist."
        self.fs = fs
        self._path_builder = _FsspecTrackerPathBuilder(root_dir)

    def _persist(
        self, run_id: str, category: str, content: Mapping[str, object]
    ) -> None:
        category_path_builder = self._path_builder.with_run_id(run_id).with_subpath(
            category
        )
        self.fs.mkdirs(category_path_builder.path(), exist_ok=True)
        filename = category_path_builder.with_unique_filename().path()
        with self.fs.open(filename, mode="w") as f:
            json.dump(content, f)

    def _load(self, run_id: str, category: str) -> Iterable[Mapping[str, object]]:
        data = []
        category_path_builder = self._path_builder.with_run_id(run_id).with_subpath(
            category
        )
        if not self.fs.exists(category_path_builder.path()):
            return data
        for listing in self.fs.listdir(category_path_builder.path()):
            file_name = listing["name"]
            with self.fs.open(file_name) as f:
                content = json.load(f)
                data.append(content)
        return data

    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        entry = {"name": name, "path": path, "metadata": metadata}
        self._persist(run_id, "artifacts", entry)

    def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]:
        artifacts = {}
        entries = self._load(run_id, "artifacts")
        for entry in entries:
            name = str(entry["name"])
            path = str(entry["path"])
            metadata = entry["metadata"]
            artifact = TrackerArtifact(name, path, metadata)  # pyre-ignore
            artifacts[name] = artifact
        return artifacts

    def add_metadata(self, run_id: str, **metadata: object) -> None:
        entry = {"metadata": metadata}
        self._persist(run_id, "metadata", entry)

    def metadata(self, run_id: str) -> Mapping[str, object]:
        metadata = {}
        entries = self._load(run_id, "metadata")
        for entry in entries:
            stored_metadata = entry["metadata"]
            for k, v in stored_metadata.items():  # pyre-ignore
                metadata[k] = v
        return metadata

    def add_source(
        self,
        run_id: str,
        source_id: str,
        artifact_name: str | None = None,
    ) -> None:
        sources_path_builder = self._path_builder.with_run_id(run_id).with_subpath(
            "sources"
        )
        self.fs.mkdirs(sources_path_builder.path(), exist_ok=True)
        parent_ref_file = sources_path_builder.with_run_id(source_id).path()
        artifact_name = artifact_name or ""

        with self.fs.open(parent_ref_file, mode="w") as f:
            artifact_name = artifact_name or ""
            f.write(f"{artifact_name}\n")

        # write into parent as well (if exists) that will allow traversing descendants
        parent_descendants_path_builder = self._path_builder.with_run_id(
            source_id
        ).with_subpath("descendants")
        parent_descendants_ref_path = parent_descendants_path_builder.path()
        if self.fs.exists(parent_descendants_ref_path):
            if not self.fs.exists(parent_descendants_ref_path):
                self.fs.mkdirs(parent_descendants_ref_path)
            descendant_ref_path = parent_descendants_path_builder.with_run_id(
                run_id
            ).path()
            with self.fs.open(descendant_ref_path, mode="a") as f:
                f.write(f"{artifact_name}\n")

    def _read_source_file(
        self, source_file: str, artifact_name: str | None = None
    ) -> Iterable[TrackerSource]:
        entries = []

        name = os.path.basename(source_file)
        with self.fs.open(source_file) as f:
            lines = [l.strip() for l in f.readlines()]
            lines = [line.decode() for line in lines if line]
            source_run_id = _decode_torchx_run_id(name)
            if lines:
                for artifact_type in lines:
                    if artifact_name and artifact_type != artifact_name:
                        continue
                    tracker_source = TrackerSource(source_run_id, artifact_type)
                    entries.append(tracker_source)
            elif not artifact_name:
                tracker_source = TrackerSource(source_run_id, None)
                entries.append(tracker_source)
            return entries

    def sources(
        self,
        run_id: str,
        artifact_name: str | None = None,
    ) -> Iterable[TrackerSource]:
        entries = []

        sources_path_builder = self._path_builder.with_run_id(run_id).with_subpath(
            "sources"
        )
        sources_path = sources_path_builder.path()
        if self.fs.exists(sources_path):
            source_files = self.fs.listdir(sources_path)
            for source_file in source_files:
                source = source_file["name"]
                entries.extend(self._read_source_file(source, artifact_name))

        return entries

    # pyre-ignore[14]:
    def run_ids(self, parent_run_id: str | None = None) -> Iterable[str]:
        all_sources = []
        root_dir = self._path_builder.path()
        if self.fs.exists(root_dir):
            source_files = self.fs.listdir(root_dir)
            for source_file in source_files:
                if source_file["type"] != "directory":
                    continue
                if parent_run_id:
                    parent_id_file = f"{source_file['name']}/sources/{_encode_torchx_run_id(parent_run_id)}"
                    if not self.fs.exists(parent_id_file):
                        continue
                source = source_file["name"]
                encoded_source_run_id = os.path.basename(source)
                all_sources.append(_decode_torchx_run_id(encoded_source_run_id))
        return all_sources

    def lineage(self, run_id: str) -> Lineage:
        raise NotImplementedError("")

    def __repr__(self) -> str:
        return f"<FsspecTracker: root_path={self._path_builder.path()}>"


def _put_config(key: str, value: str, config: dict[str, Any]) -> None:
    idx = key.find(".")
    if idx < 0:  # not a nested key -> set key = val
        config[key] = value
    elif idx == len(key) - 1:  # key ends with "." -> illegal
        raise ValueError(
            f"Illegal config key `{key}`. Key should not have a `.` suffix"
        )
    else:
        first_key = key[:idx]
        rest_keys = key[idx + 1 :]
        nested_config = config.setdefault(first_key, {})
        _put_config(rest_keys, value, nested_config)


def _read_config(config_file: str) -> Mapping[str, Any]:
    # TODO add support for resource based config
    data: dict[str, Any] = {}
    with fsspec.open(config_file, "rt") as f:
        for line in f:
            if line.startswith("#"):  # skip comments
                continue

            k, sep, v = line.partition("=")
            if k and sep and v:
                _put_config(k.strip(), v.strip(), data)

    return data


def create(config_file: str) -> TrackerBase:
    """
    Entry-point to build Tracker.

    Note the configuration itself expects fsspec URI with at least two entries:
    - protocol: FSSpec protocol (eg. S3, local)
    - root_path: path to use to store the data (in hiearchical fashion)

    In addition any other non-comment entries will be passed to a constructor (eg. authentication data)
    """
    config = _read_config(config_file)
    if "protocol" not in config or "root_path" not in config:
        raise Exception(f"Please specify 'protocol' and 'root_path' in {config_file}")
    protocol = config["protocol"]
    del config["protocol"]

    root = config["root_path"]
    del config["root_path"]

    fs = fsspec.filesystem(protocol, **config)
    tracker = FsspecTracker(fs, root)
    return tracker
