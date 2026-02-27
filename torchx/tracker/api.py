# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping

from torchx.util.entrypoints import load_group
from torchx.util.modules import load_module

logger: logging.Logger = logging.getLogger(__name__)

ENV_TORCHX_TRACKERS = "TORCHX_TRACKERS"
ENV_TORCHX_PARENT_RUN_ID = "TORCHX_PARENT_RUN_ID"
ENV_TORCHX_JOB_ID = "TORCHX_JOB_ID"


@dataclass
class TrackerSource:
    """A source link at the backend tracker level.

    ``source_run_id`` is a TorchX handle or external entity ID.
    ``artifact_name`` classifies the relationship (used for filtering).
    """

    source_run_id: str
    artifact_name: str | None


@dataclass
class TrackerArtifact:
    """An artifact stored by a backend tracker (name, path, and optional metadata)."""

    name: str
    path: str
    metadata: Mapping[str, object] | None


@dataclass
class AppRunTrackableSource:
    """A source link at the user API level (wraps :py:class:`AppRun` parent)."""

    parent: AppRun
    artifact_name: str | None


class Lineage: ...


class TrackerBase(ABC):
    """Abstract base for tracker backend implementations.

    .. warning::

        This API is experimental and may change significantly.
    """

    @abstractmethod
    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Add an artifact with the given name, path, and optional metadata."""
        ...

    @abstractmethod
    def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]:
        """Return all artifacts for the given run."""
        ...

    @abstractmethod
    def add_metadata(self, run_id: str, **kwargs: object) -> None:
        """Store arbitrary key-value metadata for the given run."""
        ...

    @abstractmethod
    def metadata(self, run_id: str) -> Mapping[str, object]:
        """Return metadata for the given run."""
        ...

    @abstractmethod
    def add_source(
        self,
        run_id: str,
        source_id: str,
        artifact_name: str | None = None,
    ) -> None:
        """Link a source run (lineage) to the given run."""
        ...

    @abstractmethod
    def sources(
        self,
        run_id: str,
        artifact_name: str | None = None,
    ) -> Iterable[TrackerSource]:
        """Return sources for the given run, optionally filtered by ``artifact_name``."""
        ...

    @abstractmethod
    def lineage(self, run_id: str) -> Lineage:
        """Return full lineage (parents and consumers) for the given run."""
        ...

    @abstractmethod
    def run_ids(self, **kwargs: str) -> Iterable[str]:
        """Return run IDs, optionally filtered by keyword arguments."""
        ...


def tracker_config_env_var_name(entrypoint_key: str) -> str:
    """Return the ``TORCHX_TRACKER_<NAME>_CONFIG`` env var name for a tracker."""
    return f"TORCHX_TRACKER_{entrypoint_key.upper()}_CONFIG"


def _extract_tracker_name_and_config_from_environ() -> Mapping[str, str | None]:
    if ENV_TORCHX_TRACKERS not in os.environ:
        logger.info("No trackers were configured, skipping setup.")
        return {}

    tracker_backend_entrypoints = os.environ[ENV_TORCHX_TRACKERS]
    logger.info(f"Trackers: {ENV_TORCHX_TRACKERS}={tracker_backend_entrypoints}")

    entries = {}
    for entrypoint_key in tracker_backend_entrypoints.split(","):
        config = None
        config_env_name = tracker_config_env_var_name(entrypoint_key)
        if config_env_name in os.environ:
            config = os.environ[config_env_name]
        entries[entrypoint_key] = config

    return entries


def build_trackers(
    factory_and_config: Mapping[str, str | None],
) -> Iterable[TrackerBase]:
    trackers = []

    entrypoint_factories = load_group("torchx.tracker") or {}
    if not entrypoint_factories:
        logger.warning("No 'torchx.tracker' entry_points are defined.")

    for factory_name, config in factory_and_config.items():
        factory = entrypoint_factories.get(factory_name) or load_module(factory_name)
        if not factory:
            logger.warning(
                f"No tracker factory `{factory_name}` found in entry_points or modules. See https://meta-pytorch.org/torchx/main/tracker.html#module-torchx.tracker"
            )
            continue
        if config:
            logger.info(f"Tracker config found for `{factory_name}` as `{config}`")
        else:
            logger.info(f"No tracker config specified for `{factory_name}`")
        tracker = factory(config)
        trackers.append(tracker)
    return trackers


def trackers_from_environ() -> Iterable[TrackerBase]:
    """Build trackers from ``TORCHX_TRACKERS`` env var (comma-separated entry-point keys).

    Per-tracker config is read from ``TORCHX_TRACKER_<NAME>_CONFIG`` env vars.
    Entry-point factories must be importable at runtime (runs in user-job space).
    """

    entrypoint_and_config = _extract_tracker_name_and_config_from_environ()
    if entrypoint_and_config:
        return build_trackers(entrypoint_and_config)
    return []


@dataclass
class AppRun:
    """Job-level tracker API that delegates to one or more :py:class:`TrackerBase` backends.

    .. warning::

        This API is experimental and may change significantly.

    .. doctest::

        >>> from torchx.tracker.api import AppRun
        >>> run = AppRun(id="my_job_123", backends=[])
        >>> run.add_metadata(lr=0.01, epochs=10)  # no-op with empty backends
        >>> run.job_id()
        'my_job_123'

    """

    id: str
    backends: Iterable[TrackerBase]

    @staticmethod
    @lru_cache(maxsize=1)  # noqa: B019
    def run_from_env() -> AppRun:
        """Create a singleton :py:class:`AppRun` from environment variables.

        Reads ``TORCHX_JOB_ID`` and ``TORCHX_TRACKERS`` (set by the torchx runner).
        Returns a cached singleton so all callers share the same tracker backends.

        .. note::

            When not launched via torchx, returns an empty ``AppRun`` with
            ``job_id="<UNDEFINED>"`` and no backends (write methods become no-ops).

        .. doctest::

            >>> from torchx.tracker.api import AppRun
            >>> apprun = AppRun.run_from_env()
            >>> apprun.add_metadata(md_1="foo", md_2="bar")

        """

        torchx_job_id = os.getenv(ENV_TORCHX_JOB_ID, default="<UNDEFINED>")

        trackers = trackers_from_environ()
        if ENV_TORCHX_PARENT_RUN_ID in os.environ:
            parent_run_id = os.environ[ENV_TORCHX_PARENT_RUN_ID]
            logger.info(f"Tracker parent run ID: '{parent_run_id}'")
            for tracker in trackers:
                tracker.add_source(torchx_job_id, parent_run_id, artifact_name=None)

        return AppRun(id=torchx_job_id, backends=trackers)

    def add_metadata(self, **kwargs: object) -> None:
        """Store key-value metadata for this run."""
        for backend in self.backends:
            backend.add_metadata(self.id, **kwargs)

    def add_artifact(
        self, name: str, path: str, metadata: Mapping[str, object] | None = None
    ) -> None:
        """Store an artifact (name, path, optional metadata) for this run."""
        for backend in self.backends:
            backend.add_artifact(self.id, name, path, metadata)

    def job_id(self) -> str:
        """Return the run ID."""
        return self.id

    def add_source(self, source_id: str, artifact_name: str | None = None) -> None:
        """Link a source (TorchX run or external entity) to this run for lineage tracking."""
        for backend in self.backends:
            backend.add_source(self.id, source_id, artifact_name)

    def sources(self) -> Iterable[AppRunTrackableSource]:
        """Return source links for this run (queries the first backend)."""
        model_run_sources = []
        if self.backends:
            backend = next(iter(self.backends))
            sources = backend.sources(self.id)
            for source in sources:
                parent = AppRun(source.source_run_id, backends=self.backends)
                model_run_source = AppRunTrackableSource(parent, source.artifact_name)
                model_run_sources.append(model_run_source)

        return model_run_sources

    def children(self) -> Iterable[AppRun]: ...
