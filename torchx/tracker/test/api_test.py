#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing import cast, DefaultDict, Dict, Iterable, Mapping, Optional, Tuple
from unittest import mock, TestCase
from unittest.mock import MagicMock, patch

from torchx.tracker import app_run_from_env
from torchx.tracker.api import (
    _extract_tracker_name_and_config_from_environ,
    AppRun,
    build_trackers,
    ENV_TORCHX_JOB_ID,
    ENV_TORCHX_PARENT_RUN_ID,
    ENV_TORCHX_TRACKERS,
    Lineage,
    tracker_config_env_var_name,
    TrackerArtifact,
    TrackerBase,
    trackers_from_environ,
    TrackerSource,
)
from torchx.tracker.mlflow import MLflowTracker

RunId = str

DEFAULT_SOURCE: str = "__parent__"


class TestTrackerBackend(TrackerBase):
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._artifacts: DefaultDict[
            RunId,
            DefaultDict[str, Tuple[str, Optional[Mapping[str, object]]]],
        ] = defaultdict(lambda: defaultdict())
        self._metdata: DefaultDict[RunId, DefaultDict[str, object]] = defaultdict(
            lambda: defaultdict()
        )

        self._sources: DefaultDict[RunId, Dict[str, str]] = defaultdict(dict)
        self.config_path = config_path

    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._artifacts[run_id][name] = (path, metadata)

    def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]:
        exp_artifacts = self._artifacts[run_id]
        return {
            k: TrackerArtifact(k, path, metadata)
            for k, [path, metadata] in exp_artifacts.items()
        }

    def add_metadata(self, run_id: str, **kwargs: object) -> None:
        self._metdata[run_id].update(kwargs)

    def metadata(self, run_id: str) -> Mapping[str, object]:
        return self._metdata[run_id]

    def add_source(
        self,
        run_id: str,
        source_id: str,
        artifact_name: Optional[str],
    ) -> None:
        if not artifact_name:
            artifact_name = DEFAULT_SOURCE
        self._sources[run_id][artifact_name] = source_id

    def sources(
        self,
        run_id: str,
        artifact_name: Optional[str] = None,
    ) -> Iterable[TrackerSource]:
        source_data = self._sources[run_id]

        sources = []
        for artifact_name, source_id in self._sources[run_id].items():
            if artifact_name == DEFAULT_SOURCE:
                sources.append(TrackerSource(source_id, None))
            else:
                sources.append(TrackerSource(source_id, artifact_name))
        return sources

    def lineage(self, run_id: str) -> Lineage:
        return Lineage()

    def run_ids(self, **kwargs: str) -> Iterable[str]:
        return []


class AppRunApiTest(TestCase):
    def setUp(self) -> None:
        os.environ[ENV_TORCHX_JOB_ID] = "scheduler://session/app_id"
        self.tracker = TestTrackerBackend()
        self.run_id = "run_id"
        self.model_run = AppRun(self.run_id, [self.tracker])

    def tearDown(self) -> None:
        del os.environ[ENV_TORCHX_JOB_ID]

    @mock.patch.dict(
        os.environ,
        {
            ENV_TORCHX_TRACKERS: "tracker1",
            ENV_TORCHX_PARENT_RUN_ID: "parent_run_id",
            ENV_TORCHX_JOB_ID: "run_id",
        },
    )
    def test_env_factory_test(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            app_run = app_run_from_env()
            self.assertEqual(app_run.id, self.run_id)
            trackers = list(app_run.backends)
            self.assertEqual(1, len(trackers))

            tracker = trackers[0]
            self.assertEqual(TestTrackerBackend, type(tracker))

            sources = list(tracker.sources("run_id"))
            self.assertEqual(1, len(sources))

            tracker_source = sources[0]
            self.assertEqual("parent_run_id", tracker_source.source_run_id)

    def test_get_exierment_run_id(self) -> None:
        self.assertEqual(self.run_id, self.model_run.job_id())

    def test_add_artifact(self) -> None:
        self.model_run.add_artifact("output", "/directory")
        artifacts = self.tracker.artifacts(self.run_id)
        self.assertEqual("/directory", artifacts["output"].path)
        self.assertIsNone(artifacts["output"].metadata)

    def test_add_artifact_with_metadata(self) -> None:
        self.model_run.add_artifact("output", "/directory", {"date": "2020/01/01"})
        artifacts = self.tracker.artifacts(self.run_id)
        metadata = artifacts["output"].metadata
        self.assertIsNotNone(metadata)
        if metadata:
            self.assertEqual("2020/01/01", metadata["date"])

    def test_add_source(self) -> None:
        self.model_run.add_source("parent_model_run")
        sources = self.model_run.sources()
        self.assertEqual(1, len(list(sources)))
        source = list(sources)[0]
        self.assertEqual(source.parent.id, "parent_model_run")
        self.assertEqual(source.parent.backends, self.model_run.backends)

    def test_add_metadata(self) -> None:
        self.model_run.add_metadata(lr=0.01, bs=25)

        metadata = self.tracker.metadata(self.run_id)
        self.assertEqual(metadata, {"lr": 0.01, "bs": 25})

    def test_add_metadata_should_update_values(self) -> None:
        self.model_run.add_metadata(lr=0.01, bs=25)
        self.model_run.add_metadata(bs=10)

        metadata = self.tracker.metadata(self.run_id)
        self.assertEqual(metadata, {"lr": 0.01, "bs": 10})

    def test_lineage(self) -> None:
        self.tracker.lineage(self.run_id)

    def test_run_ids(self) -> None:
        self.tracker.run_ids()


def tracker_factory(config: Optional[str] = None) -> TrackerBase:
    return TestTrackerBackend(config)


class TrackerFactoryMethodsTest(TestCase):
    def test_config_env_var_name(self) -> None:
        value = tracker_config_env_var_name("tracker1")
        self.assertEqual(value, "TORCHX_TRACKER_TRACKER1_CONFIG")

    def test_trackers_from_environ_with_no_trackers_specified(self) -> None:
        self.assertEqual(len(list(trackers_from_environ())), 0)

    @mock.patch.dict(os.environ, {ENV_TORCHX_TRACKERS: "tracker1"})
    def test_tracker_from_environ(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            trackers = trackers_from_environ()
            self.assertEqual(1, len(list(trackers)))
            self.assertEqual(TestTrackerBackend, type(trackers[0]))

    @mock.patch.dict(
        os.environ,
        {
            ENV_TORCHX_TRACKERS: "tracker1",
            tracker_config_env_var_name("tracker1"): "myconfig.txt",
        },
    )
    def test_tracker_from_environ_with_config_setting(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            trackers = trackers_from_environ()
            tracker = cast(TestTrackerBackend, list(trackers)[0])
            self.assertEqual("myconfig.txt", tracker.config_path)

    def test_tracker_from_environ_that_wasnt_setup(self) -> None:
        trackers = trackers_from_environ()
        self.assertEqual(len(list(trackers)), 0)

    @mock.patch.dict(os.environ, {ENV_TORCHX_TRACKERS: "tracker1"})
    def test_tracker_from_environ_with_missing_entrypoint(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={},
        ):
            trackers = trackers_from_environ()
            self.assertEqual(0, len(list(trackers)))

    def test_extract_tracker_name_and_config_from_environ_undefined(self) -> None:
        self.assertTrue(len(_extract_tracker_name_and_config_from_environ()) == 0)

    @mock.patch.dict(os.environ, {ENV_TORCHX_TRACKERS: "tracker1"})
    def test_extract_tracker_name_and_config_from_environ_with_just_name(self) -> None:
        entries = _extract_tracker_name_and_config_from_environ()
        self.assertEqual(entries, {"tracker1": None})

    @mock.patch.dict(
        os.environ,
        {
            ENV_TORCHX_TRACKERS: "tracker1",
            "TORCHX_TRACKER_TRACKER1_CONFIG": "myconfig.txt",
        },
    )
    def test_extract_tracker_name_and_config_from_environ_with_name_and_config(
        self,
    ) -> None:
        entries = _extract_tracker_name_and_config_from_environ()
        self.assertEqual(entries, {"tracker1": "myconfig.txt"})

    def test_build_trackers_with_no_trackers_defined(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            no_tracker_names = {}
            trackers = build_trackers(no_tracker_names)
            self.assertEqual(0, len(list(trackers)))

    def test_build_trackers_with_no_entrypoints_group_defined(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value=None,
        ):
            tracker_names = {"tracker1": "myconfig.txt"}
            trackers = build_trackers(tracker_names)
            self.assertEqual(0, len(list(trackers)))

    def test_build_trackers_with_module(self) -> None:
        module = MagicMock()
        module.return_value = MagicMock(spec=MLflowTracker)
        with patch("torchx.tracker.api.load_group", return_value=None) and patch(
            "torchx.tracker.api.load_module",
            return_value=module,
        ):
            tracker_names = {
                "torchx.tracker.mlflow:create_tracker": (config := "myconfig.txt")
            }
            trackers = build_trackers(tracker_names)
            trackers = list(trackers)
            self.assertEqual(1, len(trackers))
            tracker = trackers[0]
            self.assertIsInstance(tracker, MLflowTracker)
            module.assert_called_once_with(config)

    def test_build_trackers(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            tracker_names = {"tracker1": "myconfig.txt"}
            trackers = build_trackers(tracker_names)
            trackers = list(trackers)
            self.assertEqual(1, len(trackers))
            tracker = trackers[0]
            self.assertEqual(tracker.config_path, "myconfig.txt")  # pyre-ignore[16]

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_run_from_env_not_launched_with_torchx(self) -> None:
        # asserts that an empty AppRun (with no trackers and job_id defaulted to program name)
        # is created if not launched from torchx
        # makes it possible to use `torchx.tracker.app_run_from_env()` without being tightly coupled with
        # the torchx launcher

        AppRun.run_from_env.cache_clear()
        apprun = AppRun.run_from_env()
        self.assertEqual("<UNDEFINED>", apprun.job_id())
        # no backends configured if not launched via torchx
        self.assertEqual([], list(apprun.backends))
