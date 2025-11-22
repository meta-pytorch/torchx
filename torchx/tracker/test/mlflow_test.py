# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from unittest import mock

import mlflow
from mlflow.utils.name_utils import _generate_random_name
from torchx.distributed import init_pg, on_rank0_first, rank
from torchx.test.fixtures import DistributedTestCase, IS_CI, IS_MACOS, TestWithTmpDir
from torchx.tracker.mlflow import create_tracker, MLflowTracker


@dataclass
class ModelConfig:
    vocab_size: int = 256035
    hidden_size: int = 1024
    num_hidden_layers: int = 38
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1


@dataclass
class DatasetConfig:
    url: str
    split: str = "70/30"


@dataclass
class Config:
    epochs: int = 100
    lr: float = 0.01
    checkpoint_url: str = "s3://foo/bar"
    use_amp: bool = False
    locales: List[str] = field(default_factory=lambda: ["us", "eu", "fr"])
    empty_list: List[str] = field(default_factory=list)
    empty_map: Dict[str, str] = field(default_factory=dict)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    datasets: List[DatasetConfig] = field(
        default_factory=lambda: [
            DatasetConfig(url="s3://dataset1"),
            DatasetConfig(url="s3://dataset2"),
        ]
    )


class MLflowTrackerTest(TestWithTmpDir):
    def setUp(self) -> None:
        super().setUp()
        self.tracker = MLflowTracker(
            experiment_name=_generate_random_name(),
            tracking_uri=str(self.tmpdir / "experiments"),
            artifact_location=str(self.tmpdir / "artifacts"),
        )

    def tearDown(self) -> None:
        self.tracker.close()
        super().tearDown()

    def test_default_experiment_name(self) -> None:
        tracker = MLflowTracker(tracking_uri=self.tracker.tracking_uri)
        self.assertEqual(
            MLflowTracker.default_experiment_name(), tracker.experiment_name
        )

    def test_ctor_existing_experiment(self) -> None:
        # try to create another tracker and re-use the existing experiment name
        # from the experiment we created in the tracker instantiated in setUp()

        new_tracker = MLflowTracker(
            experiment_name=self.tracker.experiment_name,
            tracking_uri=self.tracker.tracking_uri,
            artifact_location=self.tracker.artifact_location,
        )

        self.assertEqual(self.tracker.experiment_id, new_tracker.experiment_id)
        self.assertEqual(self.tracker.experiment_name, new_tracker.experiment_name)

    def test_get_run(self) -> None:
        run1 = self.tracker.get_run("run1")

        self.assertEqual("run1", run1.info.run_name)
        # OK to call with same run name twice; validate same run id
        self.assertEqual(run1.info.run_id, self.tracker.get_run("run1").info.run_id)

        # NOT OK to call with different run name before close() is called
        # with self.assertRaises(RuntimeError):
        #     self.tracker.get_run("run2")

        # OK to call with different run name after close() is called
        self.tracker.close()
        run2 = self.tracker.get_run("run2")
        self.assertEqual("run2", run2.info.run_name)

    def test_get_run_duplicate_run_name_should_throw(self) -> None:
        client = mlflow.MlflowClient(tracking_uri=self.tracker.tracking_uri)
        client.create_run(self.tracker.experiment_id, run_name="run1")
        client.create_run(self.tracker.experiment_id, run_name="run1")

        with self.assertRaisesRegex(
            RuntimeError, "More than 1 run found for run_name `run1`"
        ):
            self.tracker.get_run("run1")

    def test_get_run_return_existing_run(self) -> None:
        client = mlflow.MlflowClient(tracking_uri=self.tracker.tracking_uri)
        existing_run = client.create_run(self.tracker.experiment_id, run_name="run1")
        returned_run = self.tracker.get_run("run1")

        self.assertEqual(existing_run.info.run_id, returned_run.info.run_id)

    def test_metadata(self) -> None:
        run_name = "test_run_1"
        self.tracker.add_metadata(run_name, foo="bar", bar="baz", baz=1)

        metadata = self.tracker.metadata(run_name)
        self.assertEqual("bar", metadata["foo"])
        self.assertEqual("baz", metadata["bar"])
        self.assertEqual("1", metadata["baz"])

    def test_artifact(self) -> None:
        run_name = "test_run_1"
        num_epochs = 2
        world_size = 4
        model_shard = self.write("model.bin", ["model_shard"])
        manifest = self.write("manifest.json", ["{im: 'a_manifest'}"])

        self.tracker.add_artifact(run_id=run_name, name="manifest", path=str(manifest))

        for epoch in range(num_epochs):
            for rnk in range(world_size):
                self.tracker.add_artifact(
                    run_id=run_name,
                    name=f"checkpoint_{epoch}/{rnk}",
                    path=str(model_shard),
                    metadata={"epoch": epoch, "rank": rnk, "world_size": world_size},
                )

        artifacts = self.tracker.artifacts(run_id=run_name)
        self.assertEqual(9, len(artifacts))

        manifest_artifact = artifacts["manifest"]
        self.assertEqual(
            f"{self.tracker.artifact_location}/manifest/manifest.json",
            manifest_artifact.path,
        )
        self.assertEqual({"mlflow.file_size": 18}, manifest_artifact.metadata)

        for epoch in range(num_epochs):
            for rnk in range(world_size):
                name = f"checkpoint_{epoch}/{rnk}"
                artifact = artifacts[name]
                self.assertEqual(name, artifact.name)
                self.assertEqual(
                    f"{self.tracker.artifact_location}/{name}/model.bin", artifact.path
                )
                self.assertEqual(
                    {
                        "epoch": f"{epoch}",
                        "rank": f"{rnk}",
                        "world_size": f"{world_size}",
                        "mlflow.file_size": 11,
                    },
                    artifact.metadata,
                )

    def test_unsupported_apis(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.tracker.lineage(run_id="ignored")

        with self.assertRaises(NotImplementedError):
            self.tracker.sources(run_id="ignored")

        with self.assertRaises(NotImplementedError):
            self.tracker.add_source(run_id="ignored", source_id="ignored")

    def test_run_ids(self) -> None:
        self.assertEqual([], self.tracker.run_ids())

        for i in range(10):
            self.tracker.get_run(f"run_{i}")
            self.tracker.close()

        self.assertSetEqual(
            {f"run_{i}" for i in range(10)}, set(self.tracker.run_ids())
        )

    def test_create_tracker(self) -> None:
        confdir = str(self.tmpdir)
        self.write(
            ".torchxconfig",
            [
                f"""
[tracker:mlflow]
config = {confdir}

tracking_uri = {str(self.tmpdir / 'experiments')}
artifact_location = {str(self.tmpdir / 'artifacts')}
experiment_name = foobar
        """
            ],
        )

        tracker = create_tracker(config=confdir)
        self.assertEqual(str(self.tmpdir / "experiments"), tracker.tracking_uri)
        self.assertEqual(str(self.tmpdir / "artifacts"), tracker.artifact_location)
        self.assertEqual("foobar", tracker.experiment.name)

    def test_log_params(self) -> None:
        cfg = Config()

        self.tracker.log_params_flat("run1", cfg)
        run = self.tracker.get_run("run1")
        self.maxDiff = None
        self.assertDictEqual(
            {
                "epochs": "100",
                "lr": "0.01",
                "checkpoint_url": "s3://foo/bar",
                "use_amp": "False",
                "locales": "['us', 'eu', 'fr']",
                "empty_list": "[]",
                "empty_map": "{}",
                "model_config.vocab_size": "256035",
                "model_config.hidden_size": "1024",
                "model_config.num_hidden_layers": "38",
                "model_config.num_attention_heads": "16",
                "model_config.intermediate_size": "4096",
                "model_config.hidden_act": "gelu",
                "model_config.hidden_dropout_prob": "0.1",
                "datasets._0.url": "s3://dataset1",
                "datasets._0.split": "70/30",
                "datasets._1.url": "s3://dataset2",
                "datasets._1.split": "70/30",
            },
            run.data.params,
        )


def save_artifact(run_name: str, tmpdir: Path, mlflow_conf: Path) -> None:
    init_pg()
    rankdir = Path(tmpdir) / f"rank_{rank()}"
    rankdir.mkdir()
    rank_artifact = rankdir / "output.txt"
    with open(rank_artifact, "w") as f:
        f.write(f"Hello from rank {rank()}")
    mlflow_tracker = create_tracker(str(mlflow_conf))
    with on_rank0_first():
        try:
            mlflow_tracker.add_artifact(run_name, f"rank_{rank()}", str(rank_artifact))
        except RuntimeError as e:
            print(f"Failed on rank {rank()}\n {e}")
        finally:
            mlflow_tracker.close()


class MlflowTrackerMultiRankTest(DistributedTestCase):
    @unittest.skipIf(IS_CI and IS_MACOS, "no localhost on osx CI")  # pyre-ignore[56]
    def test_multi_artifact(self) -> None:
        mlflow_conf = self.write(
            "mlflow_test.conf",
            content=[
                "[tracker:mlflow]",
                f"tracking_uri = {self.tmpdir / 'mlruns'}",
            ],
        )
        run_name = "test-run"
        with mock.patch.dict(
            os.environ,
            {
                "TORCHX_TRACKING_EXPERIMENT_NAME": "test-experiment",
            },
        ):
            self.run_ddp(world_size=4, fn=save_artifact)(
                run_name, self.tmpdir, mlflow_conf
            )

            mlflow_tracker = create_tracker(str(mlflow_conf))
            artifacts = mlflow_tracker.artifacts(run_name)
            mlflow_tracker.close()
            self.assertEqual(4, len(artifacts))
            self.assertSetEqual(
                {f"rank_{rank}" for rank in range(4)}, set(artifacts.keys())
            )
