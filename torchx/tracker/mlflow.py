# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
import socket
from getpass import getuser
from logging import getLogger, Logger
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Iterable, Mapping, Sequence

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from torchx.distributed import on_rank0_first
from torchx.runner.config import get_configs
from torchx.tracker.api import (
    ENV_TORCHX_JOB_ID,
    Lineage,
    TrackerArtifact,
    TrackerBase,
    TrackerSource,
)

log: Logger = getLogger(__name__)
TAG_ARTIFACT_MD_PREFIX = "torchx.artifact.metadata"


class MLflowTracker(TrackerBase):
    """
    An implementation of a ``Tracker`` that uses mlflow as the backend.
    Don't forget to call the ``close()`` method for orderly shutdown.
    This ensures that the run state in mlflow is properly marked as ``FINISHED``,
    otherwise the run will remain in ``UNFINISHED`` status.

    .. important::
        TorchX's run_id is used as mlflow's run_name! The run_id in TorchX
        is the job name. The job name in TorchX is made unique by adding
        a short random hash to the user-provided job name prefix. This is
        done because certain job schedulers supported by TorchX requires
        that the job name on the submitted job definition is globally unique
        (rather than the scheduler returning a unique job id as the return result
        of the job submission API).

    .. warning::
        APIs on this class may only be called with the same ``run_name`.
        Typically the user does not have to worry about manually setting
        the run_name as it is picked up by default from the environment variable
        ``TORCHX_APP_NAME``.

    """

    def __init__(
        self,
        experiment_name: str | None = None,
        tracking_uri: str = f"file://{Path(gettempdir()) / 'torchx' / 'mlruns'}",
        artifact_location: str | None = None,
    ) -> None:
        if experiment_name is None:
            experiment_name = self.default_experiment_name()

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        log.info(
            f"MLflow tracking_uri={tracking_uri}, artifact_location={artifact_location}"
        )

        # mlflow uses ids (experiment_id and run_id) rather than names
        # to uniquely identify experiment runs. But in torchx tracker
        # we use the job name (TORCHX_JOB_ID which is unique in torchx)
        # to uniquely identify experiment runs. So always look up experiments and runs
        # by their names, then use the mlflow returned ids.
        # Since this tracker may be used in a distributed setting (SPMD),
        # guard against races when querying mlflow by running on rank 0 first
        with on_rank0_first():
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment:
                self.experiment: Experiment = existing_experiment
                log.info(
                    f"Found existing experiment `{experiment_name}` (id={self.experiment_id})"
                )
            else:
                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location,
                )
                self.experiment = mlflow.get_experiment(experiment_id)
                log.info(
                    f"Created new experiment `{experiment_name}` (id={experiment_id})"
                )

            # torchx gets the run name from TORCHX_JOB_ID env var
            # so this env var will exist if the job is launched with torchx
            # if so, then prime the run here so that rank0 creates the run first
            # and the rest of the ranks can point to the existing run
            run_name = os.getenv(ENV_TORCHX_JOB_ID)
            if run_name:
                run = self.get_run(run_name)
                log.info(f"Primed run `{run.info.run_name}` ({run.info.run_id})")

    @staticmethod
    def default_experiment_name() -> str:
        return f"default-experiment/{getuser()}/{socket.getfqdn()}"

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def experiment_name(self) -> str:
        return self.experiment.name

    @property
    def artifact_location(self) -> str:
        return self.experiment.artifact_location

    def get_run(self, run_name: str) -> Run:
        """
        Gets mlflow's ``Run`` object for the given ``run_name`` in the current experiment.
        If no such run exists, this method creates a new run under
        this experiment and starts the run so that subsequent calls to
        mlflow logs metadata, metrics, artifacts to the newly created run.

        .. warning::
            This method should only be called with the same run_name!
            This is because of the way mlflow
            APIs work is by setting an "active run" for which subsequent
            mlflow logging APIs are made against the current active run
            in the stack. If you call ``mlflow.start_run()`` directly
            or pass different run names, then you may be logging into two different
            mlflow runs from the same job!

        Args:
            run_name: equal to torchx's run_id

        Returns: mlflow's ``Run`` object for the ``run_name``

        """

        active_run = mlflow.active_run()

        if active_run is None:
            search_result = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                output_format="list",
                filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            )
            if not search_result:
                return mlflow.start_run(
                    experiment_id=self.experiment_id,
                    run_name=run_name,
                )
            elif len(search_result) == 1:
                return mlflow.start_run(
                    experiment_id=self.experiment_id,
                    run_id=search_result[0].info.run_id,
                )
            else:  # len(search_result) > 1
                raise RuntimeError(
                    f"More than 1 run found for run_name `{run_name}` in experiment `{self.experiment_name}`."
                    f" Did you manually create runs with the same name under this experiment?"
                    f" Remove duplicate run names and try again"
                )
        else:
            # need to query mlflow again so that the run reflects any newly written logs
            return mlflow.get_run(active_run.info.run_id)

    def get_run_id(self, run_name: str) -> str:
        """
        Gets the mlflow run's run_id for the given ``run_name`` and additionally sets
        this run as the active run. Hence this method has a side-effect where all subsequent
        calls to mlflow log APIs are against the run for the given ``run_name``.
        """
        return self.get_run(run_name).info.run_id

    def close(self) -> None:
        mlflow.end_run()

    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.get_run(run_id)
        # stores the artifact in {artifact_location}/{name} (e.g. s3://bucket/prefix/{name})
        log.info(f"Writing artifact: {name}, path: {path}")
        mlflow.log_artifact(local_path=path, artifact_path=name)

        # add artifact metadata with torchx.artifact_metadata.{name}.* tag prefix
        if metadata:
            mlflow.set_tags(
                tags={
                    f"{TAG_ARTIFACT_MD_PREFIX}.{name}.{k}": v
                    for k, v in metadata.items()
                }
            )

    def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]:
        artifacts: dict[str, TrackerArtifact] = {}
        mlflow_client: MlflowClient = MlflowClient(self.tracking_uri)

        def get_artifacts(path: str | None = None) -> None:
            for artifact_info in mlflow_client.list_artifacts(
                self.get_run(run_id).info.run_id, path=path
            ):
                if artifact_info.is_dir:
                    get_artifacts(path=artifact_info.path)
                else:
                    # we stored the artifact using the name as the path
                    # so path should never be `None` when we get to this point
                    # (e.g. the root of `artifact_location` will only have directories
                    # where the directory names are the artifact names
                    name = path or "<SHOULD_NOT_HAPPEN>"

                    # artifact metadata is stored as run tags with `torchx.artifact.metadata.*` prefix
                    tag_prefix = f"{TAG_ARTIFACT_MD_PREFIX}.{name}."
                    metadata = {
                        # k.removeprefix() only avail in python 3.9+
                        k[len(tag_prefix) :]: v
                        for k, v in self.metadata(run_id).items()
                        if k.startswith(tag_prefix)
                    }

                    # add some additional metadata about the artifact
                    metadata["mlflow.file_size"] = artifact_info.file_size

                    artifacts[name] = TrackerArtifact(
                        name=name,
                        path=f"{self.artifact_location}/{artifact_info.path}",
                        metadata=metadata,
                    )

        get_artifacts()
        return artifacts

    def add_metadata(self, run_id: str, **kwargs: object) -> None:
        self.get_run(run_id)
        mlflow.set_tags(tags={k: v for k, v in kwargs.items()})

    def metadata(self, run_id: str) -> Mapping[str, object]:
        return self.get_run(run_id).data.tags

    def run_ids(self, **kwargs: str) -> Iterable[str]:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id], output_format="list"
        )
        return [r.info.run_name for r in runs]

    def log_params_flat(
        self, run_name: str, cfg: Any, key: str = ""  # pyre-ignore[2]
    ) -> None:
        """
        Designed to be primarily used with hydra-style config objects (e.g. dataclasses),
        logs the given ``cfg``, which is one of: ``@dataclass``, ``Sequence`` (e.g. ``list``, ``tuple``, ``set``),
        or ``Mapping`` (e.g. ``dict``), where the fields of ``cfg`` are flattened recursively and logged as
        the the run's ``Parameter`` in mlflow.

        For example if ``cfg`` is:

        .. code-block:: python

            @dataclass
            class Config2:
                foo: str = "bar"

            @dataclass
            class Config:
                i: int = 1
                f: float = 2.1
                s: str = "string"
                l: List[str] = field(default_factory=lambda :["a", "b", "c"])
                cfg_list = List[Config2] = field(default_factory=lambda : [Config2(foo="hello"), Config2(foo="world")])
                cfg2: Config2 = Config2()


        Then this function logs the following parameters

        .. code-block::

            i: "1"
            f: "2.1"
            s: "string"
            l: ["a", "b", "c"]
            cfg_list._0.foo = "hello"
            cfg_list._1.foo = "hello"
            cfg2.foo = "bar"

        As shown above, primitive sequence containers are logged directly (e.g. ``l: ["a", "b", "c"]``)
        whereas nested sequence containers will be logged per element where the key is suffixed with
        ``_{INDEX}`` (e.g. ``cfg_list._0.foo = "hello"``).

        """
        if dataclasses.is_dataclass(cfg):
            cfg = dataclasses.asdict(cfg)
        self.get_run(run_name)

        def is_primitive(v: Any) -> bool:  # pyre-ignore[2]
            return isinstance(v, (str, int, float, bool))

        key_prefix = f"{key}." if key else ""

        if not cfg:
            # empty container; log as is
            mlflow.log_param(key, cfg)
        else:
            # non-empty container; check types
            if isinstance(cfg, (Sequence, set)):
                # assume list/set elements are homogeneous
                # need only check first element for type
                elem = next(iter(cfg))
                if is_primitive(elem):
                    mlflow.log_param(key, cfg)
                else:
                    for i, e in enumerate(cfg):
                        self.log_params_flat(run_name, e, f"{key_prefix}_{i}")
            elif isinstance(cfg, Mapping):
                for k, v in cfg.items():
                    if is_primitive(v):
                        mlflow.log_param(f"{key_prefix}{k}", v)
                    else:
                        self.log_params_flat(run_name, v, f"{key_prefix}{k}")

    def add_source(
        self, run_id: str, source_id: str, artifact_name: str | None = None
    ) -> None:
        raise NotImplementedError(
            f"Job's tracker sources is currently unsupported for {self.__class__.__qualname__}"
        )

    def sources(
        self, run_id: str, artifact_name: str | None = None
    ) -> Iterable[TrackerSource]:
        raise NotImplementedError(
            f"Job's tracker sources is currently unsupported for {self.__class__.__qualname__}"
        )

    def lineage(self, run_id: str) -> Lineage:
        raise NotImplementedError(
            f"Job's lineage is currently unsupported for {self.__class__.__qualname__}"
        )


def create_tracker(config: str) -> MLflowTracker:
    ctor_args = get_configs(
        prefix="tracker",
        name="mlflow",
        dirs=[config],
    )

    # remove "config" key since that one is reserved for torchx.tracker usage
    ctor_args.pop("config", None)

    # pass configs read from .torchxconfig [tracker:mlflow] section as kwargs
    # get the experiment name from an env var (set in torchx.components.dist:spmd)
    # if no such env var exists, then default the experiment_name to the one
    # specified in .torchxconfig
    return MLflowTracker(
        experiment_name=os.getenv(
            "TORCHX_TRACKING_EXPERIMENT_NAME",
            default=ctor_args.pop("experiment_name", None),
        ),
        **ctor_args,
    )
