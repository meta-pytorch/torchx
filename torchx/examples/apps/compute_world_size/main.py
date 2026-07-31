# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


"""
Compute World Size Example
============================

This is a minimal "hello world" style  example application that uses
PyTorch Distributed to compute the world size. It does not do ML training
but it does initialize process groups and performs a single collective operation (all_reduce)
which is enough to validate the infrastructure and scheduler setup.

As simple as this application is, the actual ``compute_world_size()`` function is
split into a separate submodule (``.module.util.compute_world_size``) to double
as a E2E test for workspace patching logic, which typically diff-patches a full project
directory rather than a single file. This application also uses `Hydra <https://hydra.cc/docs/intro/>`_
configs as an expository example of how to use Hydra configs in an application that launches with TorchX.

Run it with the ``dist.ddp`` builtin component to use as a validation application
to ensure that the stack has been setup properly for more serious distributed training jobs.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from torchx.examples.apps.compute_world_size.module.util import compute_world_size


@record
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.main.throws:
        raise RuntimeError(f"raising error because cfg.main.throws={cfg.main.throws}")
    compute_world_size(cfg)


if __name__ == "__main__":
    # use compose API to make this compatible with ipython notebooks
    # need to initialize the config directory as a module to make it
    # not depends on rel path (PWD) or abs path (torchx install dir)
    # see: https://hydra.cc/docs/advanced/jupyter_notebooks/
    with hydra.initialize_config_module(
        config_module="torchx.examples.apps.compute_world_size.config"
    ):
        cfg: DictConfig = hydra.compose(config_name="defaults")
        run(cfg)
