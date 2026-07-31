# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
For metrics we recommend using Tensorboard to log metrics directly to cloud
storage along side your model. As the model trains you can launch a
tensorboard instance locally to monitor your model progress:

.. code-block:: shell-session

 $ tensorboard --log-dir provider://path/to/logs

Or you can use the :py:meth:`torchx.components.metrics.tensorboard` component as
part of your pipeline.

Reference
----------------

* PyTorch Tensorboard Tutorial https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
* PyTorch Lightning Loggers https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html

"""

import torchx
import torchx.specs as specs


def tensorboard(
    logdir: str,
    image: str = torchx.IMAGE,
    timeout: float = 60 * 60,  # 1 hour
    port: int = 6006,
    start_on_file: str = "",
    exit_on_file: str = "",
) -> specs.AppDef:
    """
    This component runs a Tensorboard server which will render the logs
    specified by logdir.

    Since Tensorboard runs as a service you need to specify the termination
    conditions. This consists of a timeout as well as an optional
    ``exit_on_file`` which will cause the service to quit when that path is
    created.

    The files are periodically polled for existence via fsspec and will trigger
    the corresponding behavior when created.

    Args:
        logdir: fsspec path to the Tensorboard logs
        image: image to use
        timeout: maximum time to run before exiting (seconds)
        start_on_file: start the server when the fsspec path is created
        exit_on_file: shutdown the server when the fsspec path is created
    """

    return specs.AppDef(
        name="tensorboard",
        roles=[
            specs.Role(
                name="tensorboard",
                image=image,
                entrypoint="python",
                args=[
                    "-m",
                    "torchx.apps.utils.process_monitor",
                    "--timeout",
                    str(timeout),
                    "--start_on_file",
                    start_on_file,
                    "--exit_on_file",
                    exit_on_file,
                    "--",
                    "tensorboard",
                    "--bind_all",
                    "--port",
                    str(port),
                    "--logdir",
                    logdir,
                ],
                port_map={"http": port},
            )
        ],
    )
