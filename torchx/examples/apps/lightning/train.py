#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Trainer Example
=============================================

This is an example TorchX app that uses PyTorch Lightning to train a model.

This app only uses standard OSS libraries and has no runtime torchx
dependencies. For saving and loading data and models it uses fsspec which makes
the app agnostic to the environment it's running in.

Usage
---------

To run the trainer locally as a ddp application with 1 node and 2 workers-per-node (world size = 2):

.. code:: shell-session

  $ torchx run -s local_cwd dist.ddp
     -j 1x2
     --script ./lightning/train.py
     --
     --epochs=1
     --output_path=/tmp/torchx/train
     --log_path=/tmp/torchx/logs
     --skip_export

.. note:: ``--`` is used to delimit between component (``dist.ddp``) and
          application arguments.

Use the ``--help`` option to see the full list of application options:

.. code:: shell-session

  $ torchx run -s local_cwd dist.ddp -j 1x1 --script ./lightning/train.py -- --help

Which is effectively the same as ``./train.py --help``. To run on a remote scheduler,
specify the scheduler with the ``-s`` option. Depending on the type of remote scheduler
you may have to pass additional scheduler cfgs with the ``-cfg`` option.
See :ref:`Quickstart:Remote Schedulers` for more details.


"""

import argparse
import os
import sys
import tempfile

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.distributed.elastic.multiprocessing import errors
from torchx.examples.apps.lightning.data import (
    create_random_data,
    download_data,
    TinyImageNetDataModule,
)
from torchx.examples.apps.lightning.model import (
    export_inference_model,
    TinyImageNetModel,
)
from torchx.examples.apps.lightning.profiler import SimpleLoggingProfiler


# ensure data and module are on the path
sys.path.append(".")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch lightning TorchX example app")
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=32,
        help="number of samples in the dataset",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to load the training data from, if not provided, random data will be generated",
    )
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--load_path", type=str, help="checkpoint path to load from")
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to place checkpoints and model outputs, if not specified, checkpoints are not saved",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="path to place the tensorboard logs",
        default="/tmp",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="the MLP hidden layers and sizes, used for neural architecture search",
    )
    return parser.parse_args(argv)


def get_model_checkpoint(args: argparse.Namespace) -> ModelCheckpoint | None:
    if not args.output_path:
        return None
    # Note: It is important that each rank behaves the same.
    # All of the ranks, or none of them should return ModelCheckpoint
    # Otherwise, there will be deadlock for distributed training
    return ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.output_path,
        save_last=True,
    )


@errors.record
def main(argv: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(argv)

        # Init our model
        model = TinyImageNetModel(args.layers)
        print(model)

        # Download and setup the data module
        if not args.data_path:
            data_path = os.path.join(tmpdir, "data")
            os.makedirs(data_path)
            create_random_data(data_path, args.num_samples)
        else:
            data_path = download_data(args.data_path, tmpdir)

        data = TinyImageNetDataModule(
            data_dir=data_path,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
        )

        # Setup model checkpointing
        checkpoint_callback = get_model_checkpoint(args)
        callbacks = []
        if checkpoint_callback:
            callbacks.append(checkpoint_callback)
        if args.load_path:
            print(f"loading checkpoint: {args.load_path}...")
            model.load_from_checkpoint(checkpoint_path=args.load_path)

        logger = TensorBoardLogger(
            save_dir=args.log_path, version=1, name="lightning_logs"
        )
        # Initialize a trainer
        trainer = pl.Trainer(
            num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
            strategy="ddp",
            logger=logger,
            max_epochs=args.epochs,
            callbacks=callbacks,
            profiler=SimpleLoggingProfiler(logger),
        )

        # Train the model ⚡
        trainer.fit(model, data)
        print(
            f"train acc: {model.train_acc.compute()}, val acc: {model.val_acc.compute()}"
        )

        rank = int(os.environ.get("RANK", 0))
        if rank == 0 and not args.skip_export and args.output_path:
            # Export the inference model
            export_inference_model(model, args.output_path, tmpdir)


if __name__ == "__main__" and "NOTEBOOK" not in globals():
    main(sys.argv[1:])


# sphinx_gallery_thumbnail_path = '_static/img/gallery-app.png'
