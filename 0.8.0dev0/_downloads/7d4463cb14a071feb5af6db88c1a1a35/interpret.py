#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Interpretability Example
=============================================

This is an example TorchX app that uses captum to analyze inputs to for model
interpretability purposes. It consumes the trained model from the trainer app
example and the preprocessed examples from the datapreproc app example. The
output is a series of images with integrated gradient attributions overlayed on
them.

See https://captum.ai/tutorials/CIFAR_TorchVision_Interpret for more info on
using captum.

Usage
---------

Runs this main module as a python process locally. The run below assumes
that the model has been trained using the usage instructions in
``torchx/examples/apps/lightning/train.py``.

.. code:: shell-session

  $ torchx run -s local_cwd utils.python
      --script ./lightning/interpret.py
      --
      --load_path /tmp/torchx/train/last.ckpt
      --output_path /tmp/torchx/interpret

Use an image viewer to visualize the ``*.png`` files generated under the ``output_path``.

.. note:: For local runs with TorchX's  ``utils.python`` built-in is effectively
          equal to running the main module directly (e.g. ``python ./interpret.py``).
          The benefit of using TorchX to launch simple single-process python programs
          is to launch on remote schedulers by swapping out ``-s local_cwd`` to
          a remote scheduler like kubernetes by specifying ``-s kubernetes``.

"""

import argparse
import itertools
import os.path
import sys
import tempfile

import fsspec
import torch
from torchx.examples.apps.lightning.data import (
    create_random_data,
    download_data,
    TinyImageNetDataModule,
)
from torchx.examples.apps.lightning.model import TinyImageNetModel


# ensure data and module are on the path
sys.path.append(".")


# FIXME: captum must be imported after torch otherwise it causes python to crash
if True:
    import numpy as np
    from captum.attr import IntegratedGradients, visualization as viz


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="example TorchX captum app")
    parser.add_argument(
        "--load_path",
        type=str,
        help="checkpoint path to load model weights from",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to load the training data from, if not provided, random dataset will be created",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to place analysis results",
        required=True,
    )

    return parser.parse_args(argv)


def convert_to_rgb(arr: torch.Tensor) -> np.ndarray:  # pyre-ignore[24]
    """
    This converts the image from a torch tensor with size (1, 1, 64, 64) to
    numpy array with size (64, 64, 3).
    """
    out = arr.squeeze().swapaxes(0, 2)
    assert out.shape == (64, 64, 3), "invalid shape produced"
    return out.numpy()


def main(argv: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(argv)

        # Init our model
        model = TinyImageNetModel()

        print(f"loading checkpoint: {args.load_path}...")
        model.load_from_checkpoint(checkpoint_path=args.load_path)

        # Download and setup the data module
        if not args.data_path:
            data_path = os.path.join(tmpdir, "data")
            os.makedirs(data_path)
            create_random_data(data_path)
        else:
            data_path = download_data(args.data_path, tmpdir)
        data = TinyImageNetDataModule(
            data_dir=data_path,
            batch_size=1,
        )

        ig = IntegratedGradients(model)

        data.setup("test")
        dataloader = data.test_dataloader()

        # process first 5 images
        for i, (input, label) in enumerate(itertools.islice(dataloader, 5)):
            print(f"analyzing example {i}")
            # input = input.unsqueeze(dim=0)
            model.zero_grad()
            attr_ig, delta = ig.attribute(
                input,
                target=label,
                baselines=input * 0,
                return_convergence_delta=True,
            )

            if attr_ig.count_nonzero() == 0:
                # Our toy model sometimes has no IG results.
                print("skipping due to zero gradients")
                continue

            fig, axis = viz.visualize_image_attr(
                convert_to_rgb(attr_ig),
                convert_to_rgb(input),
                method="blended_heat_map",
                sign="all",
                show_colorbar=True,
                title="Overlayed Integrated Gradients",
            )
            out_path = os.path.join(args.output_path, f"ig_{i}.png")
            print(f"saving heatmap to {out_path}")
            with fsspec.open(out_path, "wb") as f:
                fig.savefig(f)


if __name__ == "__main__" and "NOTEBOOK" not in globals():
    main(sys.argv[1:])


# sphinx_gallery_thumbnail_path = '_static/img/gallery-app.png'
