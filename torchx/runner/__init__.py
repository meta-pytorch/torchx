#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Submits :py:class:`~torchx.specs.AppDef` jobs to :ref:`schedulers<Schedulers>`.

The runner takes an ``AppDef`` (the result of evaluating a component function)
along with a scheduler name and run config, and submits it as a job.

.. code-block:: python

    from torchx.runner import get_runner

    with get_runner() as runner:
        app_handle = runner.run(app, scheduler="kubernetes", cfg=cfg)
        status = runner.status(app_handle)
        print(status)

"""

from torchx.runner.api import get_runner, Runner  # noqa: F401 F403
