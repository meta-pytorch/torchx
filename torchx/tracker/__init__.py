# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
.. note:: PROTOTYPE, USE AT YOUR OWN RISK, APIs SUBJECT TO CHANGE


Practitioners running ML jobs often need to track information such as:

* Job inputs:
    * configuration
        * model configuration
        * HPO parameters
    * data
        * version
        * sources
* Job results:
    * metrics
    * model location
* Conceptual job groupings


:py:class:`~torchx.tracker.api.AppRun` provides a uniform **interface** as an experiment and artifact tracking solution that
supports wrapping pluggable tracking implementations by providing  :py:class:`~torchx.tracker.api.TrackerBase` adapter
implementation.


Example usage
-------------
Sample `code <https://github.com/meta-pytorch/torchx/blob/main/torchx/examples/apps/tracker/main.py>`__ using tracker API.


Tracker Setup
-------------
To enable tracking it requires:

1. Defining tracker backends (entrypoints/modules and configuration) on launcher side using :doc:`runner.config`
2. Adding entrypoints within a user job using entry_points (`specification`_)

.. _specification: https://packaging.python.org/en/latest/specifications/entry-points/


1. Launcher side configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

User can define any number of tracker backends under **torchx:tracker** section in :doc:`runner.config`, where:
   * Key: is an arbitrary name for the tracker, where the name will be used to configure its properties
        under [tracker:<TRACKER_NAME>]
   * Value: is *entrypoint* or *module* factory method that must be available within user job. The value will be injected into a
        user job and used to construct tracker implementation.

.. code-block:: ini

    [torchx:tracker]
    tracker_name=<entry_point_or_module_factory_method>


Each tracker can be additionally configured (currently limited to `config` parameter) under `[tracker:<TRACKER NAME>]` section:

.. code-block:: ini

    [tracker:<TRACKER NAME>]
    config=configvalue

For example, ~/.torchxconfig may be setup as:

.. code-block:: ini

    [torchx:tracker]
    tracker1=tracker1
    tracker2=backend_2_entry_point
    tracker3=torchx.tracker.mlflow:create_tracker

    [tracker:tracker1]
    config=s3://my_bucket/config.json

    [tracker:tracker3]
    config=my_config.json


2. User job configuration (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Entrypoint value defined in the previous step must be discoverable under `[torchx.tracker]` group and callable within user job
(depending on packaging/distribution mechanism) to create an instance of the :py:class:`~torchx.tracker.api.TrackerBase`.

To accomplish that define entrypoint in the distribution in `entry_points.txt` as:

.. code-block:: ini

    [torchx.tracker]
    entry_point_name=my_module:create_tracker_fn


Acquiring :py:class:`~torchx.tracker.api.AppRun` instance
-------------------------------------------------------------

Use :py:meth:`~torchx.tracker.app_run_from_env`:


    >>> import os; os.environ["TORCHX_JOB_ID"] = "scheduler://session/job_id" # Simulate running job first
    >>> from torchx.tracker import app_run_from_env
    >>> app_run = app_run_from_env()


Reference :py:class:`~torchx.tracker.api.TrackerBase` implementation
--------------------------------------------------------------------
:py:class:`~torchx.tracker.backend.fsspec.FsspecTracker` provides reference implementation of a tracker backend.
GitHub example `directory <https://github.com/meta-pytorch/torchx/blob/main/torchx/examples/apps/tracker/>`__ provides example on how to
configure and use it in user application.


Querying data
-------------
* :py:class:`~torchx.cli.cmd_tracker.CmdTracker` exposes operations available to users at the CLI level:
    * ``torchx tracker list jobs [–parent-run-id RUN_ID]``
    * ``torchx tracker list metadata RUN_ID``
    * ``torchx tracker list artifacts [–artifact ARTIFACT_NAME] RUN_ID``
* Alternatively, backend implementations may expose UI for user consumption.


"""

from .api import AppRun


def app_run_from_env() -> AppRun:
    """
    Syntax sugar for `AppRun.run_from_env` method than can be referenced directly from the module.
    """
    return AppRun.run_from_env()
