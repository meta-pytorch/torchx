torchx.tracker
==============

.. tip::

   Trackers record artifacts, metadata, and lineage for training runs. Use
   :py:class:`~torchx.tracker.api.AppRun` inside your job and register custom
   backends via :term:`entry points <Entry Point>`.

**Prerequisites:** :doc:`basics` (core concepts). For registering custom
tracker backends, see :ref:`registering-custom-trackers` in the
:doc:`advanced` guide.

Overview & Usage
------------------

.. automodule:: torchx.tracker
.. currentmodule:: torchx.tracker

Trackers operate at two levels:

* **Backend level** (:py:class:`~torchx.tracker.api.TrackerBase`) -- the storage
  implementation. TorchX ships with
  :py:class:`~torchx.tracker.backend.fsspec.FsspecTracker` (filesystem-based)
  and :py:class:`~torchx.tracker.mlflow.MLflowTracker`. You can implement your
  own backend.
* **Job level** (:py:class:`~torchx.tracker.api.AppRun`) -- the user-facing API
  that delegates to one or more ``TrackerBase`` backends. ``AppRun`` is
  constructed automatically from environment variables set by the TorchX
  runner (``TORCHX_JOB_ID``, ``TORCHX_TRACKERS``).

**Typical usage inside a training job:**

.. code-block:: python

   from torchx.tracker.api import AppRun

   # Singleton created from TORCHX_JOB_ID and TORCHX_TRACKERS env vars
   run = AppRun.run_from_env()

   # Store metadata (key-value pairs)
   run.add_metadata(lr=0.001, epochs=10, model="resnet50")

   # Store an artifact (named path + optional metadata)
   run.add_artifact("checkpoint", "s3://bucket/checkpoints/epoch_10.pt")

   # Link a parent run for lineage tracking
   run.add_source("local_cwd://torchx/parent_job_123")

Implementing a Custom Tracker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subclass :py:class:`~torchx.tracker.api.TrackerBase` and implement its eight
abstract methods. Then provide a factory function and register it as a
``torchx.tracker`` entry point. See
:ref:`registering-custom-trackers` for the full walkthrough.

Testing Your Tracker
""""""""""""""""""""""

Study ``torchx/tracker/test/api_test.py`` for a complete in-memory example. A
minimal test writes metadata and artifacts, then reads them back:

.. code-block:: python

   import unittest

   class MyTrackerTest(unittest.TestCase):
       def test_round_trip_metadata(self) -> None:
           tracker = MyTracker(connection_str="test://localhost")
           tracker.add_metadata("run-1", lr=0.01, epochs=10)
           md = tracker.metadata("run-1")
           self.assertEqual(md["lr"], 0.01)
           self.assertEqual(md["epochs"], 10)

       def test_round_trip_artifact(self) -> None:
           tracker = MyTracker(connection_str="test://localhost")
           tracker.add_artifact("run-1", "checkpoint", "/path/to/ckpt.pt")
           arts = tracker.artifacts("run-1")
           self.assertIn("checkpoint", arts)
           self.assertEqual(arts["checkpoint"].path, "/path/to/ckpt.pt")

Test factory wiring with ``patch.dict``:

.. code-block:: python

   import os
   from unittest.mock import patch

   @patch.dict(os.environ, {
       "TORCHX_TRACKERS": "my_tracker",
       "TORCHX_TRACKER_MY_TRACKER_CONFIG": "test://localhost",
       "TORCHX_JOB_ID": "test-run-1",
   })
   def test_tracker_from_env(self) -> None:
       from torchx.tracker.api import trackers_from_environ
       trackers = list(trackers_from_environ())
       self.assertEqual(len(trackers), 1)

Common Pitfalls
"""""""""""""""""

* **Entry point targets the class, not the factory**: The entry point must
  reference a factory function ``(config: str | None) -> TrackerBase``, not
  the class itself.

* **Factory signature mismatch**: The factory receives ``config: str | None``,
  not keyword arguments. Parse connection strings or JSON inside the factory.

* **Forgetting to handle ``None`` config**: When no
  ``TORCHX_TRACKER_<NAME>_CONFIG`` env var is set, ``config`` is ``None``.
  Provide a sensible default or raise a clear error.

API Reference
^^^^^^^^^^^^^^^^

.. autoclass:: AppRun
   :members:

.. autoclass:: torchx.tracker.api.TrackerBase
   :members:

Data Types
""""""""""""

.. autoclass:: torchx.tracker.api.TrackerArtifact
   :members:

.. autoclass:: torchx.tracker.api.TrackerSource
   :members:

Built-in Backends
"""""""""""""""""""

.. autoclass:: torchx.tracker.backend.fsspec.FsspecTracker
   :members:
   :show-inheritance:

.. autoclass:: torchx.tracker.mlflow.MLflowTracker
   :members:
   :show-inheritance:

CLI Command
"""""""""""""

.. autoclass:: torchx.cli.cmd_tracker.CmdTracker
   :members:

Environment Variables
"""""""""""""""""""""""

Set automatically by the runner when trackers are configured.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Purpose
   * - ``TORCHX_JOB_ID``
     - The :py:data:`~torchx.specs.AppHandle` for the current job. Set by
       the runner and used by :py:meth:`AppRun.run_from_env` to identify the run.
   * - ``TORCHX_TRACKERS``
     - Comma-separated list of tracker entry-point keys to activate
       (e.g. ``fsspec,my_tracker``).
   * - ``TORCHX_TRACKER_<NAME>_CONFIG``
     - Per-tracker configuration string passed to the factory function.
       ``<NAME>`` is the upper-cased entry-point key.
   * - ``TORCHX_PARENT_RUN_ID``
     - Optional parent run ID for lineage tracking. Set by the runner;
       read by :py:meth:`AppRun.run_from_env` which automatically calls
       ``tracker.add_source()`` on each backend to record the lineage link.

.. seealso::

   :doc:`advanced`
      Entry-point registration for custom trackers, schedulers, and components.

   :doc:`runtime/tracking`
      Runtime tracking utilities for use within applications.
