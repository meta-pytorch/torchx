torchx.runner
==============

.. automodule:: torchx.runner

.. image:: runner_diagram.png

The :py:class:`Runner` submits, monitors, and manages jobs. Use
:py:func:`get_runner` to create one with all registered schedulers.

**Key methods:**

* :py:meth:`~Runner.run` / :py:meth:`~Runner.run_component` -- submit a job
* :py:meth:`~Runner.status` -- poll current state
* :py:meth:`~Runner.wait` -- block until terminal state
* :py:meth:`~Runner.cancel` -- request cancellation
* :py:meth:`~Runner.delete` -- remove a job definition from the scheduler
* :py:meth:`~Runner.log_lines` -- stream log output
* :py:meth:`~Runner.list` -- list jobs on a scheduler
* :py:meth:`~Runner.dryrun` -- preview what would be submitted without submitting
* :py:meth:`~Runner.schedule` -- submit a previously dry-run request (allows request mutation)

Scheduler instances are created lazily on first use. Use the Runner as a
context manager for automatic cleanup.

See :doc:`api_reference` for copy-pasteable recipes.

.. currentmodule:: torchx.runner

.. autofunction:: get_runner

.. autoclass:: Runner
   :members:

.. seealso::

   :doc:`api_reference`
      Single-page reference with imports, types, and copy-pasteable recipes.

   :doc:`schedulers`
      Scheduler API reference and implementation guide.

   :doc:`runner.config`
      Configuring scheduler options via ``.torchxconfig``.
