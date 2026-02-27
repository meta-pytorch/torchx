CLI
==============

.. tip::

   The ``torchx`` CLI provides subcommands for launching jobs, querying
   schedulers, and managing running applications. It can be extended with
   custom subcommands via :term:`entry points <Entry Point>`.

**Prerequisites:** :doc:`quickstart` (installation and first launch).

The ``torchx`` CLI is the primary way most users interact with TorchX. The
:py:class:`~torchx.runner.Runner` Python API provides the same capabilities
programmatically (see :doc:`runner`).

Built-in Commands
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Purpose
   * - ``run``
     - Launch a component on a scheduler.
   * - ``builtins``
     - List all registered components.
   * - ``runopts``
     - Show scheduler-specific config options.
   * - ``status``
     - Check the status of a submitted job.
   * - ``describe``
     - Describe a submitted job (reconstruct its AppDef).
   * - ``log``
     - Fetch log lines for a running or completed job.
   * - ``list``
     - List jobs on a scheduler.
   * - ``cancel``
     - Cancel a running job.
   * - ``delete``
     - Delete a job definition from the scheduler.
   * - ``configure``
     - Manage ``.torchxconfig`` settings.
   * - ``tracker``
     - Query tracker backends for artifacts and metadata.

``torchx run`` Key Flags
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell-session

   $ torchx run [--scheduler SCHED] [-cfg KEY=VAL,...] [--workspace PATH] COMPONENT [ARGS...]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Purpose
   * - ``-s`` / ``--scheduler``
     - Scheduler backend name (e.g. ``local_cwd``, ``kubernetes``, ``slurm``).
   * - ``-cfg`` / ``--scheduler_args``
     - Comma-separated scheduler config key-value pairs
       (e.g. ``-cfg namespace=default,queue=gpu``). Run ``torchx runopts``
       to see available options per scheduler.
   * - ``--workspace``
     - Path to the local workspace directory. Overrides ``Role.workspace``
       for role[0].
   * - ``--dryrun``
     - Print the scheduler request without submitting.
   * - ``--wait``
     - Block until the job reaches a terminal state.
   * - ``--log``
     - Tail logs after submission (implies ``--wait``).

Usage examples:

.. code-block:: shell-session

   $ torchx run --scheduler local_cwd utils.python --script my_app.py
   $ torchx run --scheduler kubernetes -cfg namespace=default dist.ddp -j 2x2 --script train.py
   $ torchx runopts kubernetes
   $ torchx status local_cwd://torchx/my_job_id
   $ torchx log local_cwd://torchx/my_job_id trainer/0

Extending the CLI
-------------------

Subclass :py:class:`~torchx.cli.cmd_base.SubCommand` and register via the
``torchx.cli.cmds`` entry-point group. Implement two methods:

* :py:meth:`~torchx.cli.cmd_base.SubCommand.add_arguments` -- register flags
  and positional arguments.
* :py:meth:`~torchx.cli.cmd_base.SubCommand.run` -- execute the command.

See :ref:`Registering Custom CLI Commands <registering-custom-cli-commands>` in
the Advanced Usage guide for a complete walkthrough with code examples.

Testing Your SubCommand
^^^^^^^^^^^^^^^^^^^^^^^^^

Construct an ``argparse.Namespace`` and call ``run()`` directly:

.. code-block:: python

   import argparse
   import unittest

   class CmdMyToolTest(unittest.TestCase):
       def test_run(self) -> None:
           cmd = CmdMyTool()
           parser = argparse.ArgumentParser()
           cmd.add_arguments(parser)
           args = parser.parse_args(["--config", "test.yaml", "app-123"])
           # cmd.run(args)  # call and assert side effects

See ``torchx/cli/test/`` for tests of the built-in subcommands.

Common Pitfalls
^^^^^^^^^^^^^^^^^

* **Entry-point key becomes the subcommand name**: Choose a short, descriptive
  name -- it is the exact string users type after ``torchx``.

* **Entry point targets a class, not a factory**: Unlike schedulers and
  trackers, CLI entry points reference the ``SubCommand`` **class** itself.

* **Overriding built-in commands**: If your key matches a built-in (e.g.
  ``run``), your command **replaces** it entirely.

.. _Components:

Components Library
--------------------

Components are reusable job templates that the CLI discovers via entry points.
Run ``torchx builtins`` to list all registered components, or ``torchx run
COMPONENT --help`` to see the arguments for a specific component.

.. toctree::
   :maxdepth: 1

   components/overview
   components/train
   components/distributed
   components/interpret
   components/metrics
   components/serve
   components/utils

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Components (Meta)
      :glob:

      components/fb/*

API Reference
---------------

.. automodule:: torchx.cli

.. autoclass:: torchx.cli.cmd_base.SubCommand
   :members:

.. seealso::

   :doc:`advanced`
      Entry-point registration for custom CLI commands, trackers, schedulers, and
      components.
