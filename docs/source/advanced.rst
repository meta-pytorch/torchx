Advanced Usage
======================

.. tip::

   This guide covers TorchX's extension points: registering custom schedulers,
   named resources, components, trackers, and CLI commands. The recommended
   approach is the ``@register`` decorator with ``torchx_plugins.*`` namespace
   packages. See :doc:`plugins` for the full API reference.

**Audience:** Platform engineers who want to integrate TorchX with custom
infrastructure. If you only need to **use** TorchX to launch jobs, the
:doc:`quickstart`, :doc:`basics`, and :doc:`custom_components` pages are
sufficient.

**Prerequisites:** :doc:`basics` (core concepts) and :doc:`custom_components`.

.. deprecated::
   Entry-point based plugin registration (``[torchx.*]`` sections in
   ``setup.py`` / ``pyproject.toml``) is deprecated. Use the ``@register``
   decorator with ``torchx_plugins.*`` namespace packages instead.

   Namespace plugins are **always** loaded. By default
   (``TORCHX_NO_ENTRYPOINTS=0`` or unset), entry points are also loaded
   for backward compatibility. Set ``TORCHX_NO_ENTRYPOINTS=1`` to load
   only namespace plugins. Entry-point loading will be removed in a
   future release. See :doc:`plugins` for the full API reference.

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │                     TorchX Extension Points                  │
   │                                                              │
   │  @register Decorator       What You Register                 │
   │  ──────────────────────    ──────────────────────────────    │
   │  @register.scheduler()     Scheduler factory function        │
   │  @register.named_resource  Resource factory function         │
   │  @register.tracker()       Tracker factory function          │
   │                                                              │
   │  Entry Points (not yet migrated to @register)                │
   │  ──────────────────────    ──────────────────────────────    │
   │  torchx.components         Component module path             │
   │  torchx.cli.cmds           SubCommand class                  │
   │                                                              │
   │  ┌──────────────────────────────────────────┐                │
   │  │ torchx_plugins/<group>/<module>.py       │ ◄── write here │
   │  │   @register.scheduler()                  │                │
   │  │   def my_scheduler(...): ...             │                │
   │  └──────────────────┬───────────────────────┘                │
   │                     │                                        │
   │                     ▼                                        │
   │         ┌───────────────────────┐                            │
   │         │    pip install .      │  ◄── install package       │
   │         └───────────┬───────────┘                            │
   │                     │                                        │
   │                     ▼                                        │
   │  ┌───────────────────────────────────────────────────┐       │
   │  │             TorchX Runtime Discovery              │       │
   │  │                                                   │       │
   │  │  Runner ──► discovers schedulers, resources       │       │
   │  │  CLI    ──► discovers components, subcommands     │       │
   │  │  AppRun ──► discovers tracker backends            │       │
   │  └───────────────────────────────────────────────────┘       │
   └──────────────────────────────────────────────────────────────┘

Plugins are discovered from ``torchx_plugins.*`` namespace packages.
Decorate your factory with ``@register.<type>()`` and TorchX finds it
automatically after ``pip install``.



Registering Custom Schedulers
--------------------------------

Implement the :py:class:`~torchx.schedulers.Scheduler` interface (see
:ref:`implementing-scheduler` for a full skeleton). The factory function
signature:

.. testcode::

 from torchx.schedulers import Scheduler

 def create_scheduler(session_name: str, **kwargs: object) -> Scheduler:
     return MyScheduler(session_name, **kwargs)

**Recommended: ``@register`` decorator**

Place your scheduler in a ``torchx_plugins/schedulers/`` namespace package
and decorate it:

.. code-block:: python

   # torchx_plugins/schedulers/my_scheduler.py
   from torchx.plugins import register

   @register.scheduler()
   def my_scheduler(session_name: str, **kwargs) -> Scheduler:
       return MyScheduler(session_name, **kwargs)

After ``pip install``, TorchX discovers it automatically.

**Legacy: entry points** *(deprecated)*

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.schedulers": [
           "my_scheduler = my.custom.scheduler:create_scheduler",
       ],
   }

Or in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."torchx.schedulers"]
   my_scheduler = "my.custom.scheduler:create_scheduler"

Once installed, the scheduler is available everywhere:

.. code-block:: python

   from torchx.runner import get_runner

   with get_runner() as runner:
       runner.run_component("dist.ddp", ["--script", "train.py"], scheduler="my_scheduler")



Registering Named Resources
-------------------------------

A :term:`Named Resource <Resource>` maps a human-readable name (e.g.
``gpu_x2``) to a :py:class:`~torchx.specs.Resource`.

**Recommended: ``@register`` decorator**

Place your resources in a ``torchx_plugins/named_resources/`` namespace package
and decorate them:

.. code-block:: python

   # torchx_plugins/named_resources/my_cluster.py
   from torchx.plugins import powers_of_two_gpus, register, WHOLE
   from torchx.specs import Resource

   @register.named_resource(fractionals=powers_of_two_gpus)
   def gpu_x(fractional: float = WHOLE) -> Resource:
       return Resource(
           cpu=int(64 * fractional),
           gpu=int(8 * fractional),
           memMB=int(488_000 * fractional),
       )
   # Registers: gpu_x (base), gpu_x_8, gpu_x_4, gpu_x_2, gpu_x_1

   @register.named_resource()
   def cpu_x32() -> Resource:
       return Resource(cpu=32, gpu=0, memMB=131072)

The ``fractionals`` argument auto-generates fractional variants. See
:py:func:`~torchx.plugins.powers_of_two_gpus` and
:py:func:`~torchx.plugins.halve_mem_down_to` in :doc:`plugins`.

**Legacy: entry points** *(deprecated)*

For example, on an AWS cluster with p3.16xlarge nodes:

.. testcode:: python

 from torchx.specs import Resource

 def gpu_x1() -> Resource:
     return Resource(cpu=8,  gpu=1, memMB=61_000)

 def gpu_x2() -> Resource:
     return Resource(cpu=16, gpu=2, memMB=122_000)

 def gpu_x3() -> Resource:
     return Resource(cpu=32, gpu=4, memMB=244_000)

 def gpu_x4() -> Resource:
     return Resource(cpu=64, gpu=8, memMB=488_000)

.. testcode:: python
 :hide:

 gpu_x1()
 gpu_x2()
 gpu_x3()
 gpu_x4()

Register them via entry points:

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.named_resources": [
           "gpu_x2 = my_module.resources:gpu_x2",
       ],
   }

Or in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."torchx.named_resources"]
   gpu_x2 = "my_module.resources:gpu_x2"


Once installed, use the named resource:

.. testsetup:: role

   from torchx.specs import _named_resource_factories, Resource

   _named_resource_factories["gpu_x2"] = lambda: Resource(cpu=16, gpu=2, memMB=122_000)


.. doctest:: role

   >>> from torchx.specs import resource
   >>> resource(h="gpu_x2")
   Resource(cpu=16, gpu=2, memMB=122000, ...)


.. testcode:: role

  # my_module.component
  from torchx.specs import AppDef, Role, resource

  def test_app(res: str) -> AppDef:
      return AppDef(name="test_app", roles=[
          Role(
              name="...",
              image="...",
              resource=resource(h=res),
          )
      ])

  test_app("gpu_x2")

Alternatively, define resources in a module and point to it via the
``TORCHX_CUSTOM_NAMED_RESOURCES`` environment variable:

.. code-block:: python

   # my_resources.py
   from torchx.specs import Resource

   def gpu_x8_efa() -> Resource:
       return Resource(cpu=100, gpu=8, memMB=819200, devices={"vpc.amazonaws.com/efa": 1})

   def cpu_x32() -> Resource:
       return Resource(cpu=32, gpu=0, memMB=131072)

   NAMED_RESOURCES = {
       "gpu_x8_efa": gpu_x8_efa,
       "cpu_x32": cpu_x32,
   }

Then set the environment variable:

.. code-block:: bash

   export TORCHX_CUSTOM_NAMED_RESOURCES=my_resources

This avoids the need for a package with entry points.

**Verifying registration.** After installing your package, confirm the
resource is discoverable:

.. code-block:: python

   from torchx.specs import resource
   res = resource(h="gpu_x2")
   assert res.gpu == 2, f"expected 2 GPUs, got {res.gpu}"

If the name is not found, ``resource()`` raises ``KeyError`` with a
suggestion of close matches.


Registering Custom Components
-------------------------------

Register custom components as CLI builtins.

.. note::

   Components still use entry-point registration. The ``@register`` decorator
   migration is pending for components (tracked for a future release). Use
   entry points as shown below.

.. code-block:: shell-session

 $ torchx builtins

If ``my_project.bar`` has the following directory structure:

.. code-block:: text

 $PROJECT_ROOT/my_project/bar/
     |- baz.py

And ``baz.py`` has a component function called ``trainer``:

.. code-block:: python

 # baz.py
 import torchx.specs as specs

 def trainer(...) -> specs.AppDef: ...


Register via entry points:

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "foo = my_project.bar",
       ],
   }

Or in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."torchx.components"]
   foo = "my_project.bar"

TorchX searches ``my_project.bar`` for components and groups them under the
``foo.*`` prefix. The component ``my_project.bar.baz.trainer`` becomes
``foo.baz.trainer``.

.. note::

   Only Python packages (directories with ``__init__.py``) are searched.
   Namespace packages (no ``__init__.py``) are not recursed into.

Verify registration:

.. code-block:: shell-session

 $ torchx builtins
 Found 1 builtin components:
 1. foo.baz.trainer

Use from the CLI or Python:

.. code-block:: shell-session

 $ torchx run foo.baz.trainer -- --name "test app"

.. code-block:: python

 from torchx.runner import get_runner

 with get_runner() as runner:
     runner.run_component("foo.baz.trainer", ["--name", "test app"], scheduler="local_cwd")

Custom components replace the default builtins. To keep them, add another
entry:


.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "foo = my_project.bar",
           "torchx = torchx.components",
       ],
   }

This adds back TorchX builtins with a ``torchx.*`` prefix (e.g. ``torchx.dist.ddp``
instead of ``dist.ddp``).

.. _advanced-overlapping-components:

If there are two registry entries pointing to the same component, for instance

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "foo = my_project.bar",
           "test = my_project",
       ],
   }


Components in ``my_project.bar`` will appear under both ``foo.*`` and
``test.bar.*``:

.. code-block:: shell-session

 $ torchx builtins
 Found 2 builtin components:
 1. foo.baz.trainer
 2. test.bar.baz.trainer

To omit the prefix, use underscore names (``_``, ``_0``, ``_1``, etc.):

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "_0 = my_project.bar",
           "_1 = torchx.components",
       ],
   }

This exposes ``baz.trainer`` (instead of ``foo.baz.trainer``) and restores
builtins without the ``torchx.*`` prefix:

.. code-block:: shell-session

 $ torchx builtins
 Found 11 builtin components:
 1. baz.trainer
 2. dist.ddp
 3. utils.python
 4. ... <more builtins from torchx.components.* ...>

.. _registering-custom-trackers:

Registering Custom Trackers
-------------------------------

TorchX ships with :py:class:`~torchx.tracker.backend.fsspec.FsspecTracker` and
:py:class:`~torchx.tracker.mlflow.MLflowTracker`. Implement your own by
subclassing :py:class:`~torchx.tracker.api.TrackerBase`.

**The TrackerBase ABC** defines eight abstract methods:

.. code-block:: python

   from torchx.tracker.api import TrackerBase, TrackerArtifact, TrackerSource, Lineage
   from typing import Iterable, Mapping

   class MyTracker(TrackerBase):
       def __init__(self, connection_str: str) -> None:
           self._conn = connection_str

       def add_artifact(
           self, run_id: str, name: str, path: str,
           metadata: Mapping[str, object] | None = None,
       ) -> None: ...

       def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]: ...

       def add_metadata(self, run_id: str, **kwargs: object) -> None: ...

       def metadata(self, run_id: str) -> Mapping[str, object]: ...

       def add_source(
           self, run_id: str, source_id: str,
           artifact_name: str | None = None,
       ) -> None: ...

       def sources(
           self, run_id: str, artifact_name: str | None = None,
       ) -> Iterable[TrackerSource]: ...

       def lineage(self, run_id: str) -> Lineage: ...

       def run_ids(self, **kwargs: str) -> Iterable[str]: ...

**Factory function.** Each tracker plugin must provide a factory:

.. code-block:: python

   from torchx.tracker.api import TrackerBase

   def create(config: str | None) -> TrackerBase:
       return MyTracker(connection_str=config or "default://localhost")

**Recommended: ``@register`` decorator**

Place your tracker in a ``torchx_plugins/tracker/`` namespace package
and decorate the factory:

.. code-block:: python

   # torchx_plugins/tracker/my_tracker.py
   from torchx.plugins import register

   @register.tracker()
   def my_tracker(config: str | None) -> TrackerBase:
       return MyTracker(connection_str=config or "default://localhost")

**Legacy: entry points** *(deprecated)*

Register the factory under the ``torchx.tracker`` group:

.. code-block:: python

   # setup.py
   ...
   entry_points={
       "torchx.tracker": [
           "my_tracker = my_package.tracking:create",
       ],
   }

Or in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."torchx.tracker"]
   my_tracker = "my_package.tracking:create"

**Activation via environment variables:**

.. code-block:: bash

   # Comma-separated list of tracker entry-point keys to activate
   export TORCHX_TRACKERS=my_tracker,fsspec

   # Per-tracker config (optional) — passed as the ``config`` argument to the factory
   export TORCHX_TRACKER_MY_TRACKER_CONFIG="my_tracker://db-host:5432/runs"
   export TORCHX_TRACKER_FSSPEC_CONFIG="/tmp/tracker_data"

The naming convention for per-tracker config is ``TORCHX_TRACKER_<NAME>_CONFIG``
where ``<NAME>`` is the upper-cased entry-point key.

**Alternative: .torchxconfig file.** Declare trackers in ``[torchx:tracker]``
and configure each in ``[tracker:<name>]``:

.. code-block:: ini

   [torchx:tracker]
   my_tracker =
   fsspec =

   [tracker:my_tracker]
   config = my_tracker://db-host:5432/runs

   [tracker:fsspec]
   config = /tmp/tracker_data

Environment variables take precedence over ``.torchxconfig`` values.

.. seealso::

   :doc:`tracker`
      Full tracker API reference (:py:class:`~torchx.tracker.api.TrackerBase`,
      :py:class:`~torchx.tracker.api.AppRun`).

   :doc:`runtime/tracking`
      Runtime tracking utilities for use within applications.



.. _registering-custom-cli-commands:

Registering Custom CLI Commands
----------------------------------

Extend the ``torchx`` CLI by implementing
:py:class:`~torchx.cli.cmd_base.SubCommand`.

**The SubCommand ABC** defines two abstract methods:

.. code-block:: python

   import argparse
   from torchx.cli.cmd_base import SubCommand

   class CmdMyTool(SubCommand):
       def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
           """Register CLI flags and positional arguments."""
           subparser.add_argument("--config", type=str, help="Path to config file")
           subparser.add_argument("app_id", type=str, help="Application handle")

       def run(self, args: argparse.Namespace) -> None:
           """Execute the command with parsed arguments."""
           print(f"Running my_tool on {args.app_id} with config={args.config}")

Register the class (not a factory) under ``torchx.cli.cmds``. The key
becomes the subcommand name:

.. code-block:: python

   # setup.py
   ...
   entry_points={
       "torchx.cli.cmds": [
           "my_tool = my_package.cli:CmdMyTool",
       ],
   }

Or in ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."torchx.cli.cmds"]
   my_tool = "my_package.cli:CmdMyTool"

Once installed, the command is available as:

.. code-block:: shell-session

   $ torchx my_tool --config config.yaml local://session/my_app

.. note::

   Custom commands **override** built-in commands with the same name.

The default built-in commands are: ``builtins``, ``cancel``, ``configure``,
``delete``, ``describe``, ``list``, ``log``, ``run``, ``runopts``, ``status``,
and ``tracker``.


Packaging a Plugin
---------------------

To create a TorchX plugin distribution, use
`native namespace packages <https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#native-namespace-packages>`_
-- do **not** add ``__init__.py`` files under ``torchx_plugins/``. This allows
multiple independent distributions to contribute to the same namespace.

Structure your project like:

.. code-block:: text

   my-torchx-plugin/
   ├── pyproject.toml
   └── src/
       └── torchx_plugins/          # NO __init__.py
           └── schedulers/          # NO __init__.py
               └── my_scheduler.py  # uses @register.scheduler()

Configure the build backend to include ``src/torchx_plugins`` as a package.
For example with ``hatchling``:

.. code-block:: toml

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "my-torchx-plugin"
   version = "0.1.0"
   dependencies = ["torchx"]

   [tool.hatch.build.targets.wheel]
   packages = ["src/torchx_plugins"]

After ``pip install my-torchx-plugin``, TorchX will automatically discover
your plugins from the ``@register``-decorated functions.

**Single-project layout** — You don't need a separate package just for
plugins. A single project can ship both your application code and a
``torchx_plugins/`` namespace package side-by-side. The key is to list
**both** packages in the build backend's ``packages`` configuration so the
wheel includes them together.

The example below uses a `UV <https://docs.astral.sh/uv/>`_ project with
``hatchling``. For other build backends (e.g. ``setuptools``), consult their
documentation for the equivalent package-discovery configuration.

.. code-block:: text

   my-project/
   ├── pyproject.toml
   ├── my_training/              # your application code (regular package)
   │   ├── __init__.py
   │   ├── train.py
   │   └── model.py
   └── torchx_plugins/           # NO __init__.py (namespace package)
       ├── schedulers/            # NO __init__.py
       │   └── my_scheduler.py
       └── named_resources/       # NO __init__.py
           └── my_cluster.py

A single ``pyproject.toml`` manages both packages:

.. code-block:: toml

   [project]
   name = "my-project"
   version = "0.1.0"
   dependencies = ["torchx", "torch"]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["my_training", "torchx_plugins"]

After ``uv sync``, both ``my_training`` and the TorchX plugins are installed
into the same virtual environment. TorchX discovers the
``@register``-decorated plugins automatically — no entry points needed.


Diagnostics
-------------

Print a diagnostic report of all discovered plugins:

.. code-block:: python

   from torchx import plugins
   print(plugins.registry())

This lists every discovered plugin, its source module, and the distribution
package it belongs to. Errors encountered during discovery are reported at
the bottom.


.. seealso::

   :doc:`plugins`
      Plugin API reference (``@register``, ``registry()``, ``PluginRegistry``).

   :doc:`cli`
      CLI module API reference.

   :doc:`schedulers`
      Scheduler API reference and implementation guide.

   :doc:`workspace`
      Workspace API reference and custom workspace mixin guide.

   :doc:`tracker`
      Tracker API reference and backend implementations.

   :doc:`basics`
      Core TorchX concepts and project structure.

   :doc:`custom_components`
      Step-by-step guide for writing and launching a custom component.

   :doc:`component_best_practices`
      Best practices for authoring reusable components.
