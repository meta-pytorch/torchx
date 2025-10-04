Advanced Usage
======================

TorchX defines plugin points for you to configure TorchX to best support
your infrastructure setup. Most of the configuration is done through
Python's `entry points <https://packaging.python.org/specifications/entry-points/>`__.

.. note::

  Entry points requires a python package containing them be installed.
  If you don't have a python package we recommend you make one so you can share
  your resource definitions, schedulers and components across your team and org.

The entry points described below can be specified in your project's `setup.py`
file as

.. testsetup:: setup

   import sys
   sys.argv = ["setup.py", "--version"]

.. testcode:: setup

 from setuptools import setup

 setup(
     name="project foobar",
     entry_points={
         "torchx.schedulers": [
             "my_scheduler = my.custom.scheduler:create_scheduler",
         ],
         "torchx.named_resources": [
             "gpu_x2 = my_module.resources:gpu_x2",
         ],
     }
 )

.. testoutput:: setup
   :hide:

   0.0.0



Registering Custom Schedulers
--------------------------------
You may implement a custom scheduler by implementing the
.. py::class torchx.schedulers.Scheduler interface.

The ``create_scheduler`` function should have the following function signature:

.. testcode::

 from torchx.schedulers import Scheduler

 def create_scheduler(session_name: str, **kwargs: object) -> Scheduler:
     return MyScheduler(session_name, **kwargs)

You can then register this custom scheduler by adding an entry_points definition
to your python project.


.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.schedulers": [
           "my_scheduler = my.custom.scheduler:create_schedule",
       ],
   }



Registering Named Resources
-------------------------------

A Named Resource is a set of predefined resource specs that are given a
string name. This is particularly useful
when your cluster has a fixed set of instance types. 

TorchX supports two ways to define named resources:

1. **Configuration-based**: Define resources in ``.torchxconfig`` files (recommended for most users)
2. **Entry point-based**: Register resources via Python entry points (for package authors)

Configuration-Based Named Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most users, the easiest way to define custom named resources is through configuration files.

Create a ``.torchxconfig`` file in your project directory:

.. code-block:: ini

   [named_resources]
   dynamic = {"cpu": 100, "gpu": 8, "memMB": 819200, "devices": {"vpc.amazonaws.com/efa": 1}}
   my_custom = {"cpu": 32, "gpu": 4, "memMB": 131072}

You can also use the ``TORCHXCONFIG`` environment variable to specify a custom config file path.

Usage example:

.. code-block:: python

   from torchx.specs import resource
   my_resource = resource(h="dynamic")  # Uses your config-defined resource

Entry Point-Based Named Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For instance if your deep learning training kubernetes cluster on AWS is
comprised only of p3.16xlarge (64 vcpu, 8 gpu, 488GB), then you may want to
enumerate t-shirt sized resource specs for the containers as:

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

To make these resource definitions available you then need to register them via
entry_points:

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.named_resources": [
           "gpu_x2 = my_module.resources:gpu_x2",
       ],
   }


Once you install the package with the entry_points definitions, the named
resource can then be used in the following manner:

.. testsetup:: role

   from torchx.specs import _named_resource_factories, Resource

   _named_resource_factories["gpu_x2"] = lambda: Resource(cpu=16, gpu=2, memMB=122_000)


.. doctest:: role

   >>> from torchx.specs import get_named_resources
   >>> get_named_resources("gpu_x2")
   Resource(cpu=16, gpu=2, memMB=122000, ...)


.. testcode:: role

  # my_module.component
  from torchx.specs import AppDef, Role, get_named_resources

  def test_app(resource: str) -> AppDef:
      return AppDef(name="test_app", roles=[
          Role(
              name="...",
              image="...",
              resource=get_named_resources(resource),
          )
      ])

  test_app("gpu_x2")


Registering Custom Components
-------------------------------
You can author and register a custom set of components with the
``torchx`` CLI as builtins to the CLI. This makes it possible to customize
a set of components most relevant to your team or organization and support
it as a CLI ``builtin``. This way users will see your custom components
when they run

.. code-block:: shell-session

 $ torchx builtins

Custom components can be registered via ``[torchx.components]`` entrypoints.
If ``my_project.bar`` had the following directory structure:

::

 $PROJECT_ROOT/my_project/bar/
     |- baz.py

And ``baz.py`` had a single component (function) called ``trainer``:

::

 # baz.py
 import torchx.specs as specs

 def trainer(...) -> specs.AppDef: ...


And the entrypoints were added as:

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "foo = my_project.bar",
       ],
   }

TorchX will search the module ``my_project.bar`` for all defined components and group the found
components under the ``foo.*`` prefix. In this case, the component ``my_project.bar.baz.trainer``
would be registered with the name ``foo.baz.trainer``.

.. note::
    Only python packages (those directories with an ``__init__.py`` file)
    are searched for and TorchX makes no attempt to recurse into namespace packages
    (directories without a ``__init__.py`` file).
    However you may register a top level namespace package.

``torchx`` CLI will display registered components via:

.. code-block:: shell-session

 $ torchx builtins
 Found 1 builtin components:
 1. foo.baz.trainer

The custom component can then be used as:

.. code-block:: shell-session

 $ torchx run foo.baz.trainer -- --name "test app"


When you register your own components, TorchX will not include its own builtins. To add TorchX's
builtin components you must specify another entry as:


.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "foo = my_project.bar",
           "torchx = torchx.components",
       ],
   }

This will add back the TorchX builtins but with a ``torchx.*`` component name prefix (e.g. ``torchx.dist.ddp``
versus the default ``dist.ddp``).

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


There will be two sets of overlapping components for those components in ``my_project.bar`` with different
prefix aliases: ``foo.*`` and ``test.bar.*``. Concretely,

.. code-block:: shell-session

 $ torchx builtins
 Found 2 builtin components:
 1. foo.baz.trainer
 2. test.bar.baz.trainer

To omit groupings and make the component names shorter, use underscore (e.g ``_`` or ``_0``, ``_1``, etc).
For example:

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "_0 = my_project.bar",
           "_1 = torchx.components",
       ],
   }

This has the effect of exposing the trainer component as ``baz.trainer`` (as opposed to ``foo.baz.trainer``)
and adds back the builtin components as in the vanilla installation of torchx, without the ``torchx.*`` prefix.

.. code-block:: shell-session

 $ torchx builtins
 Found 11 builtin components:
 1. baz.trainer
 2. dist.ddp
 3. utils.python
 4. ... <more builtins from torchx.components.* ...>

