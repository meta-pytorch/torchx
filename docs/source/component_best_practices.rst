Component Best Practices
==========================

.. tip::

   Best practices for authoring reusable TorchX
   :term:`components <Component>`: entrypoints, simplicity, named resources,
   composition, and testing.

**Prerequisites:** :doc:`custom_components`.

These practices reflect conventions used in the builtin components. Deviate
when your use case demands it.

Entrypoints
-------------------

Prefer ``python -m <module>`` over a path to the main module. Module
resolution works across environments (Docker, Slurm) regardless of directory
structure.

For non-Python apps, place the binary on ``PATH`` instead.

.. code-block:: python

   from torchx.specs import AppDef, Role

   def trainer(img_name: str, img_version: str) -> AppDef:
       return AppDef(name="trainer", roles=[
           Role(
               name="trainer",
               image=f"{img_name}:{img_version}",
               entrypoint="python",
               args=["-m", "your.app"],
           )
       ])


Simplify
-------------------

Keep each component as simple as possible.

Argument Processing
^^^^^^^^^^^^^^^^^^^^^

Pass ``image`` directly to ``AppDef`` without manipulation -- processing
breaks portability across environments.

.. code-block:: python

   def trainer(image: str) -> AppDef:
       return AppDef(name="trainer", roles=[Role(name="trainer", image=image)])

Branching Logic
^^^^^^^^^^^^^^^^^

Avoid ``if`` statements in components. Create multiple components with shared
private helpers instead.

.. code-block:: python

   def trainer_test() -> AppDef:
       return _trainer(num_replicas=1)

   def trainer_prod() -> AppDef:
       return _trainer(num_replicas=10)

   # not a component — just a shared helper
   def _trainer(num_replicas: int) -> AppDef:
       return AppDef(
           name="trainer",
           roles=[Role(name="trainer", image="my_image:latest", num_replicas=num_replicas)],
       )


Documentation
^^^^^^^^^^^^^^^^^^^^^

Document component functions. See :doc:`components/overview` for examples.


Named Resources
-----------------

Use :term:`named resources <Resource>` instead of hard-coding CPU and memory
values:

.. code-block:: python

   from torchx.specs import resource

   resource(h="aws_p3.2xlarge")

See :ref:`advanced:Registering Named Resources` for defining custom named
resources.

Composing Components
----------------------

Start from base component definitions rather than building ``AppDef`` from
scratch:

* :py:mod:`torchx.components.base` for simple single node components.
* :py:func:`torchx.components.dist.ddp` for distributed components.

You can also merge roles from multiple components to run sidecars alongside
the main job.

Distributed Components
------------------------

Use :py:func:`torchx.components.dist.ddp` for distributed training. Extend it
by writing a wrapper that calls ``ddp`` with your configuration.

Define All Arguments
----------------------

Define all arguments as function parameters instead of consuming a dictionary.
This enables discoverability and static type checking.

Unit Tests
--------------

.. automodule:: torchx.components.component_test_base
.. currentmodule:: torchx.components.component_test_base

.. autoclass:: torchx.components.component_test_base.ComponentTestCase
   :members:

Integration Tests
-------------------

Use the :py:class:`~torchx.runner.Runner` API or CLI scripts. See the
`scheduler integration tests <https://github.com/meta-pytorch/torchx/tree/main/.github/workflows>`__
for examples.

.. seealso::

   :doc:`api_reference`
      Single-page reference with imports, types, and copy-pasteable recipes.

   :doc:`app_best_practices`
      Best practices for writing TorchX applications.

   :doc:`custom_components`
      Step-by-step guide for building and launching a custom component.

   :doc:`components/overview`
      Browse the builtin component library.
