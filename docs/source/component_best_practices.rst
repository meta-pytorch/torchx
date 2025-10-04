Component Best Practices
==========================

This has a list of common things you might want to do with a component and best
practices for them. Components are designed to be flexible so you can deviate
from these practices if necessary however these are the best practices we use
for the builtin TorchX components.

See :ref:`app_best_practices:App Best Practices` for information how to write
apps using TorchX.

Entrypoints
-------------------

When possible it's best to call your reusable component via ``python -m <module>``
instead of specifying the path to the main module. This makes it so it can be
used in multiple different environments such as docker and slurm by relying on
the python module resolution instead of the directory structure.

If your app isn't python based, you can place your app in a folder on your
`PATH` so it's accessible regardless of the directory structure.

.. code-block:: python

   def trainer(img_name: str, img_version: str) -> AppDef:
       return AppDef(roles=[
           Role(
               entrypoint="python",
               args=[
                   "-m",
                   "your.app",
               ],
           )
       ])


Simplify
-------------------

When writing a component you want to keep each component as simple as
possible to make it easier for others to reuse and understand.

Argument Processing
^^^^^^^^^^^^^^^^^^^^^

Argument processing makes it hard to use the component in other environments.
For images in particular we want to directly pass the image field to the AppDef
since any sort of manipulation will make it impossible to use in other
environments with different image naming conventions.

.. code-block:: python

   def trainer(image: str):
       return AppDef(roles=[Role(image=image)...)

Branching Logic
^^^^^^^^^^^^^^^^^

You should avoid branching logic in the components. If you have a case where you
feel like you need an ``if`` statement in the component you should prefer to
create multiple components with shared logic. Complex arguments make it hard for
others to understand how to use it.

.. code-block:: python

   def trainer_test():
       return _trainer(num_replicas=1)

   def trainer_prod() -> AppDef:
       return _trainer(num_replicas=10)

   # not a component just a function
   def _trainer(num_replicas: int) -> AppDef:
        return AppDef(roles=[Role(..., num_replicas=num_replicas)])


Documentation
^^^^^^^^^^^^^^^^^^^^^

The documentation is optional, but it is the best practice to keep component functions documented,
especially if you want to share your components. See :ref:Component Authoring<components/overview:Authoring>
for more details.


Named Resources
-----------------

When writing components it's best to use TorchX's named resources support
instead of manually specifying cpu and memory allocations. Named resources allow
your component to be environment independent and allow for better scheduling
behavior by using t-shirt sizes.

See :py:meth:`torchx.specs.get_named_resources` for more info.

Composing Components
----------------------

For common component styles we provide base component definitions. These can be
called from your custom component definition and an alternative to creating a
full AppDef from scratch.

See:

* :py:mod:`torchx.components.base` for simple single node components.
* :py:meth:`torchx.components.dist.ddp` for distributed components.

For even more complex components it's possible to merge multiple existing
components into a single one. For instance you could use a metrics UI component
and merge the roles from it into training component roles to have a sidecar
service to your main training job.

Distributed Components
------------------------

If you're writing a component for distributed training or other similar
distributed computation, we recommend using the
:py:meth:`torchx.components.dist.ddp` component since it provides out of the box
support for ``torch.distributed.elastic`` jobs.

You can extend the ``ddp`` component by writing a custom component that simple
imports the ``ddp`` component and calls it with your app configuration.

Define All Arguments
----------------------

It's preferable to define all component arguments as function arguments instead
of consuming a dictionary of arguments. This makes it easier for users to figure
out the options as well as can provide static typing when used with `pyre
<https://pyre-check.org/>`__ or `mypy <http://mypy-lang.org/>`__.

Unit Tests
--------------

.. automodule:: torchx.components.component_test_base
.. currentmodule:: torchx.components.component_test_base

.. autoclass:: torchx.components.component_test_base.ComponentTestCase
   :members:

Integration Tests
-------------------

You can setup integration tests with your components by either using the
programmatic runner API or write a bash script to call the CLI.

You can see both styles in use in the core
`TorchX scheduler integration tests <https://github.com/meta-pytorch/torchx/tree/main/.github/workflows>`__.
