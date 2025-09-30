# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This module contains a collection of builtin TorchX components. The directory
structure is organized by component category. Components are simply
templetized app specs. Think of them as a factory methods for different types
of job definitions. The functions that return ``specs.AppDef`` in this
module are what we refer to as components.

You can browse the library of components in the ``torchx.components`` module
or on our :ref:`docs page<index:Components Library>`.

Using Builtins
---------------------------
Once you've found a builtin component, you can either:

1. Run the component as a job
2. Use the component in the context of a workflow (pipeline)


In both cases, the component will run as a job, with the difference being that
the job will run as a standalone job directly on a scheduler or a "stage" in a
workflow with upstream and/or downstream dependencies.

.. note:: Depending on the semantics o the component, the job may be single
          node or distributed. For instance, if the component has a single
          role where the ``role.num_replicas == 1``, then the job is a single
          node job. If the component has multiple roles and/or if any of the
          role's ``num_replicas > 1``, then the job is a multi-node distributed job.

Not sure whether you should run the component as a job or as a pipeline stage?
Use this rule of thumb:

1. Just getting started? Familiarize yourself with the component by running it as a job
2. Need job dependencies? Run the components as pipeline stages
3. Don't need job dependencies? Run the component as a job


Authoring
-----------

Since a component is simply a python function that returns an ``specs.AppDef``,
authoring your own component is as simple as writing a python function with the following
rules:

1. The component function must return an ``specs.AppDef`` and the return type must be specified
2. All arguments of the component must be PEP 484 type annotated and the type must be one of
    #. Primitives: ``int``, ``float``, ``str``, ``bool``
    #. Optional primitives: ``Optional[int]``, ``Optional[float]``, ``Optional[str]``
    #. Maps of primitives: ``Dict[Primitive_key, Primitive_value]``
    #. Lists of primitives: ``List[Primitive_values]``
    #. Optional collections: ``Optional[List]``, ``Optional[Dict]``
    #. VAR_ARG: ``*arg`` (useful when passing through arguments to the entrypoint script)
3. (optional) A docstring in `google format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
   (in particular see ``function_with_pep484_type_annotations``). This docstring is purely informative
   in that torchx cli uses it to autogenerate an informative ``--help`` message, which is
   useful when sharing components with others. If the component does not have a docstring
   the ``--help`` option will still work, but the parameters will have a canned description (see below).
   Note that when running components programmatically via :py:mod:`torchx.runner`, the docstring
   is not picked up by torchx at all.

Below is an example component that launches DDP scripts, it is a simplified version of
the :py:func:`torchx.components.dist.ddp` builtin.

.. doctest:: [component_example]

 import os
 import torchx.specs as specs

 def ddp(
     *script_args: str,
     image: str,
     script: str,
     host: str = "aws_p3.2xlarge",
     nnodes: int = 1,
     nproc_per_node: int = 1,
 ) -> specs.AppDef:
    return specs.AppDef(
        name=os.path.basename(script),
        roles=[
            spec.Role(
                name="trainer",
                image=image,
                resource=specs.named_resources[host],
                num_replicas=nnodes,
                entrypoint="python",
                args=[
                    "-m",
                    "torch.distributed.run",
                    "--rdzv_backend=c10d",
                    "--rdzv_endpoint=localhost:5900",
                    f"--nnodes={nnodes}",
                    f"--nprocs_per_node={nprocs_per_node}",
                    "-m",
                    script,
                    *script_args,
                ],
            ),
        ]
    )

Assuming the component above is saved in ``example.py``, we can run ``--help``
on it as:

.. code-block:: shell-session

 $ torchx ./example.py:ddp --help
 usage: torchx run ...torchx_params... ddp  [-h] --image IMAGE --script SCRIPT [--host HOST]
                                           [--nnodes NNODES] [--nproc_per_node NPROC_PER_NODE]
                                           ...

 AppDef: ddp. TIP: improve this help string by adding a docstring ...<omitted for brevity>...

 positional arguments:
   script_args           (required)

 optional arguments:
   -h, --help            show this help message and exit
   --image IMAGE         (required)
   --script SCRIPT       (required)
   --host HOST           (default: aws_p3.2xlarge)
   --nnodes NNODES       (default: 1)
   --nproc_per_node NPROC_PER_NODE
                         (default: 1)

If we include a docstring as such:

.. code-block:: python

  def ddp(...) -> specs.AppDef:
    \"""
    DDP Simplified.

    Args:
       image: name of the docker image containing the script + deps
       script: path of the script in the image
       script_args: arguments to the script
       host: machine type (one from named resources)
       nnodes: number of nodes to launch
       nproc_per_node: number of scripts to launch per node

    \"""

    # ... component body same as above ...
    pass

Then the ``--help`` message would reflect the function and parameter descriptions
in the docstring as such:

::

 usage: torchx run ...torchx_params... ddp  [-h] --image IMAGE --script SCRIPT [--host HOST]
                                           [--nnodes NNODES] [--nproc_per_node NPROC_PER_NODE]
                                           ...

 App spec: DDP simplified.

 positional arguments:
   script_args           arguments to the script

 optional arguments:
   -h, --help            show this help message and exit
   --image IMAGE         name of the docker image containing the script + deps
   --script SCRIPT       path of the script in the image
   --host HOST           machine type (one from named resources)
   --nnodes NNODES       number of nodes to launch
   --nproc_per_node NPROC_PER_NODE
                         number of scripts to launch per node


Validating
-----------------------

To validate that you've defined your component correctly you can either:

1. (easiest) Dryrun your component's ``--help`` with the cli: ``torchx run --dryrun ~/component.py:train --help``
2. Use the component :ref:`linter<specs:Component Linter>`
   (see `dist_test.py <https://github.com/meta-pytorch/torchx/blob/main/torchx/components/test/dist_test.py>`_ as an example)


Running as a Job
------------------------

You can run a component as a job with the :ref:`torchx cli<cli:CLI>` or programmatically with
the :ref:`torchx.runner<runner:torchx.runner>`. Both are identical, in fact the
cli uses the runner under the hood, so the choice is yours. The `quickstart <../quickstart.md>`_
guide walks though the basics for you to get started.

Programmatic Run
~~~~~~~~~~~~~~~~~~

To run builtins or your own component programmatically, simply invoke the
component as a regular python function and pass it along to the :py:mod:`torchx.runner`.
Below is an example of calling the ``utils.echo`` builtin:

.. doctest:: [running_component_programmatically]

 from torchx.components.utils import echo
 from torchx.runner import get_runner

 get_runner().run(echo(msg="hello world"), scheduler="local_cwd")

CLI Run (Builtins)
~~~~~~~~~~~~~~~~~~~

When running components from the cli, you have to pass which component function to invoke.
For builtin components this  is of the form ``{component_module}.{component_fn}``, where
the ``{component_module}`` is the module path of the component relative to ``torchx.components``
and the ``{component_fn}`` is the component function within that module. So for
``torchx.components.utils.echo``, we'd drop the ``torchx.components`` prefix and run it as

.. code-block:: shell-session

 $ torchx run utils.echo --msg "hello world"

See :ref:`CLI docs<cli:CLI>` for more information.


CLI Run (Custom)
~~~~~~~~~~~~~~~~~~~

To run your custom component with the cli, you have to use a slightly different syntax of
the form ``{component_path}:{component_fn}``. Where ``{component_path}`` is the
file path of your component's python file, and ``{component_fn}`` is the name of the
component function within that file. Assume your component is in ``/home/bob/component.py``
and the component function is called ``train()``, you would run this as

.. code-block:: shell-session

 # option 1. use absolute path
 $ torchx run /home/bob/component.py:train --help

 # option 2. let the shell do the expansion
 $ torchx run ~/component.py:train --help

 # option 3. same but after CWD to $HOME
 $ cd ~/
 $ torchx run ./component.py:train --help

 # option 4. files can be relative to CWD
 $ cd ~/
 $ torchx run component.py:train --help

.. note:: builtins can be run this way as well given that you know the install directory of TorchX!

Passing Component Params from CLI
--------------------------------------

Since components are simply python functions, using them programmatically is straight forward.
As seen above, when running components via the cli's ``run`` subcommand the component parameters are passed
as program arguments using the double-dash + param_name syntax (e.g ``--param1=1`` or ``--param1 1``).
The cli autogenerates `argparse <https://docs.python.org/3/library/argparse.html>`_ parser based on the
docstring of the component. Below is a summary on how to pass component parameters of various types,
imagine the component is defined as:

.. doctest:: [component_param_passing]

 # in comp.py
 from typing import Dict, List
 import torchx.specs as specs

 def f(i: int, f: float, s: str, b: bool, l: List[str], d: Dict[str, str], *args) -> specs.AppDef:
    \"""
    Example component

    Args:
        i: int param
        f: float param
        s: string param
        b: bool param
        l: list param
        d: map param
        args: varargs param

    Returns: specs.AppDef
    \"""

    pass


#. Help: ``torchx run comp.py:f --help``
#. Primitives (``int``, ``float``, ``str``): ``torchx run comp.py:f --i 1 --f 1.2 --s "bar"``
#. Bool: ``torchx run comp.py:f --b True`` (or ``--b False``)
#. Maps: ``torchx run comp.py:f --d k1=v1,k2=v2,k3=v3``
#. Lists: ``torchx run comp.py:f --l a,b,c``
#. VAR_ARG: ``*args`` are passed as positionals rather than arguments, hence they are specified
   at the end of the command. The ``--`` delimiter is used to start the VAR_ARGS section. This
   is useful when the component and application have the same arguments or when passing through
   the ``--help`` arg. Below are a few examples:
   * ``*args=["arg1", "arg2", "arg3"]``: ``torchx run comp.py:f --i 1 arg1 arg2 arg3``
   * ``*args=["--flag", "arg1"]``: ``torchx run comp.py:f --i 1 --flag arg1 ``
   * ``*args=["--help"]``: ``torchx run comp.py:f -- --help``
   * ``*args=["--i", "2"]``: ``torchx run comp.py:f --i 1 -- --i 2``

Additional Resources
-----------------------

See:

1. Components defined in this module as expository examples
2. Defining your own component `quick start guide <../quickstart.md>`_
3. Component best practices :ref:`guide<component_best_practices:Component Best Practices>`
4. App best practices :ref:`guide<app_best_practices:App Best Practices>`

"""
