Quickstart
==========

.. tip::

   Install TorchX, write a simple app, and launch it locally and remotely --
   including distributed jobs. Estimated time: 10--15 minutes.

Installation
------------

Install TorchX (provides the ``torchx`` CLI and the
:py:class:`~torchx.runner.Runner` Python API):

.. code:: shell-session

    $ pip install "torchx[dev]"

Verify the installation:

.. code:: shell-session

    $ torchx --help

Hello World
-----------

Create a simple ``my_app.py``:

.. code-block:: python

    import sys

    print(f"Hello, {sys.argv[1]}!")

Launching
---------

Launch the app with ``torchx run``. The :term:`scheduler <Scheduler>` is the
backend that runs the job -- ``local_cwd`` runs it in your current directory.
You'll use the ``utils.python`` :term:`component <Component>` (a reusable job
template):

.. code:: shell-session

    $ torchx run --scheduler local_cwd utils.python --help

The component takes a script name; extra arguments are passed through to the
script.

.. code:: shell-session

    $ torchx run --scheduler local_cwd utils.python --script my_app.py "your name"

Using the Python API
^^^^^^^^^^^^^^^^^^^^

The same operations are available via :py:func:`~torchx.runner.get_runner`:

.. code-block:: python

    from torchx.runner import get_runner

    with get_runner() as runner:
        app_handle = runner.run_component(
            "utils.python",
            ["--script", "my_app.py", "your name"],
            scheduler="local_cwd",
        )
        # Wait for the job to complete and print its final status
        final_status = runner.wait(app_handle, wait_interval=1)
        print(final_status)

You can also construct an :py:class:`~torchx.specs.AppDef` directly and pass
it to :py:meth:`~torchx.runner.Runner.run`:

.. code-block:: python

    import torchx.specs as specs
    from torchx.runner import get_runner

    app = specs.AppDef(
        name="hello",
        roles=[
            specs.Role(
                name="worker",
                entrypoint="python",
                # "image" is the base runtime environment. For local schedulers
                # it's a filesystem path; for container schedulers it's a Docker
                # image name (e.g. "my_image:latest").
                image="/tmp",
                args=["my_app.py", "your name"],
            )
        ],
    )

    with get_runner() as runner:
        app_handle = runner.run(app, scheduler="local_cwd")

The ``local_docker`` scheduler packages your local workspace as a layer on top
of the specified image -- a close approximation of remote container environments.

.. note::

   This requires Docker installed and won't work in environments such as Google
   Colab. See the Docker install instructions:
   https://docs.docker.com/get-docker/

.. code:: shell-session

    $ torchx run --scheduler local_docker utils.python --script my_app.py "your name"

TorchX defaults to using the
`ghcr.io/pytorch/torchx <https://ghcr.io/pytorch/torchx>`_ Docker container image
which contains the PyTorch libraries, TorchX and related dependencies.

Distributed
-----------

The ``dist.ddp`` component (DDP = Distributed Data Parallel) uses
`TorchElastic <https://pytorch.org/docs/stable/distributed.elastic.html>`_
to manage workers, enabling multi-node jobs on all supported schedulers.

.. code:: shell-session

    $ torchx run --scheduler local_docker dist.ddp --help

Create ``dist_app.py``:

.. code-block:: python

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    print(f"I am worker {dist.get_rank()} of {dist.get_world_size()}!")

    a = torch.tensor([dist.get_rank()])
    dist.all_reduce(a)
    print(f"all_reduce output = {a}")

Launch with 2 nodes and 2 workers per node (``-j 2x2`` = ``<nodes>x<workers_per_node>``):

.. code:: shell-session

    $ torchx run --scheduler local_docker dist.ddp -j 2x2 --script dist_app.py

Workspaces / Patching
---------------------

TorchX uses **workspaces** to automatically overlay your local code onto the
job's base image, so you don't need to rebuild and push a Docker image after
every code change. See :doc:`workspace` for details.

``.torchxconfig``
-----------------

Configure scheduler defaults in a ``.torchxconfig`` file instead of passing
``-cfg`` flags every time:

.. code-block:: ini

    [kubernetes]
    queue=torchx
    image_repo=<your docker image repository>

    [slurm]
    partition=torchx

Remote Schedulers
-----------------

The same ``torchx run`` command works on remote schedulers -- only the
``--scheduler`` flag changes.

.. code:: shell-session

    $ torchx run --scheduler slurm dist.ddp -j 2x2 --script dist_app.py
    $ torchx run --scheduler kubernetes dist.ddp -j 2x2 --script dist_app.py
    $ torchx run --scheduler aws_batch dist.ddp -j 2x2 --script dist_app.py

List all scheduler-specific options:

.. code:: shell-session

    $ torchx runopts

Custom Images
-------------

Docker-based Schedulers
^^^^^^^^^^^^^^^^^^^^^^^

Provide a custom Dockerfile to add libraries beyond the standard PyTorch set.

Create ``timm_app.py``:

.. code-block:: python

    import timm

    print(timm.models.resnet18())

Create ``Dockerfile.torchx``:

.. code-block:: dockerfile

    FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

    RUN pip install timm

    COPY . .

TorchX uses this Dockerfile automatically:

.. code:: shell-session

    $ torchx run --scheduler local_docker utils.python --script timm_app.py

Slurm
^^^^^

The ``slurm`` and ``local_cwd`` schedulers use the current environment, so
``pip`` and ``conda`` work as usual.

Next Steps
----------

1. Explore the :doc:`API Quick Reference <api_reference>` for copy-pasteable recipes
2. Explore the :doc:`torchx CLI <cli>` and the :doc:`Runner Python API <runner>`
3. Review :doc:`supported schedulers <schedulers>`
4. Browse :doc:`builtin components <components/overview>`

.. seealso::

   :doc:`basics`
      Core concepts behind AppDef, Component, Runner, and Scheduler.

   :doc:`runner.config`
      Configuring scheduler options via ``.torchxconfig``.

   :doc:`custom_components`
      Writing and registering your own components.
