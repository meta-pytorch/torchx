:github_url: https://github.com/meta-pytorch/torchx

TorchX
==================

TorchX is a universal job launcher for PyTorch applications. Define your job
once and run it on any supported backend -- from your laptop to Kubernetes to
Slurm clusters -- without rewriting configuration for each environment.

**Why TorchX?**

* **Write once, run anywhere.** The same ``torchx run`` command (or
  :py:class:`~torchx.runner.Runner` call) works across all schedulers. Switch
  from local development to production clusters by changing a single flag.
* **No vendor lock-in.** TorchX abstracts the scheduler, so your job
  definitions stay portable across Kubernetes, Slurm, AWS Batch, and more.
* **Batteries included.** A built-in :ref:`components library <Components>`
  provides ready-made launchers for distributed training, inference, and
  common utilities -- so you don't start from scratch.
* **Zero runtime dependency.** Your application has no dependency on TorchX
  at runtime. TorchX is only needed at launch time.

.. tip:: **New to TorchX?**

   Follow the recommended reading order:

   1. :doc:`quickstart` -- install, write a simple app, and launch it (10 min)
   2. :doc:`api_reference` -- the Python API at a glance (imports, types, recipes)
   3. :doc:`basics` -- core concepts: AppDef, Component, Runner, Scheduler
   4. :doc:`custom_components` -- write your own reusable component
   5. :doc:`advanced` -- register plugins and extend TorchX


In 1-2-3
-----------------

Step 1. Install

.. code-block:: shell

   pip install torchx[dev]

Step 2. Run Locally

.. code-block:: python

   import torchx.specs as specs
   from torchx.runner import get_runner

   app = specs.AppDef(
       name="hello",
       roles=[
           specs.Role(
               name="worker",
               entrypoint="python",
               image="/tmp",
               args=["my_app.py", "Hello, localhost!"],
           )
       ],
   )

   with get_runner() as runner:
       app_handle = runner.run(app, scheduler="local_cwd")
       print(runner.status(app_handle))

Or from the CLI:

.. code-block:: shell

   torchx run --scheduler local_cwd utils.python --script my_app.py "Hello, localhost!"

Step 3. Run Remotely -- only the ``scheduler`` argument changes:

.. code-block:: python

   with get_runner() as runner:
       app_handle = runner.run(app, scheduler="kubernetes")

.. code-block:: shell

   torchx run --scheduler kubernetes utils.python --script my_app.py "Hello, Kubernetes!"


Ecosystem
-----------

TorchX is part of the `PyTorch <https://pytorch.org>`_ ecosystem. It
complements -- rather than replaces -- other PyTorch projects:

* **TorchElastic** handles fault-tolerant distributed training *within* a job.
  TorchX launches the job itself and the built-in ``dist.ddp`` component
  uses TorchElastic under the hood.
* **TorchServe** serves models in production. TorchX can launch TorchServe
  instances on remote clusters.
* **TorchRec / TorchVision / TorchAudio** provide domain-specific libraries.
  TorchX launches training and inference jobs that use them.

TorchX does **not** prescribe a training framework, model architecture, or data
pipeline. It operates at the job-launching layer and works with any Python
application. For workflow orchestration (DAGs of jobs), integrate TorchX with
`Airflow <https://airflow.apache.org/>`_ or
`Kubeflow Pipelines <https://www.kubeflow.org/docs/components/pipelines/>`_.
For hyperparameter tuning, use `Ax <https://ax.dev/>`_ with TorchX as the
trial launcher. See :ref:`basics:When to Use TorchX (and When Not To)` for a
detailed comparison with alternatives.


Documentation
---------------

.. _torchx.api:
.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference
   specs
   runner
   schedulers
   workspace

.. toctree::
   :maxdepth: 1
   :caption: Guides

   quickstart
   basics
   custom_components
   advanced
   cli
   runner.config
   tracker

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Guides (Meta)

      fb/quickstart
      fb/setup
      fb/images
      fb/workspace
      fb/cogwheel
      fb/named_resources
      fb/tracker_usage
      fb/troubleshooting


Works With
---------------

.. _Schedulers:

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Schedulers (Meta)
      :glob:

      schedulers/fb/*

.. toctree::
   :maxdepth: 1
   :caption: Schedulers

   schedulers/local
   schedulers/docker
   schedulers/kubernetes
   schedulers/kubernetes_mcad
   schedulers/slurm
   schedulers/aws_batch
   schedulers/aws_sagemaker
   schedulers/lsf

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Workspaces (Meta)

      workspaces/fb/sapling
      workspaces/fb/conda_env
      workspaces/fb/jetter

.. toctree::
   :maxdepth: 1
   :caption: Workspaces

   workspaces/docker

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Pipelines (Meta)
      :glob:

      pipelines/fb/*


Runtime Library
----------------

.. toctree::
   :maxdepth: 1
   :caption: Application (Runtime)

   runtime/overview
   runtime/tracking

.. toctree::
   :maxdepth: 1
   :caption: Best Practices

   app_best_practices
   component_best_practices
