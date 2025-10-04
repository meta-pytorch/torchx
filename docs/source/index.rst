:github_url: https://github.com/meta-pytorch/torchx

TorchX
==================

TorchX is a universal job launcher for PyTorch applications.

**GETTING STARTED?** Follow the :ref:`quickstart guide<quickstart:Quickstart>`.


In 1-2-3
-----------------

Step 1. Install

.. code-block:: shell

   pip install torchx[dev]

Step 2. Run Locally

.. code-block:: shell

   torchx run --scheduler local_cwd utils.python --script my_app.py "Hello, localhost!"

Step 3. Run Remotely

.. code-block:: shell

   torchx run --scheduler kubernetes utils.python --script my_app.py "Hello, Kubernetes!"


Documentation
---------------

.. toctree::
   :maxdepth: 1
   :caption: Usage

   quickstart.md
   cli
   basics
   runner.config
   advanced
   custom_components.md
   tracker

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Usage (Meta)

      fb/quickstart.md
      fb/setup.md
      fb/images.md
      fb/workspace.md
      fb/cogwheel.rst
      fb/named_resources.rst
      fb/tracker.md
      fb/troubleshooting.md


Works With
---------------

.. _Schedulers:
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
      :caption: Schedulers (Meta)
      :glob:

      schedulers/fb/*

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :caption: Pipelines (Meta)
      :glob:

      pipelines/fb/*


Examples
------------
.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples_apps/index


Components Library
---------------------
.. _Components:
.. toctree::
   :maxdepth: 1
   :caption: Components

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

Runtime Library
----------------
.. toctree::
   :maxdepth: 1
   :caption: Application (Runtime)

   runtime/overview
   runtime/tracking


Reference
-----------

.. _torchx.api:
.. toctree::
   :maxdepth: 1
   :caption: API

   specs
   runner
   schedulers
   workspace

.. toctree::
   :maxdepth: 1
   :caption: Best Practices

   app_best_practices
   component_best_practices
