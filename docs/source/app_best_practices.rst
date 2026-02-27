App Best Practices
====================

.. tip::

   This page covers practical recommendations for TorchX applications:
   portable storage with fsspec, train loop frameworks, checkpointing, metrics,
   and testing. See :ref:`component_best_practices:Component Best Practices`
   for component authoring guidance.

**Prerequisites:** :doc:`quickstart` and :doc:`basics`.

Data Passing and Storage
--------------------------

Use `fsspec (Filesystem Spec) <https://filesystem-spec.readthedocs.io>`__ for
storage access. fsspec provides a unified interface to many storage backends
(local, S3, GCS, etc.), so apps run on different infrastructures by changing
paths alone.

.. code-block:: python

    import fsspec

    with fsspec.open("s3://bucket/data.pt", "rb") as f:
        data = torch.load(f, weights_only=True)

Train Loops
-------------

Common choices:

* Pure PyTorch
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`__
* `PyTorch Ignite <https://github.com/pytorch/ignite>`__

See :ref:`components/train:Train` for more information.

Metrics
----------------

Use `TensorBoard <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>`__
for logging metrics. TensorBoard supports remote storage (S3, GCS) for viewing
metrics during training.

See :ref:`components/metrics:Metrics` for TorchX metric components.

Checkpointing
----------------

Periodic checkpoints enable failure recovery and training resumption. PyTorch
Lightning provides
`standardized checkpointing <https://lightning.ai/docs/pytorch/stable/common/checkpointing.html>`__.

Fine Tuning
-------------

Provide a command line argument to your app that resumes from a checkpoint file.
This enables transfer learning, fine tuning, and failure recovery with a single
application.

Interpretability
----------------

Use `Captum <https://captum.ai/>`__ for model interpretability.
See :ref:`components/interpret:Interpret` for built-in components.

Model Packaging
-----------------

Python + Saved Weights
^^^^^^^^^^^^^^^^^^^^^^^^^

The most common format. Load a model definition from Python, then load weights
from a ``.ckpt`` or ``.pt`` file.

TorchScript Models
^^^^^^^^^^^^^^^^^^^^^^

Serializable, optimized models executable without Python.
See the `TorchScript docs <https://pytorch.org/docs/stable/jit.html>`__.

``torch.export``
^^^^^^^^^^^^^^^^^

PyTorch's modern export path for production deployment. Produces
`ExportedProgram <https://pytorch.org/docs/stable/export.html>`__ artifacts
that work with ``torch.compile`` and inference runtimes.

Serving / Inference
---------------------

Use `TorchServe <https://github.com/pytorch/serve>`_ for standard use cases.
See :ref:`components/serve:Serve` for built-in components.

Testing
---------

Since TorchX apps are standard Python, test them like any other Python code:

.. code-block:: python

    import unittest
    from your.custom.app import main

    class CustomAppTest(unittest.TestCase):
        def test_main(self) -> None:
            main(["--src", "src", "--dst", "dst"])
            self.assertTrue(...)

.. seealso::

   :doc:`component_best_practices`
      Best practices for authoring reusable TorchX components.

   :doc:`quickstart`
      Getting started with TorchX from scratch.
