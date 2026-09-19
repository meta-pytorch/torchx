torchx.deprecations
=====================

.. automodule:: torchx.deprecations

.. autofunction:: torchx.deprecations.deprecated_module

.. autofunction:: torchx.deprecations.deprecated

Distributed components
----------------------

Use :func:`torchx.components.dist.torchrun` to launch distributed PyTorch
applications. ``dist.ddp`` and ``dist.spmd`` remain available as deprecated
aliases and emit a ``UserWarning`` on each call.

* ``dist.torchrun`` retains the ``dist.ddp`` defaults: ``j="1x2"``, no named
  host, two CPUs, no GPUs, and 1024 MB of memory per replica.
* To preserve ``dist.spmd`` defaults, pass ``h="gpu.small", j="1x1"``.
* With a named host, ``dist.torchrun`` treats a bare ``j`` as the node count
  and infers processes per node from the host's GPU count, as ``dist.spmd``
  does. Specify ``NxP`` for a host without GPUs.
* ``dist.ddp`` retains its original bare-``j`` meaning: processes on one
  node. When migrating ``dist.ddp(h="gpu.small", j="2", ...)``, use
  ``dist.torchrun(h="gpu.small", j="1x2", ...)`` to keep the same topology.
* Explicit ``NxP`` and elastic ``MIN:MAXxP`` topologies are unchanged when
  migrating from ``dist.ddp``.
