Distributed
==============

.. automodule:: torchx.components.dist
.. currentmodule:: torchx.components.dist

.. autofunction:: torchx.components.dist.torchrun

Deprecated aliases
------------------

``dist.ddp`` and ``dist.spmd`` emit a ``UserWarning`` and forward to
``dist.torchrun`` while retaining their existing signatures and defaults.
See :doc:`/deprecations` for migration details.

.. autofunction:: torchx.components.dist.ddp

.. autofunction:: torchx.components.dist.spmd
