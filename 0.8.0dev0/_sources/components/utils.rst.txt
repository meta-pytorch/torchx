Utils
================

.. automodule:: torchx.components.utils
.. currentmodule:: torchx.components.utils

.. autofunction:: echo
.. autofunction:: touch
.. autofunction:: sh
.. autofunction:: copy
.. autofunction:: python

``utils.python`` works like regular ``python`` but supports remote launches.
The ``torchx run`` command patches (overlays) your current working directory
onto the image so local changes are reflected in the remote job.

.. code-block:: bash

    # run inline code locally
    $ torchx run utils.python -c "import torch; print(torch.__version__)"

    # run a module locally
    $ torchx run utils.python -m foo.bar.main

    # run a script locally
    $ torchx run utils.python --script my_app.py

    # run on Kubernetes
    $ torchx run -s kubernetes utils.python --script my_app.py

.. important::

   * Be careful with ``-c CMD`` -- schedulers have a character limit on
     arguments. Prefer ``-m`` or ``--script`` for anything non-trivial.
   * Exactly **one** of ``-m``, ``-c``, or ``--script`` must be specified.


.. autofunction:: booth
.. autofunction:: binary
.. autofunction:: hydra

``utils.hydra`` builds an :py:class:`~torchx.specs.AppDef` from a
`Hydra <https://hydra.cc>`_ config, letting you declare jobs as YAML with
config groups, interpolation, and CLI overrides instead of Python kwargs.

.. important::

   ``utils.hydra`` requires ``hydra-core`` (which also pulls in ``omegaconf``).
   It is **not** installed by default -- install it explicitly:

   .. code-block:: bash

       pip install hydra-core

The config must have an ``app`` key whose ``_target_`` is
``torchx.specs.AppDef``. Example ``.torchx/my_job.yaml``:

.. code-block:: yaml

    app:
      _target_: torchx.specs.AppDef
      name: my_job
      roles:
        - _target_: torchx.specs.Role
          name: trainer
          image: alpine:latest
          entrypoint: echo
          num_replicas: 1
          args:
            - hello

Run it (with optional Hydra-style overrides after ``--``):

.. code-block:: bash

    # uses config_dir=.torchx by default
    $ torchx run utils.hydra -cn my_job

    # override any field from the CLI
    $ torchx run utils.hydra -cn my_job -- app.roles.0.num_replicas=3

TorchX macros are exposed as OmegaConf resolvers and can be referenced from
configs: ``${torchx.app_id:}``, ``${torchx.replica_id:}``,
``${torchx.rank0_env:}``, ``${torchx.img_root:}``.

