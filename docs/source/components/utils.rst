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
