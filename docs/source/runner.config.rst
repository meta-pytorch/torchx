.torchxconfig
=============================

.. tip::

   ``.torchxconfig`` is an INI-format file where you store default scheduler
   options (queue names, image repositories, etc.) so you don't have to pass
   them on every ``torchx run`` invocation. Place it in your project root or
   home directory.

**Prerequisites:** :doc:`quickstart` (basic CLI usage).

.. automodule:: torchx.runner.config
.. currentmodule:: torchx.runner.config

Config API Functions
----------------------

.. autofunction:: apply
.. autofunction:: load
.. autofunction:: dump
.. autofunction:: find_configs
.. autofunction:: get_configs
.. autofunction:: get_config
.. autofunction:: load_sections

.. seealso::

   :doc:`runner`
      The Runner API that uses these configuration values.

   :doc:`schedulers`
      Scheduler-specific options that can be set in ``.torchxconfig``.
