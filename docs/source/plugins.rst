torchx.plugins
================

.. automodule:: torchx.plugins
   :no-members:

Core API
---------

.. autofunction:: torchx.plugins.registry

.. autoclass:: torchx.plugins.PluginRegistry
   :members: get, info

.. autoclass:: torchx.plugins.PluginType
   :members:

Registration
-------------

.. autoclass:: torchx.plugins.register
   :members: scheduler, tracker, named_resource

Fractional Helpers
-------------------

.. autofunction:: torchx.plugins.powers_of_two_gpus
.. autofunction:: torchx.plugins.halve_mem_down_to

Constants
----------

.. autodata:: torchx.plugins.WHOLE
.. autodata:: torchx.plugins.HALF
.. autodata:: torchx.plugins.QUARTER
.. autodata:: torchx.plugins.EIGHTH
.. autodata:: torchx.plugins.SIXTEENTH
