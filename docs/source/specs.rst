torchx.specs
======================

.. tip::

   API reference for TorchX's core data types. For a conceptual overview see
   :doc:`basics`. For copy-pasteable recipes see :doc:`api_reference`.

.. automodule:: torchx.specs
.. currentmodule:: torchx.specs


AppDef
------------

.. autoclass:: AppDef
   :members:

Role
------------

.. autoclass:: Role
   :members:

.. autoclass:: RetryPolicy
   :members:

Resource
------------

.. autoclass:: Resource
   :members:

.. autofunction:: resource

Workspace
------------

.. autoclass:: Workspace
   :members:

Macros
------------

.. currentmodule:: torchx.specs
.. autoclass:: macros
   :members:

.. note::

   In addition to the three macros listed in the class docstring, two more
   attributes exist:

   * ``macros.rank0_env`` -- expands to the **name** of the environment variable
     that provides the rank-0 (master) host address. Resolve it via shell
     expansion (``$${rank0_env}``) or in application code. Not available on all
     schedulers.
   * ``macros.base_img_root`` -- **deprecated**. Do not use in new code.

Run Configs
--------------

.. autodata:: CfgVal

   Type alias for run config values:
   ``str | int | float | bool | list[str] | dict[str, str] | None``.
   Used in ``cfg`` dicts passed to :py:meth:`~torchx.runner.Runner.run` and
   scheduler methods.

.. autoclass:: runopts
   :members:

.. autoclass:: runopt
   :members:

Structured Opts
^^^^^^^^^^^^^^^^^^

.. currentmodule:: torchx.schedulers.api

.. autoclass:: StructuredOpts
   :members:
   :noindex:

.. currentmodule:: torchx.specs

Run Status
--------------

.. autoclass:: AppStatus
   :members:

.. autoclass:: AppState
   :members:

.. autoclass:: ReplicaState
   :members:

.. autoclass:: AppDryRunInfo
   :members:

App Handle
--------------

.. autodata:: AppHandle

.. autofunction:: parse_app_handle

.. autoclass:: ParsedAppHandle
   :members:

Mounts
--------

.. autofunction:: parse_mounts

.. autoclass:: BindMount
   :members:

.. autoclass:: VolumeMount
   :members:

.. autoclass:: DeviceMount
   :members:

Overlays
------------

.. automodule:: torchx.specs.overlays
   :members:

.. fbcode::

   MAST/MSL Overlays
   ^^^^^^^^^^^^^^^^^^^^

   .. automodule:: torchx.specs.fb.overlay_mast
      :members:

Named Resources
-----------------

Use :py:func:`resource` with the ``h`` parameter to look up a named resource::

    from torchx.specs import resource
    resource(h="gpu.small")   # generic t-shirt size
    resource(h="aws_p3.2xlarge")  # AWS instance type

See :ref:`advanced:Registering Named Resources` for defining custom named
resources.

Generic Named Resources
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torchx.specs.named_resources_generic

These are cloud-agnostic, t-shirt-sized defaults. The exact cpu/gpu/memory
values may change between releases -- define your own named resources for
production workloads.

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 15

   * - Name
     - CPU
     - GPU
     - Memory
   * - ``gpu.small``
     - 8
     - 1
     - 32 GiB
   * - ``gpu.medium``
     - 16
     - 2
     - 64 GiB
   * - ``gpu.large``
     - 32
     - 4
     - 128 GiB
   * - ``gpu.xlarge``
     - 64
     - 8
     - 256 GiB
   * - ``cpu.nano``
     - 1
     - 0
     - 512 MiB
   * - ``cpu.micro``
     - 1
     - 0
     - 1 GiB
   * - ``cpu.small``
     - 1
     - 0
     - 2 GiB
   * - ``cpu.medium``
     - 2
     - 0
     - 4 GiB
   * - ``cpu.large``
     - 2
     - 0
     - 8 GiB
   * - ``cpu.xlarge``
     - 8
     - 0
     - 32 GiB

AWS Named Resources
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torchx.specs.named_resources_aws
   :members:

Component Linter
-----------------
.. automodule:: torchx.specs.file_linter
.. currentmodule:: torchx.specs.file_linter

.. autofunction:: validate
.. autofunction:: get_fn_docstring

.. autoclass:: LinterMessage
   :members:

.. seealso::

   :doc:`api_reference`
      Single-page reference with imports, types, and copy-pasteable recipes.

   :doc:`runner`
      The Runner API that submits AppDefs as jobs.

   :doc:`advanced`
      Registering named resources, custom components, and other plugins.
