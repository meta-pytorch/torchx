torchx.workspace
================

.. tip::

   Workspaces handle automatic image patching -- copying local code changes into
   the job's runtime environment so users don't need manual image rebuilds.
   Scheduler authors add workspace support by mixing in a ``WorkspaceMixin``
   subclass alongside their ``Scheduler``.

**Why workspaces exist.** Without workspaces, every code change requires
rebuilding and pushing a Docker image before submitting a remote job.
Workspaces automate this: TorchX overlays local changes onto the base
:term:`image <Image>` and submits the patched image in one step.

How It Works
--------------

.. code-block:: text

   User's workspace directory            Scheduler submission
   ┌──────────────────────┐
   │  src/                │
   │  train.py            │     build_workspaces()
   │  Dockerfile.torchx   │ ──────────────────────────┐
   │  .torchxignore       │                           │
   └──────────────────────┘                           ▼
                                  ┌──────────────────────────────────┐
                                  │  For each Role with a workspace: │
                                  │                                  │
                                  │  1. Walk workspace               │
                                  │     (respecting .torchxignore)   │
                                  │                                  │
                                  │  2. Build patched artifact       │
                                  │     ┌────────────────────────┐   │
                                  │     │ DockerWorkspaceMixin:  │   │
                                  │     │   docker build + push  │   │
                                  │     │ DirWorkspaceMixin:     │   │
                                  │     │   copy to shared dir   │   │
                                  │     └────────────────────────┘   │
                                  │                                  │
                                  │  3. Mutate role.image in-place   │
                                  │     to reference patched image   │
                                  └──────────────┬───────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────────┐
                                  │  Scheduler._submit_dryrun(app)   │
                                  │  uses the patched role.image     │
                                  └──────────────────────────────────┘

When ``workspace=`` is passed to :py:meth:`~torchx.runner.Runner.run` (or
``--workspace`` on the CLI), TorchX patches the image before submission:

1. :py:meth:`~torchx.workspace.WorkspaceMixin.build_workspaces` iterates over
   each role's :py:attr:`~torchx.specs.Role.workspace`.
2. For each role with a workspace, it calls
   :py:meth:`~torchx.workspace.WorkspaceMixin.caching_build_workspace_and_update_role`,
   which builds the workspace and mutates ``role.image`` in-place to
   reference the patched artifact.
3. For remote schedulers,
   :py:meth:`~torchx.workspace.WorkspaceMixin.dryrun_push_images` and
   :py:meth:`~torchx.workspace.WorkspaceMixin.push_images` handle pushing
   the built image to a remote registry.

.. note::

   ``DockerWorkspaceMixin`` uses ``Dockerfile.torchx`` from the workspace root
   (if present) instead of the default Dockerfile.

Built-in Mixins
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Mixin
     - Strategy
   * - :py:class:`~torchx.workspace.docker_workspace.DockerWorkspaceMixin`
     - Builds a Docker image from a ``Dockerfile.torchx`` in the workspace,
       tags it with a content hash, and pushes to the configured
       ``image_repo``. Used by ``kubernetes``, ``aws_batch``, ``local_docker``.
   * - :py:class:`~torchx.workspace.dir_workspace.DirWorkspaceMixin`
     - Copies workspace files into a shared job directory on the filesystem.
       Used by ``slurm``, ``lsf``.

Implementing a Custom WorkspaceMixin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subclass :py:class:`~torchx.workspace.WorkspaceMixin` and implement
:py:meth:`~torchx.workspace.WorkspaceMixin.caching_build_workspace_and_update_role`:

.. code-block:: python

   from typing import Any, Mapping

   from torchx.specs import CfgVal, Role, runopts
   from torchx.workspace import WorkspaceMixin


   class MyWorkspaceMixin(WorkspaceMixin[None]):
       """Patches images by uploading workspace to a custom artifact store."""

       def workspace_opts(self) -> runopts:
           opts = runopts()
           opts.add("artifact_bucket", type_=str, required=True, help="S3 bucket for workspace artifacts")
           return opts

       def caching_build_workspace_and_update_role(
           self,
           role: Role,
           cfg: Mapping[str, CfgVal],
           build_cache: dict[object, object],
       ) -> None:
           workspace = role.workspace
           if not workspace:
               return

           bucket = cfg.get("artifact_bucket")
           # ... upload workspace files to bucket ...
           # ... update role.image or role.env to reference the artifact ...
           role.env["WORKSPACE_ARTIFACT"] = f"s3://{bucket}/{role.name}/workspace.tar.gz"

Then mix it into your scheduler:

.. code-block:: python

   from torchx.schedulers.api import Scheduler

   class MyScheduler(MyWorkspaceMixin, Scheduler[Mapping[str, CfgVal]]):
       def __init__(self, session_name: str, **kwargs: object) -> None:
           super().__init__("my_backend", session_name)
       # ... scheduler methods ...

The generic parameter ``T`` is the type returned by ``dryrun_push_images`` and
consumed by ``push_images``. Use ``None`` if no separate push step is needed.

.. note::

   The *build_cache* dict is shared across all roles in a single
   ``build_workspaces`` call. Use it to skip redundant builds when roles share
   the same image and workspace.

Testing Your Workspace Mixin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a ``Role`` with a workspace and assert the role was mutated correctly:

.. code-block:: python

   import unittest
   from torchx.specs import Role, Resource, Workspace

   class MyWorkspaceMixinTest(unittest.TestCase):
       def test_build_updates_role(self) -> None:
           mixin = MyWorkspaceMixin()
           role = Role(
               name="worker", image="base:latest", entrypoint="echo",
               resource=Resource(cpu=1, gpu=0, memMB=512),
               workspace=Workspace.from_str("/tmp/my_workspace"),
           )
           mixin.caching_build_workspace_and_update_role(
               role, cfg={"artifact_bucket": "my-bucket"}, build_cache={},
           )
           self.assertIn("WORKSPACE_ARTIFACT", role.env)

See ``torchx/workspace/test/`` for the built-in mixin tests.

Common Pitfalls
^^^^^^^^^^^^^^^^^

* **Implementing the deprecated method**: Override
  ``caching_build_workspace_and_update_role`` (not the older
  ``build_workspace_and_update_role``).

* **MRO**: List the mixin **first** in the base class list:
  ``class MyScheduler(MyWorkspaceMixin, Scheduler[...]):``. Python's MRO
  requires cooperative ``super().__init__()``.

* **Forgetting to check ``role.workspace``**: Guard with
  ``if not role.workspace: return`` -- the method is called for every role.

``.torchxignore``
^^^^^^^^^^^^^^^^^^^

Place a ``.torchxignore`` file (same syntax as ``.dockerignore``) at the
workspace root to exclude files from the job image:

.. code-block:: text

   # Exclude version control and IDE files
   .git
   .vscode
   __pycache__

   # Exclude data directories
   data/
   *.csv

   # But include a specific config file
   !data/config.yaml

Lines starting with ``!`` negate a previous pattern (include the file even if a
prior rule excluded it). Blank lines and lines starting with ``#`` are ignored.

API Reference
--------------

.. currentmodule:: torchx.workspace

.. autoclass:: WorkspaceMixin
  :members:

.. autofunction:: walk_workspace

torchx.workspace.docker_workspace
---------------------------------------

.. automodule:: torchx.workspace.docker_workspace
  :noindex:
.. currentmodule:: torchx.workspace.docker_workspace

.. autoclass:: DockerWorkspaceMixin
  :members:
  :show-inheritance:
  :noindex:

torchx.workspace.dir_workspace
---------------------------------------

.. automodule:: torchx.workspace.dir_workspace
.. currentmodule:: torchx.workspace.dir_workspace

.. autoclass:: DirWorkspaceMixin
  :members:
  :show-inheritance:

.. fbcode::

   torchx.workspace.fb.jetter_workspace
   ---------------------------------------

   .. automodule:: torchx.workspace.fb.jetter_workspace
   .. currentmodule:: torchx.workspace.fb.jetter_workspace

   .. autoclass:: JetterWorkspaceMixin
     :members:
     :show-inheritance:
     :noindex:

   torchx.workspace.fb.conda_env_workspace
   ------------------------------------------

   .. automodule:: torchx.workspace.fb.conda_env_workspace
   .. currentmodule:: torchx.workspace.fb.conda_env_workspace

   .. autoclass:: CondaEnvWorkspace
     :members:
     :show-inheritance:
     :noindex:

   torchx.workspace.fb.sapling_workspace
   ----------------------------------------

   .. automodule:: torchx.workspace.fb.sapling_workspace
   .. currentmodule:: torchx.workspace.fb.sapling_workspace

   .. autoclass:: SaplingWorkspace
     :members:
     :show-inheritance:
     :noindex:

.. seealso::

   :doc:`schedulers`
      Scheduler API reference and implementation guide (including workspace integration).

   :doc:`advanced`
      Registering custom schedulers and workspace mixins via entry points.
