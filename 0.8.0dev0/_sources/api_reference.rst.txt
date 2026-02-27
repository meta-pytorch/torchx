Quick Reference
====================

.. tip::

   Imports, core types, Runner lifecycle, and copy-pasteable recipes on one
   page. For detailed API docs see :doc:`specs` and :doc:`runner`.


Imports
--------

.. code-block:: python

   # Core types — job definitions
   import torchx.specs as specs
   from torchx.specs import AppDef, Role, Resource, AppState, macros

   # Named resources — t-shirt-sized hardware presets
   from torchx.specs import resource

   # Runner — submit, monitor, and manage jobs
   from torchx.runner import get_runner

   # Mounts — bind, volume, and device mounts
   from torchx.specs import BindMount, VolumeMount, DeviceMount


Core Types
-----------

AppDef
^^^^^^^^

A job definition containing one or more :py:class:`~torchx.specs.Role`\s.

.. code-block:: python

   app = AppDef(
       name="my_job",             # str — job name
       roles=[...],               # list[Role] — one or more roles
       metadata={},               # dict[str, str] — scheduler-specific metadata
   )

Role
^^^^^^

A set of identical replicas within an AppDef.

.. code-block:: python

   role = Role(
       name="trainer",            # str — role name
       image="my_image:latest",   # str — Docker image, fbpkg, or path
       entrypoint="python",       # str — command to run
       args=["-m", "my_app"],     # list[str] — arguments to entrypoint
       env={"KEY": "value"},      # dict[str, str] — environment variables
       num_replicas=1,            # int — number of container replicas
       resource=Resource(         # Resource — hardware requirements per replica
           cpu=4, gpu=1, memMB=8192,
       ),
       # Optional fields:
       # min_replicas=1,          # int | None — minimum for elastic scaling
       # max_retries=3,           # int — retries before giving up
       # retry_policy=RetryPolicy.APPLICATION,
       # port_map={"tb": 6006},   # dict[str, int] — named port mappings
       # mounts=[...],            # list[BindMount | VolumeMount | DeviceMount]
       # workspace=Workspace(...),
   )

Resource
^^^^^^^^^^

Hardware requirements per replica. Prefer named resources over raw values.

.. code-block:: python

   # Option 1: named resource (preferred)
   from torchx.specs import resource
   res = resource(h="gpu.small")   # 8 CPU, 1 GPU, 32 GiB

   # Option 2: explicit values
   res = Resource(cpu=4, gpu=1, memMB=8192)

   # Option 3: AWS instance type
   res = resource(h="aws_p3.2xlarge")

See :ref:`specs:Named Resources` for the full list.


Runner Lifecycle
------------------

Create a Runner
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchx.runner import get_runner

   # As a context manager (recommended — auto-closes scheduler connections)
   with get_runner() as runner:
       ...

   # Or manually
   runner = get_runner()
   # ... use runner ...
   runner.close()

Submit a Job
^^^^^^^^^^^^^^^

``run_component`` resolves a component by name; ``run`` takes a direct AppDef.

.. code-block:: python

   with get_runner() as runner:
       # Method 1: run_component — resolves a component by name
       # Same resolution as `torchx run` CLI
       app_handle = runner.run_component(
           "dist.ddp",                           # component name
           ["--script", "train.py", "-j", "2x2"], # args (list[str])
           scheduler="kubernetes",                # scheduler backend
           cfg={"namespace": "default"},           # scheduler config (optional)
       )

       # Method 2: run — submit an AppDef directly
       app = AppDef(name="my_job", roles=[...])
       app_handle = runner.run(
           app,
           scheduler="kubernetes",
           cfg={"namespace": "default"},
       )

Monitor and Wait
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Poll status
   status = runner.status(app_handle)
   print(status.state)       # AppState enum (see table below)
   print(status.msg)         # human-readable message
   print(status.ui_url)      # scheduler UI link (if available)

   # Block until terminal state
   final_status = runner.wait(app_handle, wait_interval=10)

   # Check if terminal
   if final_status and final_status.is_terminal():
       print("Done:", final_status.state)

   # Raise an exception if the job did not succeed
   final_status.raise_for_status()   # raises AppStatusError on non-SUCCEEDED

**AppState values:**

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - State
     - Terminal?
     - Description
   * - ``UNSUBMITTED``
     - No
     - Not yet submitted.
   * - ``SUBMITTED``
     - No
     - Submitted to the scheduler.
   * - ``PENDING``
     - No
     - Waiting for resource allocation.
   * - ``RUNNING``
     - No
     - Running.
   * - ``SUCCEEDED``
     - Yes
     - Completed successfully.
   * - ``FAILED``
     - Yes
     - Completed unsuccessfully.
   * - ``CANCELLED``
     - Yes
     - Cancelled before completing.
   * - ``UNKNOWN``
     - No
     - State cannot be determined.

The ``app_handle`` is a URI: ``{scheduler}://{session_name}/{app_id}`` (e.g.
``kubernetes://torchx/my_job_123``). Pass it to all Runner methods.

Cancel and Delete
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   runner.cancel(app_handle)   # request cancellation (async)
   runner.delete(app_handle)   # remove from scheduler

Fetch Logs
^^^^^^^^^^^^

.. code-block:: python

   # Get log lines for replica 0 of the "trainer" role
   for line in runner.log_lines(app_handle, role_name="trainer", k=0):
       print(line, end="")    # lines include trailing \n


Common Recipes
----------------

Single-Node Training
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import torchx.specs as specs
   from torchx.runner import get_runner
   from torchx.specs import resource

   app = specs.AppDef(
       name="train",
       roles=[
           specs.Role(
               name="trainer",
               image="my_image:latest",
               entrypoint="python",
               args=["-m", "my_train", "--epochs", "10"],
               resource=resource(h="gpu.small"),
               env={"CUDA_VISIBLE_DEVICES": "0"},
           )
       ],
   )

   with get_runner() as runner:
       app_handle = runner.run(app, scheduler="local_cwd")
       status = runner.wait(app_handle, wait_interval=1)
       print(status)

Distributed Training (DDP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the built-in ``dist.ddp`` component:

.. code-block:: python

   from torchx.runner import get_runner

   with get_runner() as runner:
       app_handle = runner.run_component(
           "dist.ddp",
           [
               "--script", "train.py",
               "-j", "2x2",              # 2 nodes x 2 workers per node
               "--gpu", "2",
               "--memMB", "8192",
           ],
           scheduler="kubernetes",
       )
       status = runner.wait(app_handle)

Or build the AppDef directly for full control:

.. code-block:: python

   import torchx.specs as specs
   from torchx.runner import get_runner

   app = specs.AppDef(
       name="ddp_train",
       roles=[
           specs.Role(
               name="trainer",
               image="my_image:latest",
               entrypoint="python",
               args=[
                   "-m", "torch.distributed.run",
                   "--nnodes", "2",
                   "--nproc_per_node", "2",
                   "train.py",
               ],
               num_replicas=2,
               resource=specs.Resource(cpu=8, gpu=2, memMB=16384),
           )
       ],
   )

   with get_runner() as runner:
       app_handle = runner.run(app, scheduler="kubernetes")

Custom Component
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # my_component.py
   import torchx.specs as specs
   from torchx.specs import resource

   def trainer(
       script: str,
       image: str = "my_image:latest",
       resource_name: str = "gpu.small",
   ) -> specs.AppDef:
       """Launch single-node training."""
       return specs.AppDef(
           name="trainer",
           roles=[
               specs.Role(
                   name="trainer",
                   image=image,
                   entrypoint="python",
                   args=["-m", script],
                   resource=resource(h=resource_name),
               )
           ],
       )

   # Launch it:
   from torchx.runner import get_runner

   with get_runner() as runner:
       # By name (same as CLI: torchx run my_component.py:trainer ...)
       app_handle = runner.run_component(
           "my_component.py:trainer",
           ["--script", "my_train", "--resource_name", "gpu.medium"],
           scheduler="local_cwd",
       )

       # Or call the function directly
       app = trainer(script="my_train", resource_name="gpu.medium")
       app_handle = runner.run(app, scheduler="local_cwd")

Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   role = specs.Role(
       name="trainer",
       image="my_image:latest",
       entrypoint="python",
       args=["-m", "my_train"],
       env={
           "NCCL_DEBUG": "INFO",
           "CUDA_VISIBLE_DEVICES": "0,1",
           "MY_CONFIG": "/data/config.yaml",
       },
   )

Mounts
^^^^^^^^

.. code-block:: python

   from torchx.specs import BindMount, VolumeMount

   role = specs.Role(
       name="trainer",
       image="my_image:latest",
       entrypoint="python",
       args=["-m", "my_train"],
       mounts=[
           # Bind-mount a host directory
           BindMount(src_path="/data/datasets", dst_path="/mnt/data", read_only=True),
           # Mount a persistent volume (Kubernetes PVC = Persistent Volume Claim, etc.)
           VolumeMount(src="my-pvc", dst_path="/mnt/checkpoints"),
       ],
   )

Runtime Macros
^^^^^^^^^^^^^^^^

Substitute scheduler-assigned values into ``args``, ``env``, and ``metadata``
at runtime:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Macro
     - Value
   * - ``macros.app_id``
     - Scheduler-assigned job ID
   * - ``macros.replica_id``
     - Per-role replica index (``0, 1, 2, ...``)
   * - ``macros.img_root``
     - Root directory of the pulled image
   * - ``macros.rank0_env``
     - Name of the env var holding the rank-0 host address
       (resolve via shell expansion or application code; not available on all
       schedulers)

.. code-block:: python

   from torchx.specs import macros

   role = specs.Role(
       name="trainer",
       image="my_image:latest",
       entrypoint="python",
       args=[
           "-m", "my_train",
           "--job_id", macros.app_id,          # scheduler-assigned job ID
           "--replica", macros.replica_id,      # 0, 1, 2, ...
       ],
       env={
           "IMG_ROOT": macros.img_root,         # root dir of pulled image
       },
   )

Scheduler Config
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   with get_runner() as runner:
       app_handle = runner.run(
           app,
           scheduler="kubernetes",
           cfg={
               "namespace": "my-namespace",
               "image_repo": "my-registry.example.com/images",
               "queue": "default",
           },
       )

   # Or via .torchxconfig file:
   # [kubernetes]
   # namespace=my-namespace
   # image_repo=my-registry.example.com/images


Scheduler Reference
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Name
     - Backend
     - Notes
   * - ``local_cwd``
     - Current working directory
     - No container; runs as subprocesses. Good for development.
   * - ``local_docker``
     - Local Docker daemon
     - Builds patched image from local workspace.
   * - ``kubernetes``
     - Kubernetes
     - Creates ``Job`` resources. Requires cluster access.
   * - ``kubernetes_mcad``
     - Kubernetes (MCAD)
     - Uses Multi-Cluster Application Dispatcher (MCAD) for gang scheduling
       (all pods in a job start together or none start).
   * - ``slurm``
     - Slurm HPC
     - Generates ``sbatch`` scripts.
   * - ``aws_batch``
     - AWS Batch
     - Creates job definitions and submits jobs.
   * - ``aws_sagemaker``
     - AWS SageMaker
     - Creates training jobs.
   * - ``lsf``
     - IBM LSF
     - Generates ``bsub`` scripts.


Named Resources Reference
----------------------------

Common cloud-agnostic sizes. For the complete list see
:ref:`specs:Named Resources`.

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 15

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

See :py:mod:`torchx.specs.named_resources_aws` for AWS instance-type
resources (e.g. ``aws_p3.2xlarge``, ``aws_m5.2xlarge``).


Anti-Patterns
--------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Avoid
     - Prefer
   * - Hard-coding ``Resource(cpu=8, gpu=1, memMB=32000)``
     - ``resource(h="gpu.small")`` — portable across environments
   * - ``entrypoint="python /path/to/train.py"``
     - ``entrypoint="python", args=["-m", "my_module"]`` — works across image layouts
   * - ``if env == "prod": ... else: ...`` inside a component
     - Separate ``trainer_dev()`` and ``trainer_prod()`` components with a shared ``_trainer()`` helper
   * - Constructing ``image`` strings inside components
     - Accept ``image: str`` as a parameter — callers control naming
   * - Using ``runner.stop()``
     - ``runner.cancel()`` — ``stop()`` is deprecated
   * - Parsing ``runner.log_lines()`` output programmatically
     - Use scheduler-native log APIs — log completeness is not guaranteed


Plugin Implementor Quick Reference
-------------------------------------

.. note::

   The rest of this page is for **platform engineers** extending TorchX.
   Job authors can stop here. For full extension guides see :doc:`advanced`
   and :doc:`schedulers`.

Condensed skeletons linking to full guides.

Scheduler Plugin
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from dataclasses import dataclass
   from typing import Any, Mapping

   from torchx.schedulers.api import (
       DescribeAppResponse,
       ListAppResponse,
       Scheduler,
       StructuredOpts,
   )
   from torchx.specs import AppDef, AppDryRunInfo, CfgVal, runopts

   @dataclass
   class MyRequest:
       """Native request type — passed from _submit_dryrun to schedule."""
       job_name: str
       cmd: list[str]

   @dataclass
   class MyOpts(StructuredOpts):
       cluster: str = "default"
       """Cluster to submit to."""

   class MyScheduler(Scheduler[Mapping[str, CfgVal]]):
       def __init__(self, session_name: str, **kwargs: object) -> None:
           super().__init__("my_backend", session_name)

       def _run_opts(self) -> runopts:
           return MyOpts.as_runopts()

       def _submit_dryrun(self, app: AppDef, cfg: Mapping[str, CfgVal]) -> AppDryRunInfo[MyRequest]:
           opts = MyOpts.from_cfg(cfg)
           role = app.roles[0]
           return AppDryRunInfo(MyRequest(app.name, [role.entrypoint, *role.args]), repr)

       def schedule(self, dryrun_info: AppDryRunInfo[MyRequest]) -> str:
           request: MyRequest = dryrun_info.request
           return "job-id-123"  # submit to backend, return ID

       def describe(self, app_id: str) -> DescribeAppResponse | None:
           return DescribeAppResponse(app_id=app_id)

       def list(self, cfg: Mapping[str, CfgVal] | None = None) -> list[ListAppResponse]:
           return []

       def _cancel_existing(self, app_id: str) -> None:
           pass  # cancel the job

   def create_scheduler(session_name: str, **kwargs: Any) -> MyScheduler:
       return MyScheduler(session_name)

Abstract methods: ``_submit_dryrun``, ``schedule``, ``describe``, ``list``,
``_cancel_existing``. Optional: ``_run_opts``, ``log_iter``, ``close``,
``_validate``, ``_pre_build_validate``, ``_delete_existing``.

See :ref:`Implementing a Custom Scheduler <implementing-scheduler>` for the
full guide.

Workspace Plugin
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Mapping

   from torchx.specs import CfgVal, Role, runopts
   from torchx.workspace import WorkspaceMixin

   class MyWorkspaceMixin(WorkspaceMixin[None]):
       def workspace_opts(self) -> runopts:
           opts = runopts()
           opts.add("artifact_store", type_=str, required=True, help="Remote artifact store URL")
           return opts

       def caching_build_workspace_and_update_role(
           self,
           role: Role,
           cfg: Mapping[str, CfgVal],
           build_cache: dict[object, object],
       ) -> None:
           if not role.workspace:
               return
           # ... build and upload workspace, then update role ...
           role.env["WORKSPACE_URL"] = f"{cfg.get('artifact_store')}/{role.name}"

Mix into a scheduler: ``class MyScheduler(MyWorkspaceMixin, Scheduler[...]): ...``

See :doc:`workspace` for the full API.

Tracker Plugin
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchx.tracker.api import TrackerBase, TrackerArtifact, TrackerSource, Lineage
   from typing import Iterable, Mapping

   class MyTracker(TrackerBase):
       def __init__(self, connection_str: str) -> None:
           self._conn = connection_str

       # --- write methods ---
       def add_artifact(
           self, run_id: str, name: str, path: str,
           metadata: Mapping[str, object] | None = None,
       ) -> None: ...

       def add_metadata(self, run_id: str, **kwargs: object) -> None: ...

       def add_source(
           self, run_id: str, source_id: str,
           artifact_name: str | None = None,
       ) -> None: ...

       # --- read methods ---
       def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]: ...

       def metadata(self, run_id: str) -> Mapping[str, object]: ...

       def sources(
           self, run_id: str, artifact_name: str | None = None,
       ) -> Iterable[TrackerSource]: ...

       def lineage(self, run_id: str) -> Lineage: ...

       def run_ids(self, **kwargs: str) -> Iterable[str]: ...

   def create(config: str | None) -> TrackerBase:
       return MyTracker(connection_str=config or "default://localhost")

Abstract methods: ``add_artifact``, ``artifacts``, ``add_metadata``,
``metadata``, ``add_source``, ``sources``, ``lineage``, ``run_ids``.

See :ref:`Registering Custom Trackers <registering-custom-trackers>` for the
full guide.

CLI Command Plugin
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import argparse
   from torchx.cli.cmd_base import SubCommand

   class CmdMyTool(SubCommand):
       def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
           subparser.add_argument("--config", type=str, help="Path to config file")
           subparser.add_argument("app_id", type=str, help="Application handle")

       def run(self, args: argparse.Namespace) -> None:
           print(f"Running my_tool on {args.app_id} with config={args.config}")

Abstract methods: ``add_arguments``, ``run``.

See :ref:`Registering Custom CLI Commands <registering-custom-cli-commands>` for
the full guide.

Entry-Point Registration
^^^^^^^^^^^^^^^^^^^^^^^^^^

**setup.py format:**

.. code-block:: python

   # setup.py
   entry_points={
       "torchx.schedulers":       ["my_sched = my_pkg:create_scheduler"],
       "torchx.named_resources":  ["gpu_x4 = my_pkg.resources:gpu_x4"],
       "torchx.components":       ["myco = my_pkg.components"],
       "torchx.tracker":          ["my_tracker = my_pkg.tracking:create"],
       "torchx.cli.cmds":         ["my_tool = my_pkg.cli:CmdMyTool"],
   }

**pyproject.toml format:**

.. code-block:: toml

   [project.entry-points."torchx.schedulers"]
   my_sched = "my_pkg:create_scheduler"

   [project.entry-points."torchx.named_resources"]
   gpu_x4 = "my_pkg.resources:gpu_x4"

   [project.entry-points."torchx.components"]
   myco = "my_pkg.components"

   [project.entry-points."torchx.tracker"]
   my_tracker = "my_pkg.tracking:create"

   [project.entry-points."torchx.cli.cmds"]
   my_tool = "my_pkg.cli:CmdMyTool"

.. note::

   Entry point targets differ by plugin type:

   * **Schedulers**: factory function ``(session_name: str, **kwargs) -> Scheduler``
   * **Named resources**: factory function ``() -> Resource``
   * **Components**: module path (TorchX discovers component functions inside it)
   * **Trackers**: factory function ``(config: str | None) -> TrackerBase``
   * **CLI commands**: ``SubCommand`` class (not a factory -- TorchX calls ``cls()``)

See :doc:`advanced` for the full registration guide.


.. seealso::

   :doc:`specs`
      Full API documentation for AppDef, Role, Resource, and related types.

   :doc:`runner`
      Full API documentation for Runner and get_runner.

   :doc:`schedulers`
      Scheduler API reference and implementation guide.

   :doc:`workspace`
      Workspace API reference and custom workspace mixin guide.

   :doc:`tracker`
      Tracker API reference and backend implementations.

   :doc:`cli`
      CLI module API reference and custom command guide.

   :doc:`custom_components`
      Step-by-step guide for writing and launching custom components.

   :doc:`component_best_practices`
      Best practices for authoring reusable components.
