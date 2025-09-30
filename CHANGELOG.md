# CHANGELOG

## torchx-0.7.0

* `torchx.schedulers`
  * AWS Batch Scheduler
    * Add job_role_arn and execution_role_arn as run options for AWS permission
    * Add instance type for aws_batch_scheduler multinode jobs
    * Add neuron device mount for aws trn instances 
    * Update EFA_DEVICE details for AWS resources
  * Add aws_sagemaker_scheduler
    * Torchx scheduler for AWS SageMaker
  * Docker Scheduler
    * Add support for setting environment variables
  * GCP Batch Scheduler
    * Fix log
 
* `torchx.tracker`
  * Add module lookup for building trackers

* `torchx.cli`
  * Throw error if there are multiple scheduler arguments
 
* `torchx.specs.api`
  * Add macro support to metadata variables
  * Add NamedTuple for Tuple from parsing AppHandle

* Changes to ease maintenance
  * Migrate all active AWS resources to US-west-1 region with Terraform
  * Fix all Github Actions
  * Better linter error message.
  * Migrate Kubernetes Integration test to MiniKube
  * Deprecate KFP Integration Tests
 
* Additional changes
  * Add verbose flag to docker mixin


## torchx-0.6.0

* Breaking changes
  * Drop support for python 3.7.
  * Upgrade docker base image python version to 2.0

* `torchx.schedulers`
  * Add support for options in create_schedulers factory method that allows scheduler configuration in runner
  * Kubernetes MCAD Scheduler
     * Add support for retrying
     * Test, formatting and documentation updates
  * AWS Batch Scheduler
    * Fix logging rank attribution
  * Ray Scheduler
    * Add ability to programmatically define ray job client

* `torchx.tracker`
  * Fix adding artifacts to MLFlowTracker by multiple ranks

* `torchx.components`
  * dst.ddp
    * Add ability to specify rendezvous backend and use c10d as a default mechanism
    * Add node_rank parameter value for static rank setup

* `torchx.runner`
  * Resolve run_opts when passing to `torchx.workpsace` and for dry-run to correctly populate the values

* `torchx.runner.events`
  * Add support to log CPU and wall times

* `torchx.cli`
  * Wait for app to start when logging

* `torchx.specs`
  * Role.resources uses default_factory method to initialize its value


## torchx-0.5.0

* Milestone: https://github.com/meta-pytorch/torchx/milestone/7

* `torchx.schedulers`
  * Kubernetes MCAD Scheduler (Prototype)
    * Newly added integration for easily scheduling jobs on Multi-Cluster-Application-Dispatcher (MCAD).
    * Features include:
      * scheduling different types of components including DDP components
      * scheduling on different compute resources (CPU, GPU)
      * support for docker workspace
      * support for bind, volume and device mounts
      * getting logs for jobs
      * describing, listing and cancelling jobs
      * can be used with a secondary scheduler on Kubernetes
  * AWS Batch
    * Add privileged option to enable running containers on EFA enabled instances with elevated networking permissions

* `torchx.tracker`
  * MLflow backend (Prototype)
    * New support for MLFlow backend for torchx tracker
  * Add ability for fsspec tracker to read nested kwargs
  * Support for tracking apps not launched by torchx
  * Load tracker config from .torchxconfig

* `torchx.components`
    * Add dist.spmd component to support Single-Process-Multiple-Data style applications

* `torchx.workspace`
  * Add ability to access image and workspace path from Dockerfile while building docker workspace

* Usability imporvements
  *  Fix entrypoint loading to deal with deferred loading of modules to enable component registration to work properly

* Changes to ease maintenance
  * Add ability to run integration tests for AWS Batch, Slurm, and Kubernetes, instead of running in a remote dedicated clusters. This makes the environment reproducible, reduces maintenance, and makes it easier for more users to contribute.

* Additional changes
  * Bug fixes: Make it possible to launch jobs with more than 5 nodes on AWS Batch


## torchx-0.4.0

* Milestone: https://github.com/meta-pytorch/torchx/milestone/6

* `torchx.schedulers`
  * GCP Batch (Prototype)
    * Newly added integration for easily scheduling jobs on GCP Batch.
    * Features include:
      * scheduling different types of components including DDP components
      * scheduling on different compute resources (CPU, GPU)
      * describing jobs including getting job status
      * getting logs for jobs
      * listing jobs
      * cancelling jobs
  * AWS Batch
    * Listing jobs now returns just jobs launched on AWS Batch by TorchX and uses pagination to enable listing all jobs in all queues.
    * Named resources now account for ECS and EC2 memtax, and suggests closest match when resource is not found.
    * Named resources expanded to include all instance types for g4d, g5, p4d, p3 and trn1.

* `torchx.workspace`
  * Improve docker push logging to prevent log spamming when pushing for the first time

* Additional Changes
  * Remove classyvision from examples since it's no longer supported in OSS. Uses torchvision/torch dataset APIs instead of ClassyDataset.


## torchx-0.3.0

* Milestone: https://github.com/meta-pytorch/torchx/milestone/5

* `torchx.schedulers`
  * List API (Prototype)
    * New list API to list jobs and their statuses for all schedulers which removes the need to use secondary tools to list jobs
  * AWS Batch (promoted to Beta)
    * Get logs for running jobs
    * Added configs for job priorities and queue policies
    * Easily access job UI via ui_url
  * Ray
    * Add elasticity to jobs launched on ray cluster to automatically scale jobs up as resources become available
  * Kubernetes
    * Add elasticity to jobs launched on Kubernetes
  * LSF Scheduler (Prototype)
    * Newly added support for scheduling on IBM Spectrum LSF scheduler
  * Local Scheduler
    * Better formatting when using pdb

* `torchx.tracker` (Prototype)
    * TorchX Tracker is a new lightweight experiment and artifact tracking tool
    * Add tracker API that can track any inputs and outputs to your model in any infrastructure
    * FSSpec based Torchx tracking implementation and sample app

* `torchx.runner`
    * Allow overriding TORCHX_IMAGE via entrypoints
    * Capture the image used when logging schedule calls

* `torchx.components`
    * Add debug flag to dist component

* `torchx.cli`
    * New list feature also available as a subcommand to list jobs and their statuses on a given scheduler
    * New tracker feature also available as a subcommand to track experiments and artifacts
    * Defer loading schedulers until used

* `torchx.workspace`
    * Preserve Unix file mode when patching files into docker image.

* Docs
    * Add airflow example

* Additional changes
    * Bug fixes for Python 3.10 support


## torchx-0.2.0

* Milestone: https://github.com/meta-pytorch/torchx/milestone/4

* `torchx.schedulers`
    * DeviceMounts
        * New mount type 'DeviceMount' that allows mounting a host device into a container in the supported schedulers (Docker, AWS Batch, K8). Custom accelerators and network devices such as Infiniband or Amazon EFA are now supported.
    * Slurm
        * Scheduler integration now supports "max_retries" the same way that our other schedulers do. This only handles whole job level retries and doesn't support per replica retries.
        * Autodetects "nomem" setting by using `sinfo` to get the "Memory" setting for the specified partition
        * More robust slurmint script
    * Kubernetes
        * Support for k8s device plugins/resource limits
            * Added "devices" list of (str, int) tuples to role/resource
            * Added devices.py to map from named devices to DeviceMounts
            * Added logic in kubernetes_scheduler to add devices from resource to resource limits
            * Added logic in aws_batch_scheduler and docker_scheduler to add DeviceMounts for any devices from resource
        * Added "priority_class" argument to kubernetes scheduler to set the priorityClassName of the volcano job.
    * Ray
        * fixes for distributed training, now supported in Beta

* `torchx.specs`
    * Moved factory/builder methods from datastruct specific "specs.api" to "specs.factory" module

* `torchx.runner`
    * Renamed "stop" method to "cancel" for consistency. `Runner.stop` is now deprecated
    * Added warning message when "name" parameter is specified. It is used as part of Session name, which is deprecated so makes "name" obsolete.
    * New env variable TORCHXCONFIG for specified config

* `torchx.components`
    * Removed "base" + "torch_dist_role" since users should prefer to use the `dist.ddp` components instead
    * Removed custom components for example apps in favor of using builtins.
    * Added "env", "max_retries" and "mounts" arguments to utils.sh

* `torchx.cli`
    * Better parsing of configs from a string literal
    * Added support to delimit kv-pairs and list values with "," and ";" interchangeably
    * allow the default scheduler to be specified via .torchxconfig
    * better invalid scheduler messaging
    * Log message about how to disable workspaces
    * Job cancellation support via `torchx cancel <job>`

`torchx.workspace`
    * Support for .dockerignore files used as include lists to fixe some behavioral differences between how .dockerignore files are interpreted by torchx and docker

* Testing
    * Component tests now run sequentially
    * Components can be tested with a runner using `components.components_test_base.ComponentTestCase#run_component()` method.

* Additional Changes
    * Updated Pyre configuration to preemptively guard again upcoming semantic changes
    * Formatting changes from black 22.3.0
    * Now using pyfmt with usort 1.0 and the new import merging behavior.
    * Added script to automatically get system diagnostics for reporting purposes


## torchx-0.1.2

Milestone: https://github.com/meta-pytorch/torchx/milestones/3

* PyTorch 1.11 Support
* Python 3.10 Support
* `torchx.workspace`
  * TorchX now supports a concept of workspaces. This enables seamless launching
    of jobs using changes present in your local workspace. For Docker based
    schedulers, we automatically build a new docker container on job launch
    making it easier than ever to run experiments. #333
* `torchx.schedulers`
  * Ray #329
    * Newly added Ray scheduler makes it easy to launch jobs on Ray.
    * https://pytorch.medium.com/large-scale-distributed-training-with-torchx-and-ray-1d09a329aacb
  * AWS Batch #381
    * Newly added AWS Batch scheduler makes it easy to launch jobs in AWS with minimal infrastructure setup.
  * Slurm
    * Slurm jobs will by default launch in the current working directory to match `local_cwd` and workspace behavior. #372
    * Replicas now have their own log files and can be accessed programmatically. #373
    * Support for `comment`, `mail-user` and `constraint` fields. #391
    * WorkspaceMixin support (prototype) - Slurm jobs can now be launched in isolated experiment directories. #416
  * Kubernetes
    * Support for running jobs under service accounts. #408
    * Support for specifying instance types. #433
  * All Docker-based Schedulers (Kubernetes, Batch, Docker)
    * Added bind mount and volume supports #420, #426
    * Bug fix: Better shm support for large dataloader #429
    * Support for `.dockerignore` and custom Dockerfiles #401
  * Local Scheduler
    * Automatically set `CUDA_VISIBLE_DEVICES` #383
    * Improved log ordering #366
* `torchx.components`
  * `dist.ddp`
    * Rendezvous works out of the box on all schedulers #400
    * Logs are now prefixed with local ranks #412
    * Can specify resources via the CLI #395
    * Can specify environment variables via the CLI #399
  * HPO
    * Ax runner now lives in the Ax repo https://github.com/facebook/Ax/commit/8e2e68f21155e918996bda0b7d97b5b9ef4e0cba
* `torchx.cli`
  * `.torchxconfig`
    * You can now specify component argument defaults `.torchxconfig` https://github.com/meta-pytorch/torchx/commit/c37cfd7846d5a0cb527dd19c8c95e881858f8f0a
    * `~/.torchxconfig` can now be used to set user level defaults. #378
    * `--workspace` can be configured #397
  * Color change and bug fixes #419
* `torchx.runner`
  * Now supports workspace interfaces. #360
  * Returned lines now preserve whitespace to provide support for progress bars #425
  * Events are now logged to `torch.monitor` when available. #379
* `torchx.notebook` (prototype)
  * Added new workspace interface for developing models and launching jobs via a Jupyter Notebook. #356
* Docs
  * Improvements to clarify TorchX usage w/ workspaces and general cleanups.
  * #374, #402, #404, #407, #434

## torchx-0.1.1

* Milestone: https://github.com/meta-pytorch/torchx/milestone/2

* `torchx.schedulers`
  * #287, #286 - Implement `local_docker` scheduler using docker client lib

* Docs
  * #336 - Add context/intro to each docs page
  * Minor document corrections

* `torchx`
  * #267 - Make torchx.version.TORCHX_IMAGE follow the same semantics as __version__
  * #299 - Use base docker image `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime`

* `torchx.specs`
  * #301 - Add `metadata` field to `torchx.specs.Role` dataclass
  * #302 - Deprecate RunConfig in favor of raw `Dict[str, ConfigValue]`

* `torchx.cli`
  * #316 - Implement `torchx builtins --print` that prints the source code of the component

* `torchx.runner`
  * #331 - Split run_component into run_component and dryrun_component

## torchx-0.1.0

* `torchx.schedulers`
  * `local_docker` print a nicer error if Docker is not installed #284
* `torchx.cli`
  *  Improved error messages when `-cfg` is not provided #271
* `torchx.components`
  * Update `dist.ddp` to use `c10d` backend as default #263
* `torchx.aws`
  * Removed entirely as it was unused

* Docs
  * Restructure documentation to be more clear
  * Merged Hello World example with the Quickstart guide to reduce confusion
  * Updated Train / Distributed component documentation
  * Renamed configure page to "Advanced Usage" to avoid confusion with experimental .torchxconfig
  * Renamed Localhost page to just Local to better match the class name
  * Misc cleanups / improvements

* Tests
  * Fixed test failure when no secrets are present #274
  * Added macOS variant to our unit tests #209

## torchx-0.1.0rc1

* `torchx.specs`
  * base_image has been deprecated
  * Some predefined AWS specific named_resources have been added
  * Docstrings are no longer required for component definitions to make it
  easier to write them. They will be still rendered as help text if present and
  are encouraged but aren't required.
  * Improved vararg handling logic for components

* `torchx.runner`
  * Username has been removed from the session name
  * Standardized `runopts` naming

* `torchx.cli`
  * Added experimental `.torchxconfig` file which can be used to set default
  scheduler arguments for all runs.
  * Added `--version` flag
  * `builtins` ignores `torchx.components.base` folder

* Docs
  * Improved entry_points and resources docs
  * Better component documentation
  * General improvements and fixes

* Examples
  * Moved examples to be under torchx/ and merged the examples container with
  the primary container to simplify usage.
  * Added a self contained "Hello World" example
  * Switched lightning_classy_vision example to use ResNet model architecture so
  it will actually converage
  * Removed CIFAR example and merged functionality into lightning_classy_vision

* CI
  * Switched to OpenID Connect based auth

## torchx-0.1.0rc0

* `torchx.specs` API release candidate
  (still experimental but no major changes expected for `0.1.0`)
* `torchx.components`
  * made all components use docker images by default for consistency
  * removed binary_component in favor of directly writing app defs
  * `serve.torchserve` - added optional `--port` argument for upload server
  * `utils.copy` - added copy component for easy file transfer between `fsspec` path locations
  * `ddp`
    * `nnodes` no longer needs to be specified and is set from `num_replicas` instead.
    * Bug fixes.
    * End to end integration tests on Slurm and Kubernetes.
  * better unit testing support via `ComponentTestCase`.
* `torchx.schedulers`
  * Split `local` scheduler into `local_docker` and `local_cwd`.
    * For local execution `local_docker` provides the closest experience to remote behavior.
    * `local_cwd` allows reusing the same component definition for local development purposes but resolves entrypoint and deps relative to the current working directory.
  * Improvements/bug fixes to Slurm and Kubernetes schedulers.
* `torchx.pipelines`
  * `kfp` Added the ability to launch distributed apps via the new `resource_from_app` method which creates a Volcano Job from Kubeflow Pipelines.
* `torchx.runner` - general fixes and improvements around wait behavior
* `torchx.cli`
  * Improvements to output formatting to improve clarity.
  * `log` can now log from all roles instead of just one
  * `run` now supports boolean arguments
  * Experimental support for CLI being used from scripts. Exit codes are consistent and only script consumable data is logged on stdout for key commands such as `run`.
  * `--log_level` configuration flag
  * Default scheduler is now `local_docker` and decided by the first scheduler in entrypoints.
  * More robust component finding and better behavior on malformed components.
* `torchx.examples`
   * Distributed CIFAR Training Example
   * HPO
   * Improvements to lightning_classy_vision example -- uses components, datapreproc separated from injection
   * Updated to use same file directory layout as github repo
   * Added documentation on setting up kubernetes cluster for use with TorchX
   * Added distributed KFP pipeline example
* `torchx.runtime`
  * Added experimental `hpo` support with Ax (https://github.com/facebook/Ax)
  * Added experimental `tracking.ResultTracker` for distributed tracking of metrics for use with HPO.
  * Bumped pytorch version to 1.9.0.
  * Deleted deprecated storage/plugins interface.
* Docs
  * Added app/component best practices
  * Added more information on different component archetypes such as training
  * Refactored structure to more accurately represent components, runtime and
    scheduler libraries.
  * README: added information on how to install from source, nightly and different dependencies
  * Added scheduler feature compatibility matrices
  * General cleanups and improvements
* CI
  * component integration test framework
  * codecoverage
  * renamed primary branch to main
  * automated doc push
  * distributed kubernetes integration tests
  * nightly builds at https://pypi.org/project/torchx-nightly/
  * pyre now uses nightly builds
  * added slurm integration tests


## torchx-0.1.0b0

* `torchx.specs` API release candidate
  (still experimental but no major changes expected for `0.1.0`)

* `torchx.pipelines` - Kubeflow Pipeline adapter support
* `torchx.runner` - SLURM and local scheduler support
* `torchx.components` - several utils, ddp, torchserve builtin components
* `torchx.examples`
   * Colab support for examples
   * `apps`:
     * classy vision + lightning trainer
     * torchserve deploy
     * captum model visualization
   * `pipelines`:
     * apps above as a Kubeflow Pipeline
     * basic vs advanced Kubeflow Pipeline examples
* CI
  * unittest, pyre, linter, KFP launch, doc build/test
