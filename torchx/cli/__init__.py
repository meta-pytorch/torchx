# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
The ``torchx`` CLI is a commandline tool around :py:class:`torchx.runner.Runner`.
It allows users to launch :py:class:`torchx.specs.AppDef` directly onto
one of the supported schedulers without authoring a pipeline (aka workflow).
This is convenient for quickly iterating on the application logic without
incurring both the technical and cognitive overhead of learning, writing, and
dealing with pipelines.


.. note:: When in doubt use ``torchx --help``.

Listing the builtin components
----------------------------------
Most of the components under the ``torchx.components`` module are what the CLI
considers "built-in" apps. Before you write your own component you should
browse through the builtins to see if there is one that fits your needs already.
If so, no need to even author an app spec!

.. code-block:: shell-session

 $ torchx builtins
 Found <n> builtin configs:
  1. metrics.tensorboard
  2. serve.torchserve
  3. utils.binary
  ... <omitted for brevity>

Listing the supported schedulers and arguments
-------------------------------------------------
To get a list of supported schedulers that you can launch your job into run:

.. code-block:: shell-session

 $ torchx runopts
 local_docker:
 { 'log_dir': { 'default': 'None',
                'help': 'dir to write stdout/stderr log files of replicas',
                'type': 'str'}}
 local_cwd:
 ...
 slurm:
 ...
 kubernetes:
 ...

Running a component as a job
---------------------------------
The ``run``  subcommand takes either one of:

1. name of the builtin

   .. code-block:: shell-session

    $ torchx run --scheduler <sched_name> utils.echo

2. full python module path of the component function

   .. code-block:: shell-session

     $ torchx run --scheduler <sched_name> torchx.components.utils.echo

3. file path of the ``*.py`` file the defines the component
   along with the component function name in that file.

   .. code-block:: shell-session

     $ cat ~/my_trainer_spec.py
     import torchx.specs as specs

     def my_trainer(foo: int, bar: str) -> specs.AppDef:
       <...spec file details omitted for brevity...>

     $ torchx run --scheduler <sched_name> ~/my_trainer_spec.py:my_trainer

Now that you have understood how to chose which app to launch, now it is time
to see what parameters need to be passed. There are three sets of
parameters:

1. arguments to the ``run`` subcommand, see a list of them by running:

   .. code-block:: shell-session

    $ torchx run --help

2. arguments to the scheduler (``--scheduler_args``, also known as ``run_options`` or ``run_configs``),
   each scheduler takes different args, to find out the args for a specific scheduler run (command for
   ``local_cwd`` scheduler shown below:

   .. code-block:: shell-session

    $ torchx runopts local_cwd
    { 'log_dir': { 'default': 'None',
               'help': 'dir to write stdout/stderr log files of replicas',
               'type': 'str'}}

    # pass run options as comma-delimited k=v pairs
    $ torchx run --scheduler local_cwd --scheduler_args log_dir=/tmp ...

3. arguments to the component (the app args are included here), this also depends on the
   component and can be seen with the ``--help`` string on the component

   .. code-block:: shell-session

    $ torchx run --scheduler local_cwd utils.echo --help
    usage: torchx run echo.torchx [-h] [--msg MSG]

    Echos a message

    optional arguments:
    -h, --help  show this help message and exit
    --msg MSG   Message to echo

Putting everything together, running ``echo`` with the ``local_cwd`` scheduler:

.. code-block:: shell-session

 $ torchx run --scheduler local_cwd --scheduler_args log_dir=/tmp utils.echo --msg "hello $USER"
 === RUN RESULT ===
 torchx 2022-06-15 16:08:57 INFO     Log files located in: /tmp/torchx/echo-crls3hcpwjmhc/echo/0
 local_cwd://torchx/echo-crls3hcpwjmhc

By default the ``run`` subcommand does not block for the job to finish, instead it simply
schedules the job on the specified scheduler and prints an ``app handle``
which is a URL of the form: ``$scheduler_name://torchx/$job_id``.
Keep note of this handle since this is what you'll need to provide to other
subcommands to identify your job.

.. note:: If the ``--scheduler`` option is not provided, then it defaults to
          the scheduler backend ``default`` which points to ``local_cwd``. To change
          the default scheduler, see: :ref:`advanced:Registering Custom Schedulers`.


Inspecting what will run (dryrun)
----------------------------------------------
When you are authoring or debugging a component, you may want to find out and
inspect both the scheduler request object that the runner submits as well as
the ``AppDef`` object that is created by the component function. To do this,
use the ``--dryrun`` option to the ``run`` subcommand:

.. code-block:: shell-session

 $ torchx run --dryrun utils.echo --msg hello_world
 === APPLICATION ===
 { 'metadata': {},
   'name': 'echo',
   'roles': [ { 'args': ['hello_world'],
                'entrypoint': '/bin/echo',
                'env': {},
                'image': '/tmp',
                'max_retries': 0,
                'name': 'echo',
                'num_replicas': 1,
                'port_map': {},
                'resource': { 'capabilities': {},
                              'cpu': -1,
                              'gpu': -1,
                              'memMB': -1},
                'retry_policy': <RetryPolicy.APPLICATION: 'APPLICATION'>}]}
 === SCHEDULER REQUEST ===
 PopenRequest(
     app_id='echo_c944ffb2',
     log_dir='/tmp/torchx_asmtmyqj/torchx_kiuk/echo_c944ffb2',
     role_params={
         'echo': [
             ReplicaParam(
                 args=['/bin/echo', 'hello_world'],
                 env={'TORCHELASTIC_ERROR_FILE': '/tmp/torchx_asmtmyqj/torchx_kiuk/echo_c944ffb2/echo/0/error.json'},
                 stdout=None,
                 stderr=None)
             ]
         },
     role_log_dirs={'echo': ['/tmp/torchx_asmtmyqj/torchx_kiuk/echo_c944ffb2/echo/0']})


.. note:: The scheduler request print out will look different based on the scheduler
          type. The example above is a faux request since the scheduler is a local scheduler
          which simply uses ``subprocess.Popen`` to simulate replicas as a POSIX process.
          Nevertheless the scheduler request contains valuable insight into what the runner
          translates the ``AppDef`` to for a particular scheduler backend.

Describing and querying the status of a job
---------------------------------------------
The ``describe`` subcommand is essentially the inverse of the ``run`` command.
That is, it prints the app spec given an ``app_handle``.

.. code-block:: shell-session

 $ torchx describe <app handle>


.. important:: The ``describe`` command attempts to recreate an app spec
               by querying the scheduler for the job description. So what you
               see printed is not always 100% the exact same app spec that was
               given to the ``run``. The extent to which the runner can
               recreate the app spec depends on numerous factors such as
               how descriptive the scheduler's ``describe_job`` API is as well
               as whether there were fields in the app spec that were ignored
               when submitting the job to the scheduler because the scheduler
               does not have support for such a parameter/functionality.
               NEVER rely on the ``describe`` API as a storage function for
               your app spec. It is simply there to help you spot check things.

To get the status of a running job:

.. code-block:: shell-session

 $ torchx status <app_handle> # prints status for all replicas and roles
 $ torchx status --role trainer <app_handle> # filters it down to the trainer role

Filtering by ``--role`` is useful for large jobs that have multiple roles.

Viewing Logs
---------------------

.. note:: This functionality depends on how long your scheduler setup retains logs
          TorchX DOES NOT archive logs on your behalf, but rather relies on the scheduler's
          ``get_log`` API to obtain the logs. Refer to your scheduler's user manual
          to setup log retention properly.

The ``log`` subcommand is a simple wrapper around the scheduler's ``get_log`` API
to let you pull/print the logs for all replicas and roles from one place.
It also lets you pull replica or role specific logs. Below are a few log access
patterns that are useful and self explanatory

.. code-block:: shell-session

 $ torchx log <app_handle>
 Prints all logs from all replicas and roles (each log line is prefixed with role name and replica id)

 $ torchx log --tail <app_handle>
 If the job is still running tail the logs

 $ torchx log --regex ".*Exception.*" <app_handle>
 regex filter to exceptions

 $ torchx log <app_handle>/<role>
 $ torchx log <app_handle>/trainer
 pulls all logs for the role trainer

 $ torchx log <app_handle>/<role_name>/<replica_id>
 $ torchx log <app_handle>/trainer/0,1
 only pulls trainer 0 and trainer 1 (node not rank) logs


.. note:: Some schedulers do not support server-side regex filters. In this case
          the regex filter is applied on the client-side, meaning the full logs
          will have to be passed through the client. This may be very taxing to
          the local host. Please use your best judgment when using the logs API.

Listing Jobs
---------------------

The ``list`` subcommand lets you list app handles and statuses of apps launched on a scheduler.
You can then use the app handles to further describe the app, view the logs, etc.

.. code-block:: shell-session

 $ torchx list -s <scheduler_name>

 $ torchx list -s kubernetes
 APP HANDLE                                          APP STATUS
 --------------------------------------------------  ------------
 kubernetes://torchx/default:trainer-a5qvfhe1hyq2w   SUCCEEDED
 kubernetes://torchx/default:trainer-d796ei2tdtest   SUCCEEDED
 kubernetes://torchx/default:trainer-em0iao2m90000   FAILED
 kubernetes://torchx/default:trainer-ew33oxmdg0123   SUCCEEDED

"""
