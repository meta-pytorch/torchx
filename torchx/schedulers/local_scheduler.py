#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This contains the TorchX local scheduler which can be used to run TorchX
components locally via subprocesses.
"""

import abc
import io
import json
import logging
import os
import pprint
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from types import FrameType
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    TextIO,
    Tuple,
    TypedDict,
)

from torchx.schedulers.api import (
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    split_lines_iterator,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.schedulers.streams import Tee
from torchx.specs import AppDryRunInfo
from torchx.specs.api import AppDef, AppState, is_terminal, macros, NONE, Role, runopts

from torchx.util.types import none_throws

log: logging.Logger = logging.getLogger(__name__)

STDOUT_LOG = "stdout.log"
STDERR_LOG = "stderr.log"
COMBINED_LOG = "combined.log"

NA: str = "<N/A>"

ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


class SignalException(Exception):
    """
    Exception is raised during the runtime when the torchx local scheduler process
    got termination signal.
    """

    def __init__(self, msg: str, sigval: signal.Signals) -> None:
        super().__init__(msg)
        self.sigval = sigval


def _terminate_process_handler(signum: int, frame: FrameType) -> None:
    """Termination handler that raises exceptions on the main process.

    When the process receives death signal(SIGTERM, SIGINT), this termination handler will
    be invoked. It raises the ``SignalException`` exception that should be processed by the
    user code. Python does not terminate process after the termination handler is finished,
    so the exception should not be silently ignored, otherwise the process will never
    be terminated.
    """
    sigval = signal.Signals(signum)
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)


@dataclass
class ReplicaParam:
    """
    Holds ``LocalScheduler._popen()`` parameters for each replica of the role.
    """

    args: List[str]
    env: Dict[str, str]

    # IO stream files
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    combined: Optional[str] = None

    cwd: Optional[str] = None


class ImageProvider(abc.ABC):
    """
    Manages downloading and setting up an image on localhost. This is only needed for
    ``LocalhostScheduler`` since typically real schedulers will do this
    on-behalf of the user.
    """

    def fetch_role(self, role: Role) -> str:
        """
        Identical to ``fetch(image)`` in that it fetches the role's
        image and returns the path to the image root, except that
        it allows the role to be updated by this provider. Useful
        when additional environment variables need to be set on the role
        to comply with the image provider's way of fetching and managing
        images on localhost. By default this method simply delegates
        to ``fetch(role.image)``. Override if necessary.
        """
        return self.fetch(role.image)

    @abc.abstractmethod
    def fetch(self, image: str) -> str:
        """
        Pulls the given image and returns a path to the pulled image on
        the local host or empty string if no op
        """
        raise NotImplementedError()

    def get_replica_param(
        self,
        img_root: str,
        role: Role,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        combined: Optional[str] = None,
    ) -> ReplicaParam:
        """
        Given the role replica's specs returns ``ReplicaParam`` holder
        which hold the arguments to eventually pass to ``subprocess.Popen``
        to actually invoke and run each role's replica. The ``img_root``
        is expected to be the return value of ``self.fetch(role.image)``.
        Since the role's image need only be fetched once (not for each replica)
        it is expected that the caller call the ``fetch`` method once per role
        and call this method for each ``role.num_replicas``.
        """
        return ReplicaParam(
            [self.get_entrypoint(img_root, role)] + role.args,
            role.env,
            stdout,
            stderr,
            combined,
            self.get_cwd(role.image),
        )

    def get_cwd(self, image: str) -> Optional[str]:
        """
        Returns the absolute path of the mounted img directory. Used as a working
        directory for starting child processes.
        """
        return None

    def get_entrypoint(self, img_root: str, role: Role) -> str:
        """
        Returns the location of the entrypoint.
        """
        return os.path.join(img_root, role.entrypoint)


class LocalOpts(TypedDict, total=False):
    log_dir: str
    prepend_cwd: Optional[bool]
    auto_set_cuda_visible_devices: Optional[bool]


class LocalDirectoryImageProvider(ImageProvider):
    """
    Interprets the image name as the path to a directory on
    local host. Does not "fetch" (e.g. download) anything. Used in conjunction
    with ``LocalScheduler`` to run local binaries.

    The image name must be an absolute path and must exist.

    Example:

    #. ``fetch(Image(name="/tmp/foobar"))`` returns ``/tmp/foobar``
    #. ``fetch(Image(name="foobar"))`` raises ``ValueError``
    #. ``fetch(Image(name="/tmp/dir/that/does/not_exist"))`` raises ``ValueError``

    """

    def __init__(self, cfg: LocalOpts) -> None:
        pass

    def fetch(self, image: str) -> str:
        """
        Raises:
            ValueError: if the image name is not an absolute dir and if it
                        does not exist or is not a directory

        """
        if not os.path.isdir(image):
            raise ValueError(
                f"Invalid image name: {image}, does not exist or is not a directory"
            )

        return image

    def get_cwd(self, image: str) -> Optional[str]:
        """
        Returns the absolute working directory. Used as a working
        directory for the child process.
        """
        return image

    def get_entrypoint(self, img_root: str, role: Role) -> str:
        """
        Returns the role entrypoint. When local scheduler is executed with
        image_type=dir, the childprocess working directory will be set to the
        img_root. If `role.entrypoint` is relative path, it would be resolved
        as `img_root/role.entrypoint`, if `role.entrypoint` is absolute path,
        it will be executed as provided.
        """
        return role.entrypoint


class CWDImageProvider(ImageProvider):
    """
    Similar to LocalDirectoryImageProvider however it ignores the image name and
    uses the current working directory as the image path.

    Example:

    #. ``fetch(Image(name="/tmp/foobar"))`` returns `os.getcwd()`
    #. ``fetch(Image(name="foobar:latest"))`` returns `os.getcwd()`

    """

    def __init__(self, cfg: LocalOpts) -> None:
        pass

    def fetch(self, image: str) -> str:
        return os.getcwd()

    def get_cwd(self, image: str) -> Optional[str]:
        return os.getcwd()

    def get_entrypoint(self, img_root: str, role: Role) -> str:
        return role.entrypoint


# aliases to make clear what the mappings are
AppId = str
AppName = str
RoleName = str


class PopenProtocol(Protocol):
    """
    Protocol wrapper around python's ``subprocess.Popen``. Keeps track of
    the a list of interface methods that the process scheduled by the `LocalScheduler`
    must implement.
    """

    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int: ...

    def wait(self, timeout: Optional[float] = None) -> int: ...

    def poll(self) -> Optional[int]: ...

    def kill(self) -> None: ...


@dataclass
class _LocalReplica:
    """
    Contains information about a locally running role replica.
    """

    role_name: RoleName
    replica_id: int
    proc: PopenProtocol

    # IO streams:
    # None means no log_dir (out to console)
    stdout: Optional[BinaryIO]
    stderr: Optional[BinaryIO]
    combined: Optional[Tee]

    error_file: str

    def terminate(self) -> None:
        """
        terminates the underlying process for this replica
        closes stdout and stderr file handles
        safe to call multiple times
        """
        # safe to call terminate on a process that already died
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError as e:
            log.debug(f"Process {self.proc.pid} already got terminated")

        # close stdout and stderr log file handles
        if self.stdout:
            none_throws(self.stdout).close()
        if self.stderr:
            none_throws(self.stderr).close()
        if self.combined:
            none_throws(self.combined).close()

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def failed(self) -> bool:
        if self.is_alive():  # if still running, then has not failed
            return False
        else:
            return self.proc.returncode != 0


class _LocalAppDef:
    """
    Container object used by ``LocalhostScheduler`` to group the pids that
    form an application. Each replica of a role in the application is a
    process and has a pid.
    """

    def __init__(self, id: str, log_dir: str) -> None:
        self.id = id
        # cfg.get("log_dir")/<session_name>/<app_id> or /tmp/torchx/<session_name>/<app_id>
        self.log_dir = log_dir
        # role name -> [replicas, ...]
        self.role_replicas: Dict[RoleName, List[_LocalReplica]] = {}
        self.state: AppState = AppState.PENDING
        # time (in seconds since epoch) when the last set_state method() was called
        self.last_updated: float = -1

    def add_replica(self, role_name: str, replica: _LocalReplica) -> None:
        procs = self.role_replicas.setdefault(role_name, [])
        procs.append(replica)

    def set_state(self, state: AppState) -> None:
        self.last_updated = time.time()
        self.state = state

    def kill(self) -> None:
        """
        terminates all procs associated with this app,
        and closes any resources (e.g. log file handles)
        safe to call multiple times

        The termination consists of two stages:
        1. Send SIGTERM signal to the child processes and wait for them to terminate.
        2. If timeout passed and child processes are still alive, terminate them via SIGKILL.
        """

        # Stage #1: SIGTERM
        for replicas in self.role_replicas.values():
            for r in replicas:
                r.terminate()

        timeout = 10  # seconds
        end = time.monotonic() + timeout
        for replicas in self.role_replicas.values():
            for r in replicas:
                time_to_wait = end - time.monotonic()
                if time_to_wait <= 0:
                    break
                try:
                    r.proc.wait(time_to_wait)
                except subprocess.TimeoutExpired:
                    # Ignore the timeout expired exception, since
                    # the child process will be forcefully terminated via SIGKILL
                    pass

        # Stage #2: SIGKILL
        for replicas in self.role_replicas.values():
            for r in replicas:
                if r.proc.poll() is None:
                    r.proc.kill()

        for replicas in self.role_replicas.values():
            for r in replicas:
                r.proc.wait()
                r.terminate()

    def _get_error_file(self) -> Optional[str]:
        error_file = None
        min_timestamp = sys.maxsize
        for replicas in self.role_replicas.values():
            for replica in replicas:
                if not os.path.exists(replica.error_file):
                    continue
                mtime = os.path.getmtime(replica.error_file)
                if mtime < min_timestamp:
                    min_timestamp = mtime
                    error_file = replica.error_file
        return error_file

    def get_structured_error_msg(self) -> str:
        error_file = self._get_error_file()
        if not error_file:
            return NONE

        with open(error_file, "r") as f:
            return json.dumps(json.load(f))

    def close(self) -> None:
        """
        terminates all procs associated with this app,
        and closes any resources (e.g. log file handles)
        and if log_dir has been specified,
        writes a SUCCESS file indicating that the log files
        have been flushed and closed and ready to read.
        NOT safe to call multiple times!
        """
        self.kill()

        def _fmt_io_filename(std_io: Optional[BinaryIO]) -> str:
            if std_io:
                return std_io.name
            else:
                return "<CONSOLE>"

        # drop a SUCCESS file in the log dir to signal that
        # the log file handles have all been closed properly
        # and that they can reliably be read
        roles_info = {}
        for role_name, replicas in self.role_replicas.items():
            replicas_info = []
            for replica in replicas:
                replica_info = {
                    "replica_id": replica.replica_id,
                    "pid": replica.proc.pid,
                    "exitcode": replica.proc.returncode,
                    "stdout": _fmt_io_filename(replica.stdout),
                    "stderr": _fmt_io_filename(replica.stderr),
                    "error_file": replica.error_file,
                }
                replicas_info.append(replica_info)
            roles_info[role_name] = replicas_info
        app_info = {
            "app_id": self.id,
            "log_dir": self.log_dir,
            "final_state": self.state.name,
            "last_updated": self.last_updated,
            "roles": roles_info,
        }

        info_str = json.dumps(app_info, indent=2)
        with open(os.path.join(self.log_dir, "SUCCESS"), "w") as fp:
            fp.write(info_str)

        log.debug(f"Successfully closed app_id: {self.id}.\n{info_str}")

    def __repr__(self) -> str:
        role_to_pid = {}
        for role_name, replicas in self.role_replicas.items():
            pids = role_to_pid.setdefault(role_name, [])
            for r in replicas:
                pids.append(r.proc.pid)

        return f"{{app_id:{self.id}, state:{self.state}, pid_map:{role_to_pid}}}"


def _join_PATH(*paths: Optional[str]) -> str:
    """
    Joins strings that go in the PATH env var.
    Deals with empty strings and None-types, making sure no leading
    or trailing path-sep (`:`) in the resulting string

    Usage:

    .. code-block:: python

     # PATH=/usr/local/bin:$PATH (prepend)
     join_PATH("/usr/local/bin", os.environ["PATH"])

     # PATH=$PATH:/usr/local/bin (append)
     join_PATH(os.environ["PATH"], "/usr/local/bin")

    """

    return os.pathsep.join(
        [p.strip(os.pathsep) for p in paths if p]
    )  # remove empty and null str + strip leading and trailing ":"s


@dataclass
class PopenRequest:
    """
    Holds parameters to create a subprocess for each replica of each role
    of an application.
    """

    app_id: AppId
    log_dir: str
    # maps role_name -> List[ReplicaSpec]
    # role_params["trainer"][0] -> holds trainer's 0^th replica's (NOT rank!) parameters
    role_params: Dict[RoleName, List[ReplicaParam]]
    # maps role_name -> List[replica_log_dir]
    # role_log_dirs["trainer"][0] -> holds trainer's 0^th replica's log directory path
    role_log_dirs: Dict[RoleName, List[str]]


def _register_termination_signals() -> None:
    """
    Register SIGTERM and SIGINT handlers only for the main thread.
    """
    if threading.current_thread() is threading.main_thread():
        # Register termination handlers for SIGTERM and SIGINT
        # Temporary disable signal handler registration
        signal.signal(signal.SIGTERM, _terminate_process_handler)
        signal.signal(signal.SIGINT, _terminate_process_handler)


class LocalScheduler(Scheduler[LocalOpts, AppDef, AppDryRunInfo[PopenRequest]]):
    """
    Schedules on localhost. Containers are modeled as processes and
    certain properties of the container that are either not relevant
    or that cannot be enforced for localhost
    runs are ignored. Properties that are ignored:

    1. Resource requirements
    2. Resource limit enforcements
    3. Retry policies
    4. Retry counts (no retries supported)
    5. Deployment preferences

    Scheduler support orphan processes cleanup on receiving SIGTERM or SIGINT.
    The scheduler will terminate the spawned processes.

    This is exposed via the scheduler `local_cwd`.

    * `local_cwd` runs the provided app relative to the current working
      directory and ignores the images field for faster iteration and testing
      purposes.

    .. note::
        The orphan cleanup only works if `LocalScheduler` is instantiated from the main thread.

    **Config Options**

    .. runopts::
        class: torchx.schedulers.local_scheduler.create_scheduler

    **Compatibility**

    .. note::
        Due to scheduler differences jobs that run locally may not work when
        using a different scheduler due to network or software dependencies.

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: |
                LocalScheduler supports multiple replicas but all replicas will
                execute on the local host.
            describe: true
            workspaces: |
                Partial support. LocalScheduler runs the app from a local
                directory but does not support programmatic workspaces.
            mounts: false
            elasticity: false
    """

    def __init__(
        self,
        session_name: str,
        image_provider_class: Callable[[LocalOpts], ImageProvider],
        cache_size: int = 100,
        extra_paths: Optional[List[str]] = None,
    ) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("local", session_name)

        # TODO T72035686 replace dict with a proper LRUCache data structure
        self._apps: Dict[AppId, _LocalAppDef] = {}
        self._image_provider_class = image_provider_class

        if cache_size <= 0:
            raise ValueError("cache size must be greater than zero")
        self._cache_size = cache_size
        _register_termination_signals()

        self._extra_paths: List[str] = extra_paths or []

        # sets lazily on submit or dryrun based on log_dir cfg
        self._base_log_dir: Optional[str] = None
        self._created_tmp_log_dir: bool = False

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "log_dir",
            type_=str,
            default=None,
            help="dir to write stdout/stderr log files of replicas",
        )
        opts.add(
            "prepend_cwd",
            type_=bool,
            default=False,
            help="if set, prepends CWD to replica's PATH env var"
            " making any binaries in CWD take precedence over those in PATH",
        )
        opts.add(
            "auto_set_cuda_visible_devices",
            type_=bool,
            default=False,
            help="sets the `CUDA_AVAILABLE_DEVICES` for roles that request GPU resources."
            " Each role replica will be assigned one GPU. Does nothing if the device count is less than replicas.",
        )
        return opts

    def _validate(self, app: AppDef, scheduler: str, cfg: LocalOpts) -> None:
        # Skip validation step for local application
        pass

    def _evict_lru(self) -> bool:
        """
        Evicts one least recently used element from the apps cache. LRU is defined as
        the oldest app in a terminal state (e.g. oldest finished app).

        Returns:
            ``True`` if an entry was evicted, ``False`` if no entries could be evicted
            (e.g. all apps are running)
        """
        lru_time = sys.maxsize
        lru_app_id = None
        for app_id, app in self._apps.items():
            if is_terminal(app.state):
                if app.last_updated <= lru_time:
                    lru_app_id = app_id

        if lru_app_id:
            # evict LRU finished app from the apps cache
            del self._apps[lru_app_id]

            log.debug(f"evicting app: {lru_app_id}, from local scheduler cache")
            return True
        else:
            log.debug(f"no apps evicted, all {len(self._apps)} apps are running")
            return False

    def _get_file_io(self, file: Optional[str]) -> Optional[io.FileIO]:
        """
        Given a file name, opens the file for write and returns the IO.
        If no file name is given, then returns ``None``
        Raises a ``FileExistsError`` if the file is already present.
        """

        if not file:
            return None

        if os.path.isfile(file):
            raise FileExistsError(
                f"log file: {file} already exists,"
                f" specify a different log_dir, app_name, or remove the file and retry"
            )

        os.makedirs(os.path.dirname(file), exist_ok=True)
        return io.open(file, mode="wb", buffering=0)

    def _popen(
        self,
        role_name: RoleName,
        replica_id: int,
        replica_params: ReplicaParam,
    ) -> _LocalReplica:
        """
        Same as ``subprocess.Popen(**popen_kwargs)`` but is able to take ``stdout`` and ``stderr``
        as file name ``str`` rather than a file-like obj.
        """

        stdout_, stderr_, combined_ = self._get_replica_output_handles(replica_params)

        args_pfmt = pprint.pformat(asdict(replica_params), indent=2, width=80)
        log.debug(f"Running {role_name} (replica {replica_id}):\n {args_pfmt}")
        env = self._get_replica_env(replica_params)

        proc = self.run_local_job(
            args=replica_params.args,
            env=env,
            stdout=stdout_,
            stderr=stderr_,
            cwd=replica_params.cwd,
        )
        return _LocalReplica(
            role_name,
            replica_id,
            proc,
            stdout=stdout_,
            stderr=stderr_,
            combined=combined_,
            error_file=env.get("TORCHELASTIC_ERROR_FILE", "<N/A>"),
        )

    def run_local_job(
        self,
        args: List[str],
        env: Dict[str, str],
        stdout: Optional[io.FileIO],
        stderr: Optional[io.FileIO],
        cwd: Optional[str] = None,
    ) -> "subprocess.Popen[bytes]":
        return subprocess.Popen(
            args=args,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            cwd=cwd,
        )

    def _get_replica_output_handles(
        self,
        replica_params: ReplicaParam,
    ) -> Tuple[Optional[io.FileIO], Optional[io.FileIO], Optional[Tee]]:
        """
        Returns the stdout, stderr, and combined outputs of the replica.
        If the combined output file is not specified, then the combined output is ``None``.
        """

        stdout_ = self._get_file_io(replica_params.stdout)
        stderr_ = self._get_file_io(replica_params.stderr)
        combined_: Optional[Tee] = None
        combined_file = self._get_file_io(replica_params.combined)
        if combined_file:
            combined_ = Tee(
                combined_file,
                none_throws(replica_params.stdout),
                none_throws(replica_params.stderr),
            )
        return stdout_, stderr_, combined_

    def _get_replica_env(
        self,
        replica_params: ReplicaParam,
    ) -> Dict[str, str]:
        """
        Returns environment variables for the ``_LocalReplica``
        """

        # inherit parent's env vars since 99.9% of the time we want this behavior
        # just make sure we override the parent's env vars with the user_defined ones
        env = os.environ.copy()
        env.update(replica_params.env)
        # PATH is a special one, instead of overriding, append
        env["PATH"] = _join_PATH(replica_params.env.get("PATH"), os.getenv("PATH"))

        # default to unbuffered python for faster responsiveness locally
        env.setdefault("PYTHONUNBUFFERED", "x")

        return env

    def _get_app_log_dir(self, app_id: str, cfg: LocalOpts) -> str:
        """
        Returns the log dir. We redirect stdout/err
        to a log file ONLY if the log_dir is user-provided in the cfg

        1. if cfg.get("log_dir") -> (user-specified log dir, True)
        2. if not cfg.get("log_dir") -> (autogen tmp log dir, False)
        """

        self._base_log_dir = cfg.get("log_dir")
        if not self._base_log_dir:
            self._base_log_dir = tempfile.mkdtemp(prefix="torchx_")
            self._created_tmp_log_dir = True
            log.info(
                "Log directory not set in scheduler cfg."
                " Creating a temporary log dir that will be deleted on exit."
                " To preserve log directory set the `log_dir` cfg option"
            )

        log.info(f"Log directory is: {self._base_log_dir}")
        return os.path.join(str(self._base_log_dir), self.session_name, app_id)

    def schedule(self, dryrun_info: AppDryRunInfo[PopenRequest]) -> str:
        if len(self._apps) == self._cache_size:
            if not self._evict_lru():
                raise IndexError(
                    f"App cache size ({self._cache_size}) exceeded. Increase the cache size"
                )

        request: PopenRequest = dryrun_info.request
        app_id = request.app_id
        app_log_dir = request.log_dir
        assert (
            app_id not in self._apps
        ), "no app_id collisions expected since uuid4 suffix is used"

        os.makedirs(app_log_dir)
        local_app = _LocalAppDef(app_id, app_log_dir)

        for role_name in request.role_params.keys():
            role_params = request.role_params[role_name]
            role_log_dirs = request.role_log_dirs[role_name]
            for replica_id in range(len(role_params)):
                replica_params = role_params[replica_id]
                replica_log_dir = role_log_dirs[replica_id]

                os.makedirs(replica_log_dir)
                replica = self._popen(
                    role_name,
                    replica_id,
                    replica_params,
                )
                local_app.add_replica(role_name, replica)
        self._apps[app_id] = local_app
        return app_id

    def _submit_dryrun(
        self, app: AppDef, cfg: LocalOpts
    ) -> AppDryRunInfo[PopenRequest]:
        request = self._to_popen_request(app, cfg)
        return AppDryRunInfo(
            request, lambda p: pprint.pformat(asdict(p), indent=2, width=80)
        )

    def _cuda_device_count(self) -> int:
        # this method deliberately does not use ``torch.cuda.device_count()``
        # to avoid taking a dependency on pytorch
        # this makes it possible to avoid a BUCK dependency (internally at Meta)
        # on //caffe2:torch which slows down builds of //torchx:* rules
        gpu_cmd = "nvidia-smi -L"
        try:
            log.debug(f"Running {gpu_cmd}")
            result = subprocess.run(
                gpu_cmd.split(), capture_output=True, text=True, check=True
            )
            log.debug(f"Cmd {gpu_cmd} returned: {result}")
            gpus_info = [gpu_info for gpu_info in result.stdout.split("\n") if gpu_info]
            return len(gpus_info)
        except subprocess.CalledProcessError as e:
            log.exception(f"Got exception while listing GPUs: {e.stderr}")
            return 0

    def auto_set_CUDA_VISIBLE_DEVICES(
        self,
        role_params: Dict[str, List[ReplicaParam]],
        app: AppDef,
        cfg: LocalOpts,
    ) -> None:
        """
        If the run option ``auto_set_cuda_visible_devices = True``, then
        sets the ``CUDA_VISIBLE_DEVICES`` env var to each replica's (node) env var
        according to the number of gpus specified in each role's resource specifications,
        overwriting any existing ``CUDA_VISIBLE_DEVICES`` in the role's ``env`` field.
        To manually set ``CUDA_VISIBLE_DEVICES``, run with ``auto_set_cuda_visible_devices = False``
        in the scheduler runcfg.

        .. note::
            If the host's device count is less than the total number of requested GPUs,
            then ``CUDA_VISIBLE_DEVICES`` is NOT set (even if ``auto_set_cuda_visible_devices=True``).

        .. note::
            This method either sets ``CUDA_VISIBLE_DEVICES`` on all gpu roles or doesn't


        Examples (all examples assume running on a host with 8 GPUs):

        #. ``Role(num_replicas=2, resource=Resource(gpus=2))``
              #. replica_0's ``CUDA_VISIBLE_DEVICES=0,1``
              #. replica_1's ``CUDA_VISIBLE_DEVICES=2,3``

        #. ``Role(num_replicas=3, resource=Resource(gpus=4))``
              #. Error - `` 3 * 4 = 12 >= 8``

        #. ``[Role(num_replicas=1, resource=Resource(gpus=2)), Role(num_replicas=3, resource=Resource(gpus=1))]``
              #. role_0, replica_0's ``CUDA_VISIBLE_DEVICES=0,1``
              #. role_1, replica_0's ``CUDA_VISIBLE_DEVICES=2``
              #. role_1, replica_1's ``CUDA_VISIBLE_DEVICES=3``
              #. role_1, replica_2's ``CUDA_VISIBLE_DEVICES=4``

        """

        total_requested_gpus = 0  # total number of gpus for the app

        for role in app.roles:
            gpus = role.num_replicas * role.resource.gpu
            total_requested_gpus += gpus

        if not cfg.get("auto_set_cuda_visible_devices") or total_requested_gpus <= 0:
            if total_requested_gpus > 0:
                log.warning(
                    """\n
======================================================================
Running multiple role replicas that require GPUs without
setting `CUDA_VISIBLE_DEVICES` may result in multiple
processes using the same GPU device with undesired consequences
such as CUDA OutOfMemory errors.

To have TorchX set `CUDA_VISIBLE_DEVICES` to divide the
available GPUs on this host equally among the role replicas
set the `auto_set_cuda_visible_devices = True` scheduler runopt
======================================================================
                            """
                )
            return

        device_count = self._cuda_device_count()
        if total_requested_gpus > device_count:
            log.warning(
                f"""\n
======================================================================
Cannot auto-set `CUDA_VISIBLE_DEVICES`
Available GPUs: {device_count} is less than the
number of requested GPUs: {total_requested_gpus}."

Reduce requested GPU resources or use a host with more GPUs
======================================================================
                """
            )
            return

        start_idx = 0
        for role in app.roles:
            # skip roles that have not requested gpus
            if role.resource.gpu <= 0:
                continue

            role_replicas = role_params[role.name]
            for replica_id, replica in enumerate(role_replicas):
                end_idx = start_idx + role.resource.gpu
                replica.env[ENV_CUDA_VISIBLE_DEVICES] = ",".join(
                    list(str(idx) for idx in range(start_idx, end_idx))
                )
                start_idx = end_idx

    def _to_popen_request(
        self,
        app: AppDef,
        cfg: LocalOpts,
    ) -> PopenRequest:
        """
        Converts the application and cfg into a ``PopenRequest``.
        """

        app_id = make_unique(app.name)
        image_provider = self._image_provider_class(cfg)
        app_log_dir = self._get_app_log_dir(app_id, cfg)

        role_params: Dict[str, List[ReplicaParam]] = {}
        role_log_dirs: Dict[str, List[str]] = {}

        for role in app.roles:
            replica_params = role_params.setdefault(role.name, [])
            replica_log_dirs = role_log_dirs.setdefault(role.name, [])

            img_root = image_provider.fetch_role(role)

            # prepend extra_paths to PATH
            role.env["PATH"] = _join_PATH(*self._extra_paths, role.env.get("PATH"))
            cwd = image_provider.get_cwd(role.image)

            if cwd:
                # if prepend_cwd is set, then prepend cwd to PATH
                # making binaries in cwd take precedence to those in PATH
                # otherwise append cwd to PATH so that the binaries in PATH
                # precede over those in cwd
                prepend_cwd = cfg.get("prepend_cwd")

                if prepend_cwd:
                    role.env["PATH"] = _join_PATH(cwd, role.env.get("PATH"))
                else:
                    role.env["PATH"] = _join_PATH(role.env.get("PATH"), cwd)

            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root=img_root,
                    app_id=app_id,
                    replica_id=str(replica_id),
                    rank0_env="TORCHX_RANK0_HOST",
                )
                replica_role = values.apply(role)
                replica_role.env["TORCHX_RANK0_HOST"] = "localhost"

                replica_log_dir = os.path.join(app_log_dir, role.name, str(replica_id))
                if "TORCHELASTIC_ERROR_FILE" not in replica_role.env:
                    # this is the top level (agent if using elastic role) error file
                    # a.k.a scheduler reply file
                    replica_role.env["TORCHELASTIC_ERROR_FILE"] = os.path.join(
                        replica_log_dir, "error.json"
                    )
                if "PET_LOG_DIR" not in replica_role.env:
                    # equivalent of passing --log_dir to torch.distributed.run
                    # keep all the downstream logs in the same directory as the app
                    # this means that these logs will be cleaned up if `log_dir` option
                    # is not set in this scheduler's cfg and preserved otherwise
                    replica_role.env["PET_LOG_DIR"] = os.path.join(
                        app_log_dir, "torchelastic", role.name
                    )

                stdout = os.path.join(replica_log_dir, STDOUT_LOG)
                stderr = os.path.join(replica_log_dir, STDERR_LOG)
                combined = os.path.join(replica_log_dir, COMBINED_LOG)

                replica_params.append(
                    image_provider.get_replica_param(
                        img_root, replica_role, stdout, stderr, combined
                    )
                )
                replica_log_dirs.append(replica_log_dir)
        self.auto_set_CUDA_VISIBLE_DEVICES(role_params, app, cfg)
        return PopenRequest(app_id, app_log_dir, role_params, role_log_dirs)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        if app_id not in self._apps:
            return None

        local_app = self._apps[app_id]
        structured_error_msg = local_app.get_structured_error_msg()

        # check if the app is known to have finished
        if is_terminal(local_app.state):
            state = local_app.state
        else:
            running = False
            failed = False
            for replicas in local_app.role_replicas.values():
                for r in replicas:
                    running |= r.is_alive()
                    failed |= r.failed()

            if running:
                state = AppState.RUNNING
            elif failed:
                state = AppState.FAILED
            else:
                state = AppState.SUCCEEDED
            local_app.set_state(state)

        if is_terminal(local_app.state):
            local_app.close()

        resp = DescribeAppResponse()
        resp.app_id = app_id
        resp.structured_error_msg = structured_error_msg
        resp.state = state
        resp.num_restarts = 0
        resp.ui_url = f"file://{local_app.log_dir}"
        return resp

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        if since or until:
            warnings.warn(
                "Since and/or until times specified for LocalScheduler.log_iter."
                " These will be ignored and all log lines will be returned"
            )

        app = self._apps[app_id]
        STREAM_FILES = {
            None: COMBINED_LOG,
            Stream.COMBINED: COMBINED_LOG,
            Stream.STDOUT: STDOUT_LOG,
            Stream.STDERR: STDERR_LOG,
        }
        log_file = os.path.join(app.log_dir, role_name, str(k), STREAM_FILES[streams])

        if not os.path.isfile(log_file):
            raise RuntimeError(
                f"app: {app_id} was not configured to log into a file."
                f" Did you run it with log_dir set in Dict[str, CfgVal]?"
            )
        iterator = LogIterator(app_id, log_file, self)
        # sometimes there's multiple lines per logged line
        iterator = split_lines_iterator(iterator)
        if regex:
            iterator = filter_regex(regex, iterator)
        return iterator

    def list(self) -> List[ListAppResponse]:
        raise Exception(
            "App handles cannot be listed for local scheduler as they are not persisted by torchx"
        )

    def _cancel_existing(self, app_id: str) -> None:
        # can assume app_id exists
        local_app = self._apps[app_id]
        local_app.close()
        local_app.state = AppState.CANCELLED

    def close(self) -> None:
        # terminate all apps
        for app_id, app in self._apps.items():
            log.debug(f"Terminating app: {app_id}")
            app.kill()
        # delete logdir if torchx created a log dir
        if self._base_log_dir and self._created_tmp_log_dir:
            shutil.rmtree(self._base_log_dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as e:
            # When the `__del__` method is invoked, we cannot rely on presence of object attributes,
            # More info: https://stackoverflow.com/questions/18058730/python-attributeerror-on-del
            log.warning(
                f"Exception {e} occurred while trying to clean `LocalScheduler` via `__del__` method"
            )


class LogIterator:
    def __init__(
        self,
        app_id: str,
        log_file: str,
        # pyre-fixme: Scheduler opts
        scheduler: Scheduler,
        should_tail: bool = True,
    ) -> None:
        self._app_id: str = app_id
        self._log_file: str = log_file
        self._log_fp: Optional[TextIO] = None
        # pyre-fixme: Scheduler opts
        self._scheduler: Scheduler = scheduler
        self._app_finished: bool = not should_tail

    def _check_finished(self) -> None:
        # either the app (already finished) was evicted from the LRU cache
        # -- or -- the app reached a terminal state (and still in the cache)
        desc = self._scheduler.describe(self._app_id)
        if not desc or is_terminal(desc.state):
            self._app_finished = True
        else:
            self._app_finished = False

    def __iter__(self) -> "LogIterator":
        # wait for the log file to appear or app to finish (whichever happens first)
        while True:
            self._check_finished()  # check to see if app has finished running

            if os.path.isfile(self._log_file):
                time.sleep(0.1)  # fix timing issue
                self._log_fp = open(
                    self._log_file,
                    mode="rt",
                    newline="\n",
                    errors="replace",  # replace bad utf-8 with \uFFFD
                )  # noqa: P201
                break

            if self._app_finished:
                # app finished without ever writing a log file
                raise RuntimeError(
                    f"app: {self._app_id} finished without writing: {self._log_file}"
                )

            time.sleep(0.1)
        return self

    def __next__(self) -> str:
        log_fp = self._log_fp
        assert log_fp is not None
        BUFSIZE = 64000
        while True:
            line = log_fp.read(BUFSIZE)
            if not line:
                # we have reached EOF and app finished
                if self._app_finished:
                    log_fp.close()
                    raise StopIteration()

                # if app is still running we need to wait for more possible log lines
                # sleep for 1 sec to avoid thrashing the follow
                time.sleep(0.1)
                self._check_finished()
            else:
                return line


def create_scheduler(
    session_name: str,
    cache_size: int = 100,
    extra_paths: Optional[List[str]] = None,
    image_provider_class: Callable[[LocalOpts], ImageProvider] = CWDImageProvider,
    **kwargs: Any,
) -> LocalScheduler:
    return LocalScheduler(
        session_name=session_name,
        image_provider_class=image_provider_class,
        cache_size=cache_size,
        extra_paths=extra_paths,
    )
