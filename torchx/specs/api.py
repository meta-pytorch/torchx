# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import copy
import inspect
import json
import logging as logger
import os
import pathlib
import re
import shutil
import typing
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from json import JSONDecodeError
from string import Template
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Pattern,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from torchx.util.types import to_dict

_APP_STATUS_FORMAT_TEMPLATE = """AppStatus:
    State: ${state}
    Num Restarts: ${num_restarts}
    Roles: ${roles}
    Msg: ${msg}
    Structured Error Msg: ${structured_error_msg}
    UI URL: ${url}
    """

# RPC Error message. Example:
# RuntimeError('On WorkerInfo(id=1, name=trainer:0:0):
# <message with the Traceback>
# ')
_RPC_ERROR_MESSAGE_RE: Pattern[str] = re.compile(
    (r"(?P<exception_type>\w*)\('On WorkerInfo\(.+\):\n" r"(.*\n)*" r"'\)")
)

# Sometimes another exception is nested as a message of the outer exception
# rather than proper exception chaining. Example:
#  c10::Error: CUDA error: an illegal memory access was encountered
# Exception
#   raised from create_event_internal at caffe2/c10/cuda/CUDACachingAllocator.cpp:733
#     (most recent call first):
_EMBEDDED_ERROR_MESSAGE_RE: Pattern[str] = re.compile(r"(?P<msg>.+)\nException.*")

YELLOW_BOLD = "\033[1;33m"
RESET = "\033[0m"


def TORCHX_HOME(*subdir_paths: str) -> pathlib.Path:
    """
    Path to the "dot-directory" for torchx.
    Defaults to `~/.torchx` and is overridable via the `TORCHX_HOME` environment variable.

    Usage:

    .. doc-test::

        from pathlib import Path
        from torchx.specs import TORCHX_HOME

        assert TORCHX_HOME() == Path.home() / ".torchx"
        assert TORCHX_HOME("conda-pack-out") ==  Path.home() / ".torchx" / "conda-pack-out"
    ```
    """

    default_dir = str(pathlib.Path.home() / ".torchx")
    torchx_home = pathlib.Path(os.getenv("TORCHX_HOME", default_dir))

    torchx_home = torchx_home / os.path.sep.join(subdir_paths)
    torchx_home.mkdir(parents=True, exist_ok=True)

    return torchx_home


# ========================================
# ==== Distributed AppDef API =======
# ========================================
@dataclass
class Resource:
    """
    Represents resource requirements for a ``Role``.

    Args:
        cpu: number of logical cpu cores. The definition of a CPU core depends
            on the scheduler. See your scheduler documentation for how a logical
            CPU core maps to physical cores and threads.
        gpu: number of gpus
        memMB: MB of ram
        capabilities: additional hardware specs (interpreted by scheduler)
        devices: a list of named devices with their quantities
        tags: metadata tags for the resource (not interpreted by schedulers)
          used to add non-functional information about resources (e.g. whether it is an alias of another resource)

    Note: you should prefer to use named_resources instead of specifying the raw
    resource requirement directly.
    """

    cpu: int
    gpu: int
    memMB: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    devices: Dict[str, int] = field(default_factory=dict)
    tags: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def copy(original: "Resource", **capabilities: Any) -> "Resource":
        """
        Copies a resource and applies new capabilities. If the same capabilities
        are present in the original resource and as parameter, the one from parameter
        will be used.
        """

        res_capabilities = dict(original.capabilities)
        res_capabilities.update(capabilities)
        return Resource(
            cpu=original.cpu,
            gpu=original.gpu,
            memMB=original.memMB,
            capabilities=res_capabilities,
            devices=original.devices,
        )


# sentinel value used for cases when resource does not matter (e.g. ignored)
NULL_RESOURCE: Resource = Resource(cpu=-1, gpu=-1, memMB=-1)


# no-arg static factory method to use with default_factory in @dataclass
# needed to support python 3.11 since mutable defaults for dataclasses are not allowed in 3.11
def _null_resource() -> Resource:
    return NULL_RESOURCE


# used as "*" scheduler backend
ALL: str = "all"

# sentinel value used to represent missing string attributes, such as image or entrypoint
MISSING: str = "<MISSING>"

# sentinel value used to represent "unset" optional string attributes
NONE: str = "<NONE>"


class macros:
    """
    Defines macros that can be used in the elements of ``Role.args``
    values of ``Role.env``. The macros will be substituted at runtime
    to their actual values.

    .. warning:: Macros used fields of :py:class:`Role` other than the ones
                 mentioned above, are NOT substituted.

    Available macros:

    1. ``img_root`` - root directory of the pulled container.image
    2. ``app_id`` - application id as assigned by the scheduler
    3. ``replica_id`` - unique id for each instance of a replica of a Role,
                        for instance a role with 3 replicas could have the 0, 1, 2
                        as replica ids. Note that when the container fails and is
                        replaced, the new container will have the same ``replica_id``
                        as the one it is replacing. For instance if node 1 failed and
                        was replaced by the scheduler the replacing node will also
                        have ``replica_id=1``.

    Example:

    ::

     # runs: hello_world.py --app_id ${app_id}
     trainer = Role(
                name="trainer",
                entrypoint="hello_world.py",
                args=["--app_id", macros.app_id],
                env={"IMAGE_ROOT_DIR": macros.img_root})
     app = AppDef("train_app", roles=[trainer])
     app_handle = session.run(app, scheduler="local_docker", cfg={})

    """

    img_root = "${img_root}"
    base_img_root = "${base_img_root}"
    app_id = "${app_id}"
    replica_id = "${replica_id}"

    # rank0_env will be filled with the name of the environment variable that
    # provides the master host address. To get the actual hostname the
    # environment variable must be resolved by the app via either shell
    # expansion (wrap sh/bash) or via the application.
    # This may not be available on all schedulers.
    rank0_env = "${rank0_env}"

    @dataclass
    class Values:
        img_root: str
        app_id: str
        replica_id: str
        rank0_env: str
        base_img_root: str = "DEPRECATED"

        def apply(self, role: "Role") -> "Role":
            """
            apply applies the values to a copy the specified role and returns it.
            """

            # Overrides might contain future values which can't be serialized so taken out for the copy
            overrides = role.overrides
            if len(overrides) > 0:
                logger.warning(
                    "Role overrides are not supported for macros. Overrides will not be copied"
                )
                role.overrides = {}
            role = copy.deepcopy(role)
            role.overrides = overrides

            role.args = [self.substitute(arg) for arg in role.args]
            role.env = {key: self.substitute(arg) for key, arg in role.env.items()}
            role.metadata = self._apply_nested(role.metadata)

            return role

        def _apply_nested(self, d: typing.Dict[str, Any]) -> typing.Dict[str, Any]:
            stack = [d]
            while stack:
                current_dict = stack.pop()
                for k, v in current_dict.items():
                    if isinstance(v, dict):
                        stack.append(v)
                    elif isinstance(v, str):
                        current_dict[k] = self.substitute(v)
                    elif isinstance(v, list):
                        for i in range(len(v)):
                            if isinstance(v[i], str):
                                v[i] = self.substitute(v[i])
            return d

        # Overrides the asdict method to generate a dictionary of macro values to be substituted.
        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

        def substitute(self, arg: str) -> str:
            """
            substitute applies the values to the template arg.
            """
            return Template(arg).safe_substitute(**self.to_dict())


class RetryPolicy(str, Enum):
    """
    Defines the retry policy for the ``Roles`` in the ``AppDef``.
    The policy defines the behavior when the role replica encounters a failure:

    1. unsuccessful (non zero) exit code
    2. hardware/host crashes
    3. preemption
    4. eviction

    .. note:: Not all retry policies are supported by all schedulers.
              However all schedulers must support ``RetryPolicy.APPLICATION``.
              Please refer to the scheduler's documentation for more information
              on the retry policies they support and behavior caveats (if any).

    1. REPLICA: Replaces the replica instance. Surviving replicas are untouched.
                Use with ``dist.ddp`` component to have torchelastic coordinate
                restarts and membership changes. Otherwise, it is up to the
                application to deal with failed replica departures and
                replacement replica admittance.
    2. APPLICATION: Restarts the entire application.
    3. ROLE: Restarts the role when any error occurs in that role. This does not
             restart the whole job.
    """

    REPLICA = "REPLICA"
    APPLICATION = "APPLICATION"
    ROLE = "ROLE"


class MountType(str, Enum):
    BIND = "bind"
    VOLUME = "volume"
    DEVICE = "device"


@dataclass
class BindMount:
    """
    Defines a bind mount to `mount --bind` a host path into the worker
    environment. See scheduler documentation on how bind mounts operate for each
    scheduler.

    Args:
        src_path: the path on the host
        dst_path: the path in the worker environment/container
        read_only: whether the mount should be read only
    """

    src_path: str
    dst_path: str
    read_only: bool = False


@dataclass
class VolumeMount:
    """
    Defines a persistent volume mount to mount into the worker environment.
    Args:
       src: the name or ID of the volume to mount
       dst_path: the path in the worker environment/container
       read_only: whether the mount should be read only
    """

    src: str
    dst_path: str
    read_only: bool = False


@dataclass
class DeviceMount:
    """
    Defines a host device to mount into the container.
    Args:
       src_path: the path on the host
       dst_path: the path in the worker environment/container
       permissions: the permissions to set on the device. Default: read, write, mknode
    """

    src_path: str
    dst_path: str
    permissions: str = "rwm"


@dataclass
class Workspace:
    """
    Specifies a local "workspace" (a set of directories). Workspaces are ad-hoc built
    into an (usually ephemeral) image. This effectively mirrors the local code changes
    at job submission time.

    For example:

      1. ``projects={"~/github/torch": "torch"}`` copies ``~/github/torch/**`` into ``$REMOTE_WORKSPACE_ROOT/torch/**``
      2. ``projects={"~/github/torch": ""}`` copies ``~/github/torch/**`` into ``$REMOTE_WORKSPACE_ROOT/**``

    The exact location of ``$REMOTE_WORKSPACE_ROOT`` is implementation dependent and varies between
    different implementations of :py:class:`~torchx.workspace.api.WorkspaceMixin`.
    Check the scheduler documentation for details on which workspace it supports.

    Note: ``projects`` maps the location of the local project to a sub-directory in the remote workspace root directory.
    Typically the local project location is a directory path (e.g. ``/home/foo/github/torch``).


    Attributes:
        projects: mapping of local project to the sub-dir in the remote workspace dir.
    """

    projects: dict[str, str]

    def __bool__(self) -> bool:
        """False if no projects mapping. Lets us use workspace object in an if-statement"""
        return bool(self.projects)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Workspace):
            return False
        return self.projects == other.projects

    def __hash__(self) -> int:
        # makes it possible to use Workspace as the key in the workspace build cache
        # see WorkspaceMixin.caching_build_workspace_and_update_role
        return hash(frozenset(self.projects.items()))

    def is_unmapped_single_project(self) -> bool:
        """
        Returns ``True`` if this workspace only has 1 project
        and its target mapping is an empty string.
        """
        return len(self.projects) == 1 and not next(iter(self.projects.values()))

    def merge_into(self, outdir: str | pathlib.Path) -> None:
        """
        Copies each project dir of this workspace into the specified ``outdir``.
        Each project dir is copied into ``{outdir}/{target}`` where ``target`` is
        the target mapping of the project dir.

        For example:

        .. code-block:: python
            from os.path import expanduser

            workspace = Workspace(
                projects={
                    expanduser("~/workspace/torch"): "torch",
                    expanduser("~/workspace/my_project": "")
                }
            )
            workspace.merge_into(expanduser("~/tmp"))

        Copies:

            * ``~/workspace/torch/**`` into ``~/tmp/torch/**``
            * ``~/workspace/my_project/**`` into ``~/tmp/**``

        """

        for src, dst in self.projects.items():
            dst_path = pathlib.Path(outdir) / dst
            if pathlib.Path(src).is_file():
                shutil.copy2(src, dst_path)
            else:  # src is dir
                shutil.copytree(src, dst_path, dirs_exist_ok=True)

    @staticmethod
    def from_str(workspace: str | None) -> "Workspace":
        import yaml

        if not workspace:
            return Workspace({})

        projects = yaml.safe_load(workspace)
        if isinstance(projects, str):  # single project workspace
            projects = {projects: ""}
        else:  # multi-project workspace
            # Replace None mappings with "" (empty string)
            projects = {k: ("" if v is None else v) for k, v in projects.items()}

        return Workspace(projects)

    def __str__(self) -> str:
        """
        Returns a string representation of the Workspace by concatenating
        the project mappings using ';' as a delimiter and ':' between key and value.
        If the single-project workspace with no target mapping, then simply
        returns the src (local project dir)

        NOTE: meant to be used for logging purposes not serde.
          Therefore not symmetric with :py:func:`Workspace.from_str`.

        """
        if self.is_unmapped_single_project():
            return next(iter(self.projects))
        else:
            return ";".join(
                k if not v else f"{k}:{v}" for k, v in self.projects.items()
            )


@dataclass
class Role:
    """
    A set of nodes that perform a specific duty within the ``AppDef``.
    Examples:

    1. Distributed data parallel app - made up of a single role (trainer).

    2. App with parameter server - made up of multiple roles (trainer, ps).

    .. note:: An ``image`` is a software bundle that is installed on the container
              scheduled by the scheduler. The container on the scheduler dictates
              what an image actually is. An image could be as simple as a tar-ball
              or map to a docker image. The scheduler typically knows how to "pull"
              the image given an image name (str), which could be a simple name
              (e.g. docker image) or a url e.g. ``s3://path/my_image.tar``).

    Usage:

    ::

     trainer = Role(name="trainer",
                    image = "pytorch/torch:1",
                    entrypoint = "my_trainer.py"
                    args = ["--arg", "foo", ENV_VAR="FOOBAR"],
                    num_replicas = 4,
                    resource = Resource(cpu=1, gpu=1, memMB=500),
                    port_map={"tcp_store":8080, "tensorboard": 8081},
                    metadata={"local_cwd.property", value})

    Args:
            name: name of the role
            image: a software bundle that is installed on a container.
            entrypoint: command (within the container) to invoke the role
            args: commandline arguments to the entrypoint cmd
            env: environment variable mappings
            num_replicas: number of container replicas to run
            min_replicas: minimum number of replicas for the job to start. When
                set the job size can automatically adjust between min_replicas
                and num_replicas depending on the cluster resources and
                policies. If the scheduler doesn't support auto scaling this
                field is ignored and the job size will be num_replicas.
            max_retries: max number of retries before giving up
            retry_policy: retry behavior upon replica failures
            resource: Resource requirement for the role. The role should be scheduled
                by the scheduler on ``num_replicas`` container, each of them should have at
                least ``resource`` guarantees.
            port_map: Port mapping for the role. The key is the unique identifier of the port
                e.g. "tensorboard": 9090
            metadata: Free form information that is associated with the role, for example
                scheduler specific data. The key should follow the pattern: ``$scheduler.$key``
            mounts: a list of mounts on the machine
            workspace: local project directories to be mirrored on the remote job.
              NOTE: The workspace argument provided to the :py:class:`~torchx.runner.api.Runner` APIs
              only takes effect on ``appdef.role[0]`` and overrides this attribute.

    """

    name: str
    image: str
    min_replicas: Optional[int] = None
    entrypoint: str = MISSING
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    max_retries: int = 0
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION
    resource: Resource = field(default_factory=_null_resource)
    port_map: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    mounts: List[BindMount | VolumeMount | DeviceMount] = field(default_factory=list)
    workspace: Workspace | None = None

    # DEPRECATED DO NOT SET, WILL BE REMOVED SOON
    overrides: Dict[str, Any] = field(default_factory=dict)

    # pyre-ignore
    def __getattribute__(self, attrname: str) -> Any:
        if attrname == "overrides":
            return super().__getattribute__(attrname)
        try:
            ov = super().__getattribute__("overrides")
        except AttributeError:
            ov = {}
        if attrname in ov:
            if inspect.isawaitable(ov[attrname]):
                result = asyncio.get_event_loop().run_until_complete(ov[attrname])
            else:
                result = ov[attrname]()
            setattr(self, attrname, result)
            ov[attrname] = lambda: result
        return super().__getattribute__(attrname)

    def pre_proc(
        self,
        scheduler: str,
        # pyre-fixme[24]: AppDryRunInfo was designed to work with Any request object
        dryrun_info: "AppDryRunInfo",
        # pyre-fixme[24]: AppDryRunInfo was designed to work with Any request object
    ) -> "AppDryRunInfo":
        """
        Modifies the scheduler request based on the role specific configuration.
        The method is invoked for each role during scheduler ``submit_dryrun``.
        If there are multiple roles, the method is invoked for each role in
        order that is defined by the ``AppDef.roles`` list.
        """
        return dryrun_info


@dataclass
class AppDef:
    """
    Represents a distributed application made up of multiple ``Roles``
    and metadata. Contains the necessary information for the driver
    to submit this app to the scheduler.

    Args:
        name: Name of application
        roles: List of roles
        metadata: metadata to the app (treatment of metadata is scheduler dependent)
    """

    name: str
    roles: List[Role] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class AppState(int, Enum):
    """
    State of the application. An application starts from an initial
    ``UNSUBMITTED`` state and moves through ``SUBMITTED``, ``PENDING``,
    ``RUNNING`` states finally reaching a terminal state:
    ``SUCCEEDED``,``FAILED``, ``CANCELLED``.

    If the scheduler supports preemption, the app moves from a ``RUNNING``
    state to ``PENDING`` upon preemption.

    If the user stops the application, then the application state moves
    to ``STOPPED``, then to ``CANCELLED`` when the job is actually cancelled
    by the scheduler.

    1. UNSUBMITTED - app has not been submitted to the scheduler yet
    2. SUBMITTED - app has been successfully submitted to the scheduler
    3. PENDING - app has been submitted to the scheduler pending allocation
    4. RUNNING - app is running
    5. SUCCEEDED - app has successfully completed
    6. FAILED - app has unsuccessfully completed
    7. CANCELLED - app was cancelled before completing
    8. UNKNOWN - app state is unknown
    """

    UNSUBMITTED = 0
    SUBMITTED = 1
    PENDING = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    CANCELLED = 6
    UNKNOWN = 7

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name} ({self.value})"


_TERMINAL_STATES: List[AppState] = [
    AppState.SUCCEEDED,
    AppState.FAILED,
    AppState.CANCELLED,
]

_STARTED_STATES: List[AppState] = _TERMINAL_STATES + [
    AppState.RUNNING,
]


def is_terminal(state: AppState) -> bool:
    return state in _TERMINAL_STATES


def is_started(state: AppState) -> bool:
    return state in _STARTED_STATES


# =======================
# ==== Status API =======
# =======================

# replica and app share the same states, simply alias it for now
ReplicaState = AppState


@dataclass
class ReplicaStatus:
    """
    The status of the replica during the job execution.

    Args:
        id: The node rank, note: this is not a worker rank.
        state: The current state of the node.
        role: The role name
        hostname: The hostname where the replica is running
        structured_error_msg: Error message if any, None if job succeeded.
    """

    id: int
    state: ReplicaState
    role: str
    hostname: str
    structured_error_msg: str = NONE


@dataclass
class RoleStatus:
    """
    The status of the role during the job execution.

    Args:
        role: Role name
        replicas: List of replica statuses
    """

    role: str
    replicas: List[ReplicaStatus]

    def to_json(self) -> Dict[str, Any]:
        """
        Convert the RoleStatus to a json object.
        """
        return {
            "role": self.role,
            "replicas": [asdict(replica) for replica in self.replicas],
        }


@dataclass
class AppStatus:
    """
    The runtime status of the ``AppDef``. The scheduler can
    return an arbitrary text message (msg field).
    If any error occurs, scheduler can populate ``structured_error_msg``
    with json response.

    ``replicas`` represent the statuses of the replicas in the job. If the job
    runs with multiple retries, the parameter will contain the statuses of the
    most recent retry. Note: if the previous retries failed, but the most recent
    retry succeeded or in progress, ``replicas`` will not contain occurred errors.
    """

    state: AppState
    num_restarts: int = 0
    msg: str = ""
    structured_error_msg: str = NONE
    ui_url: Optional[str] = None
    roles: List[RoleStatus] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def __repr__(self) -> str:
        app_status_dict = asdict(self)
        structured_error_msg = app_status_dict.pop("structured_error_msg")
        if structured_error_msg != NONE:
            structured_error_msg_parsed = json.loads(structured_error_msg)
        else:
            structured_error_msg_parsed = NONE
        app_status_dict["structured_error_msg"] = structured_error_msg_parsed
        app_status_dict["state"] = repr(app_status_dict["state"])

        import yaml

        return yaml.dump({"AppStatus": app_status_dict})

    def raise_for_status(self) -> None:
        """
        raise_for_status will raise an AppStatusError if the state is not SUCCEEDED.
        """
        if self.state != AppState.SUCCEEDED:
            raise AppStatusError(self, f"job did not succeed: {self}")

    def _format_error_message(self, msg: str, header: str, width: int = 80) -> str:
        assert len(header) < width

        match = re.search(_RPC_ERROR_MESSAGE_RE, msg)
        if match:
            start_pos, end_pos = match.span()
            msg = msg[start_pos:end_pos]

        match = re.search(_EMBEDDED_ERROR_MESSAGE_RE, msg)
        if match:
            msg = match.group("msg")

        length = 0
        lines = []
        for i in range(len(msg) + 1):
            if (i == (len(msg))) or (msg[i] == " " and length >= width):
                lines.append(f"{header}{msg[i - length: i]}")
                header = " " * len(header)
                length = 0
            length += 1
        return "\n".join(lines)

    def _format_replica_status(self, replica_status: ReplicaStatus) -> str:
        if replica_status.structured_error_msg != NONE:
            try:
                error_data = json.loads(replica_status.structured_error_msg)
            except JSONDecodeError:
                return replica_status.structured_error_msg
            error_message = self._format_error_message(
                msg=error_data["message"]["message"], header="    error_msg: "
            )
            timestamp = int(error_data["message"]["extraInfo"]["timestamp"])
            exitcode = error_data["message"]["errorCode"]
            if not exitcode:
                exitcode = "<N/A>"
            data = f"""{str(replica_status.state)} (exitcode: {exitcode})
        timestamp: {datetime.fromtimestamp(timestamp)}
        hostname: {replica_status.hostname}
    {error_message}"""
        else:
            data = f"{str(replica_status.state)}"
            if replica_status.state in [
                ReplicaState.CANCELLED,
                ReplicaState.FAILED,
            ]:
                data += " (no reply file)"

        # mark index 0 for each role with a "*" for a visual queue on role boundaries
        header = " "
        if replica_status.id == 0:
            header = "*"

        return f"\n {header}{replica_status.role}[{replica_status.id}]:{data}"

    def _get_role_statuses(
        self, roles: List[RoleStatus], filter_roles: Optional[List[str]] = None
    ) -> List[RoleStatus]:
        if not filter_roles:
            return roles
        return [
            role_status for role_status in roles if role_status.role in filter_roles
        ]

    def _format_role_status(
        self,
        role_status: RoleStatus,
    ) -> str:
        replica_data = ""

        for replica in sorted(role_status.replicas, key=lambda r: r.id):
            replica_data += self._format_replica_status(replica)
        return f"{replica_data}"

    def to_json(self, filter_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert the AppStatus to a json object, including RoleStatus.
        """
        roles = self._get_role_statuses(self.roles, filter_roles)

        return {
            "state": str(self.state),
            "num_restarts": self.num_restarts,
            "roles": [role_status.to_json() for role_status in roles],
            "msg": self.msg,
            "structured_error_msg": self.structured_error_msg,
            "url": self.ui_url,
        }

    def format(
        self,
        filter_roles: Optional[List[str]] = None,
    ) -> str:
        """
        Format logs for app status. The app status include:
            1. State: State of the application.
            2. Num Restarts: The number of application restarts.
            3. Roles: List of roles.
            4. Msg: Arbitrary text message the scheduler returned.
            5. Structured Error Msg: Json response error msg.
            6. UI URL: Application URL
        """
        roles_data = ""
        roles = self._get_role_statuses(self.roles, filter_roles)

        for role_status in roles:
            roles_data += self._format_role_status(role_status)
        return Template(_APP_STATUS_FORMAT_TEMPLATE).substitute(
            state=self.state,
            num_restarts=self.num_restarts,
            roles=roles_data,
            msg=self.msg,
            structured_error_msg=self.structured_error_msg,
            url=self.ui_url,
        )


class AppStatusError(Exception):
    """
    AppStatusError is raised when the job status is in an exceptional state i.e.
    not SUCCEEDED.
    """

    def __init__(self, status: AppStatus, *args: object) -> None:
        super().__init__(*args)

        self.status = status


# valid run cfg values; only support primitives (str, int, float, bool, List[str], Dict[str, str])
# TODO(wilsonhong): python 3.9+ supports list[T] in typing, which can be used directly
# in isinstance(). Should replace with that.
# see: https://docs.python.org/3/library/stdtypes.html#generic-alias-type
CfgVal = Union[str, int, float, bool, List[str], Dict[str, str], None]


T = TypeVar("T")


class AppDryRunInfo(Generic[T]):
    """
    Returned by ``Scheduler.submit_dryrun``. Represents the
    request that would have been made to the scheduler.
    The ``fmt_str()`` method of this object should return a
    pretty formatted string representation of the underlying
    request object such that ``print(info)`` yields a human
    readable representation of the underlying request.
    """

    def __init__(self, request: T, fmt: Callable[[T], str]) -> None:
        self.request = request
        self._fmt = fmt

        # fields below are only meant to be used by
        # Scheduler or Session implementations
        # and are back references to the parameters
        # to dryrun() that returned this AppDryRunInfo object
        # thus they are set in Runner.dryrun() and Scheduler.submit_dryrun()
        # manually rather than through constructor arguments
        # DO NOT create getters or make these public
        # unless there is a good reason to
        self._app: Optional[AppDef] = None
        self._cfg: Mapping[str, CfgVal] = {}
        self._scheduler: Optional[str] = None

    def __repr__(self) -> str:
        return self._fmt(self.request)


def get_type_name(tp: Type[CfgVal]) -> str:
    """
    Gets the type's name as a string. If ``tp` is a primitive class like int, str, etc, then
    uses its attribute ``__name__``. Otherwise, use ``str(tp)``.

    Note: we use this method to print out generic typing like List[str].
    """
    if tp.__module__ != "typing" and hasattr(tp, "__name__"):
        return tp.__name__
    else:
        return str(tp)


@dataclass
class runopt:
    """
    Represents the metadata about the specific run option
    """

    default: CfgVal
    opt_type: Type[CfgVal]
    is_required: bool
    help: str
    aliases: list[str] | None = None
    deprecated_aliases: list[str] | None = None

    @property
    def is_type_list_of_str(self) -> bool:
        """
        Checks if the option type is a list of strings.

        Returns:
            bool: True if the option type is either List[str] or list[str], False otherwise.
        """
        return self.opt_type in (List[str], list[str])

    @property
    def is_type_dict_of_str(self) -> bool:
        """
        Checks if the option type is a dict of string keys to string values.

        Returns:
            bool: True if the option type is either Dict[str, str] or dict[str, str], False otherwise.
        """
        return self.opt_type in (Dict[str, str], dict[str, str])

    def cast_to_type(self, value: str) -> CfgVal:
        """Casts the given `value` (in its string representation) to the type of this run option.
        Below are the cast rules for each option type and value literal:

        1. opt_type=str, value="foo" -> "foo"
        1. opt_type=bool, value="True"/"False" -> True/False
        1. opt_type=int, value="1" -> 1
        1. opt_type=float, value="1.1" -> 1.1
        1. opt_type=list[str]/List[str], value="a,b,c" or value="a;b;c" -> ["a", "b", "c"]
        1. opt_type=dict[str,str]/Dict[str,str],
           value="key1:val1,key2:val2" or value="key1:val1;key2:val2" -> {"key1": "val1", "key2": "val2"}

        NOTE: dict parsing uses ":" as the kv separator (rather than the standard "=") because "=" is used
        at the top-level cfg to parse runopts (notice the plural) from the CLI. Originally torchx only supported
        primitives and list[str] as CfgVal but dict[str,str] was added in https://github.com/meta-pytorch/torchx/pull/855
        """

        if self.opt_type is None:
            raise ValueError("runopt's opt_type cannot be `None`")
        elif self.opt_type == bool:
            return value.lower() == "true"
        elif self.opt_type in (List[str], list[str]):
            # lists may be ; or , delimited
            # also deal with trailing "," by removing empty strings
            return [v for v in value.replace(";", ",").split(",") if v]
        elif self.opt_type in (Dict[str, str], dict[str, str]):
            return {
                s.split(":", 1)[0]: s.split(":", 1)[1]
                for s in value.replace(";", ",").split(",")
            }
        else:
            assert self.opt_type in (str, int, float)
            return self.opt_type(value)


class runopts:
    """
    Holds the accepted scheduler run configuration
    keys, default value (if any), and help message string.
    These options are provided by the ``Scheduler`` and validated
    in ``Session.run`` against user provided run cfg.
    Allows ``None`` default values. Required opts must NOT have a
    non-None default.

    .. important:: This class has no accessors because it is intended to
                   be constructed and returned by ``Scheduler.run_config_options``
                   and printed out as a "help" tool or as part of an exception msg.

    Usage:

    .. code-block:: python

     opts = runopts()

     opts.add("run_as_user", type_=str, help="user to run the job as")
     opts.add("cluster_id", type_=int, help="cluster to submit the job", required=True)
     opts.add("priority", type_=float, default=0.5, help="job priority")
     opts.add("preemptible", type_=bool, default=False, help="is the job preemptible")

     # invalid
     opts.add("illegal", default=10, required=True)
     opts.add("bad_type", type=str, default=10)

     opts.check(cfg)
     print(opts)

    """

    def __init__(self) -> None:
        self._opts: Dict[str, runopt] = {}
        self._alias_to_key: dict[str, str] = {}

    def __iter__(self) -> Iterator[Tuple[str, runopt]]:
        return self._opts.items().__iter__()

    def __len__(self) -> int:
        return len(self._opts)

    @staticmethod
    def is_type(obj: CfgVal, tp: Type[CfgVal]) -> bool:
        """
        Returns True if ``obj`` is type of ``tp``. Similar to isinstance() but supports
        tp = List[str], thus can be used to validate ConfigValue.
        """
        try:
            return isinstance(obj, tp)
        except TypeError:
            if isinstance(obj, list):
                return all(isinstance(e, str) for e in obj)
            elif isinstance(obj, dict):
                return all(
                    isinstance(k, str) and isinstance(v, str) for k, v in obj.items()
                )
            else:
                return False

    def get(self, name: str) -> Optional[runopt]:
        """
        Returns option if any was registered, or None otherwise.
        First searches for the option by ``name``, then falls-back to matching ``name`` with any
        registered aliases.

        """
        if name in self._opts:
            return self._opts[name]
        if name in self._alias_to_key:
            return self._opts[self._alias_to_key[name]]
        return None

    def resolve(self, cfg: Mapping[str, CfgVal]) -> Dict[str, CfgVal]:
        """
        Checks the given config against this ``runopts`` and sets default configs
        if not set.

        .. note:: Extra configs unknown to this run option are ignored.

        """

        resolved_cfg: Dict[str, CfgVal] = {**cfg}

        for cfg_key, runopt in self._opts.items():
            val = resolved_cfg.get(cfg_key)
            resolved_name = None
            aliases = runopt.aliases or []
            deprecated_aliases = runopt.deprecated_aliases or []
            if val is None:
                for alias in aliases:
                    val = resolved_cfg.get(alias)
                    if alias in cfg or val is not None:
                        resolved_name = alias
                        break
                for alias in deprecated_aliases:
                    val = resolved_cfg.get(alias)
                    if val is not None:
                        resolved_name = alias
                        use_instead = self._alias_to_key.get(alias)
                        warnings.warn(
                            f"Run option `{alias}` is deprecated, use `{use_instead}` instead",
                            UserWarning,
                            stacklevel=2,
                        )
                        break
            else:
                resolved_name = cfg_key
                for alias in aliases:
                    duplicate_val = resolved_cfg.get(alias)
                    if alias in cfg or duplicate_val is not None:
                        raise InvalidRunConfigException(
                            f"Duplicate opt name. runopt: `{resolved_name}``, is an alias of runopt: `{alias}`",
                            resolved_name,
                            cfg,
                        )

            # check required opt
            if runopt.is_required and val is None:
                raise InvalidRunConfigException(
                    f"Required run option: {cfg_key}, must be provided and not `None`",
                    cfg_key,
                    cfg,
                )

            # check type (None matches all types)
            if val is not None and not runopts.is_type(val, runopt.opt_type):
                raise InvalidRunConfigException(
                    f"Run option: {cfg_key}, must be of type: {get_type_name(runopt.opt_type)},"
                    f" but was: {val} ({type(val).__name__})",
                    cfg_key,
                    cfg,
                )

            # not required and not set, set to default
            if val is None and resolved_name is None:
                resolved_cfg[cfg_key] = runopt.default
        return resolved_cfg

    def cfg_from_str(self, cfg_str: str) -> Dict[str, CfgVal]:
        """
        Parses scheduler ``cfg`` from a string literal and returns
        a cfg map where the cfg values have been cast into the appropriate
        types as specified by this runopts object. Unknown keys are ignored
        and not returned in the resulting map.

        .. note:: Unlike the method ``resolve``, this method does NOT resolve
                  default options or check that the required options are actually
                  present in the given ``cfg_str``. This method is intended to be
                  called before calling ``resolve()`` when the input is a string
                  encoded run cfg. That is to fully resolve the cfg, call
                  ``opt.resolve(opt.cfg_from_str(cfg_literal))``.

        If the ``cfg_str`` is an empty string, then an empty
        ``cfg`` is returned. Otherwise, at least one kv-pair delimited by
        ``"="`` (equal) is expected.

        Either ``","`` (comma) or ``";"`` (semi-colon)
        can be used to delimit multiple kv-pairs.

        ``CfgVal`` allows ``List`` of primitives, which can be passed as
        either ``","`` or ``";"`` (semi-colon) delimited. Since the same
        delimiters are used to delimit between cfg kv pairs, this method
        interprets the last (trailing) ``","`` or ``";"`` as the delimiter between
        kv pairs. See example below.



        Examples:

        .. doctest::

         opts = runopts()
         opts.add("FOO", type_=List[str], default=["a"], help="an optional list option")
         opts.add("BAR", type_=str, required=True, help="a required str option")

         # required and default options not checked
         # method returns strictly parsed cfg from the cfg literal string
         opts.cfg_from_str("") == {}

         # however, unknown options are ignored
         # since the value type is unknown hence cannot cast to the correct type
         opts.cfg_from_str("UNKNOWN=VALUE") == {}

         opts.cfg_from_str("FOO=v1") == {"FOO": "v1"}

         opts.cfg_from_str("FOO=v1,v2") == {"FOO": ["v1", "v2"]}
         opts.cfg_from_str("FOO=v1;v2") == {"FOO": ["v1", "v2"]}

         opts.cfg_from_str("FOO=v1,v2,BAR=v3") == {"FOO": ["v1", "v2"], "BAR": "v3"}
         opts.cfg_from_str("FOO=v1;v2,BAR=v3") == {"FOO": ["v1", "v2"], "BAR": "v3"}
         opts.cfg_from_str("FOO=v1;v2;BAR=v3") == {"FOO": ["v1", "v2"], "BAR": "v3"}

        """

        cfg: Dict[str, CfgVal] = {}
        for key, val in to_dict(cfg_str).items():
            opt = self.get(key)
            if opt:
                cfg[key] = opt.cast_to_type(val)
            else:
                logger.warning(
                    f"{YELLOW_BOLD}Unknown run option passed to scheduler: {key}={val}{RESET}"
                )
        return cfg

    def cfg_from_json_repr(self, json_repr: str) -> Dict[str, CfgVal]:
        """
        Converts the given dict to a valid cfg for this ``runopts`` object.
        """
        cfg: Dict[str, CfgVal] = {}
        cfg_dict = json.loads(json_repr)
        for key, val in cfg_dict.items():
            opt = self.get(key)
            if opt:
                # Optional runopt cfg values default their value to None,
                # but use `_type` to specify their type when provided.
                # Make sure not to treat None's as lists/dictionaries
                if val is None:
                    cfg[key] = val
                elif opt.is_type_list_of_str:
                    cfg[key] = [str(v) for v in val]
                elif opt.is_type_dict_of_str:
                    cfg[key] = {str(k): str(v) for k, v in val.items()}
                else:
                    cfg[key] = val
        return cfg

    def add(
        self,
        cfg_key: str,
        type_: Type[CfgVal],
        help: str,
        default: CfgVal = None,
        required: bool = False,
        aliases: Optional[list[str]] = None,
        deprecated_aliases: Optional[list[str]] = None,
    ) -> None:
        """
        Adds the ``config`` option with the given help string and ``default``
        value (if any). If the ``default`` is not specified then this option
        is a required option.
        """
        aliases = aliases or []
        deprecated_aliases = deprecated_aliases or []
        if required and default is not None:
            raise ValueError(
                f"Required option: {cfg_key} must not specify default value. Given: {default}"
            )
        if default is not None:
            if not runopts.is_type(default, type_):
                raise TypeError(
                    f"Option: {cfg_key}, must be of type: {type_}."
                    f" Given: {default} ({type(default).__name__})"
                )

        opt = runopt(
            default,
            type_,
            required,
            help,
            list(set(aliases)),
            list(set(deprecated_aliases)),
        )
        for alias in aliases:
            self._alias_to_key[alias] = cfg_key
        for deprecated_alias in deprecated_aliases:
            self._alias_to_key[deprecated_alias] = cfg_key
        self._opts[cfg_key] = opt

    def update(self, other: "runopts") -> None:
        self._opts.update(other._opts)

    def __repr__(self) -> str:
        required = [(key, opt) for key, opt in self._opts.items() if opt.is_required]
        optional = [
            (key, opt) for key, opt in self._opts.items() if not opt.is_required
        ]

        out = "    usage:\n        "
        for i, (key, opt) in enumerate(required + optional):
            contents = f"{key}={key.upper()}"
            if not opt.is_required:
                contents = f"[{contents}]"
            if i > 0:
                contents = "," + contents
            out += contents

        sections = [("required", required), ("optional", optional)]

        for section, opts in sections:
            if len(opts) == 0:
                continue
            out += f"\n\n    {section} arguments:"
            for key, opt in opts:
                default = "" if opt.is_required else f", {opt.default}"
                out += f"\n        {key}={key.upper()} ({get_type_name(opt.opt_type)}{default})"
                out += f"\n            {opt.help}"

        return out


class InvalidRunConfigException(Exception):
    """
    Raised when the supplied run cfg does not satisfy the
    ``runopts``, either due to missing required configs or value
    type mismatch.
    """

    def __init__(
        self, invalid_reason: str, cfg_key: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        given = str(cfg) if cfg else "<EMPTY>"
        super().__init__(f"{invalid_reason}. Given: {given}")
        self.cfg_key = cfg_key


class MalformedAppHandleException(Exception):
    """
    Raised when APIs are given a bad app handle.
    """

    def __init__(self, app_handle: str) -> None:
        super().__init__(
            f"{app_handle} is not of the form: <scheduler_backend>://<session_name>/<app_id>"
        )


class UnknownSchedulerException(Exception):
    def __init__(self, scheduler_backend: str) -> None:
        super().__init__(
            f"Scheduler backend: {scheduler_backend} does not exist."
            f" Use session.scheduler_backends() to see all supported schedulers"
        )


# encodes information about a running app in url format
# {scheduler_backend}://{session_name}/{app_id}
AppHandle = str


class ParsedAppHandle(NamedTuple):
    """
    Individual accessible components of the `AppHandle`
    """

    scheduler_backend: str
    session_name: str
    app_id: str


class UnknownAppException(Exception):
    """
    Raised by ``Session`` APIs when either the application does not
    exist or the application is not owned by the session.
    """

    def __init__(self, app_handle: "AppHandle") -> None:
        super().__init__(
            f"Unknown app = {app_handle}. Did you forget to call session.run()?"
            f" Otherwise, the app may have already finished and purged by the scheduler"
        )


def parse_app_handle(app_handle: AppHandle) -> ParsedAppHandle:
    """
    Parses the app handle into ```(scheduler_backend, session_name, and app_id)```.

    Example:

    .. doctest::

     assert parse_app_handle("k8s://default/foo_bar") == ("k8s", "default", "foo_bar")
     assert parse_app_handle("k8s:///foo_bar") == ("k8s", "", "foo_bar")

    Args:
        app_handle: a URI of the form ``{scheduler}://{session_name}/{app_id}``,
            where the ``session_name`` is optional. In this case the app handle is
            of the form ``{scheduler}:///{app_id}`` (notice the triple slashes).

    Returns: A ``Tuple`` of three elements, ``(scheduler, session_name, app_id)``
        parsed from the app_handle URI str. If the session name is not present then
        an empty string is returned in its place in the tuple.

    """

    # parse it manually b/c currently torchx does not
    # define allowed characters nor length for session name and app_id
    import re

    pattern = r"(?P<scheduler_backend>.+)://(?P<session_name>.*)/(?P<app_id>.+)"
    match = re.match(pattern, app_handle)
    if not match:
        raise MalformedAppHandleException(app_handle)
    gd = match.groupdict()
    return ParsedAppHandle(gd["scheduler_backend"], gd["session_name"], gd["app_id"])
