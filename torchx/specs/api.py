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
    Pattern,
    Type,
    TypeVar,
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
    Defaults to ``~/.torchx`` and is overridable via the ``TORCHX_HOME`` environment variable.

    .. doctest::

        >>> from torchx.specs import TORCHX_HOME
        >>> import os, pathlib
        >>> _ = os.environ.pop("TORCHX_HOME", None)  # ensure default
        >>> TORCHX_HOME() == pathlib.Path.home() / ".torchx"
        True
        >>> TORCHX_HOME("conda-pack-out") == pathlib.Path.home() / ".torchx" / "conda-pack-out"
        True

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
    Represents resource requirements for a :py:class:`Role`.

    .. important:: Prefer :py:func:`~torchx.specs.resource` with named resources
                   (t-shirt sizes) over specifying raw values directly.

    .. doctest::

        >>> from torchx.specs import Resource
        >>> Resource(cpu=4, gpu=1, memMB=8192)
        Resource(cpu=4, gpu=1, memMB=8192, capabilities={}, devices={}, tags={})

    Args:
        cpu: number of logical cpu cores
        gpu: number of gpus
        memMB: MB of ram
        capabilities: additional hardware specs (interpreted by scheduler)
        devices: named devices with their quantities (e.g. ``{"vpc.amazonaws.com/efa": 1}``)
        tags: metadata tags (not interpreted by schedulers)
    """

    cpu: int
    gpu: int
    memMB: int
    capabilities: dict[str, Any] = field(default_factory=dict)
    devices: dict[str, int] = field(default_factory=dict)
    tags: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def copy(original: "Resource", **capabilities: Any) -> "Resource":
        """Copies a resource, merging in the given ``capabilities``."""

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
    Template variables substituted at runtime in :py:attr:`Role.args`,
    :py:attr:`Role.env`, and :py:attr:`Role.metadata`.

    .. warning:: Macros in other :py:class:`Role` fields are NOT substituted.

    Available macros:

    1. ``img_root`` — root directory of the pulled image
    2. ``app_id`` — application id as assigned by the scheduler
    3. ``replica_id`` — per-role replica index (``0, 1, ...``).
       When a replica is replaced after failure, the replacement retains
       the same ``replica_id``.

    .. doctest::

        >>> from torchx.specs import AppDef, Role, macros
        >>> trainer = Role(
        ...     name="trainer",
        ...     image="my_image:latest",
        ...     entrypoint="train.py",
        ...     args=["--app_id", macros.app_id],
        ...     env={"IMG_ROOT": macros.img_root},
        ... )
        >>> app = AppDef("train_app", roles=[trainer])

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
            """Returns a deep copy of ``role`` with macros substituted."""

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

        def _apply_nested(self, d: dict[str, Any]) -> dict[str, Any]:  # noqa: D102
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
                            if isinstance(v[i], dict):
                                stack.append(v[i])
                            elif isinstance(v[i], str):
                                v[i] = self.substitute(v[i])
            return d

        def to_dict(self) -> dict[str, Any]:
            """Returns the macro values as a plain dict."""
            return asdict(self)

        def substitute(self, arg: str) -> str:
            """Substitutes macro placeholders in ``arg``."""
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
    """Bind-mounts a host path into the worker container."""

    src_path: str
    dst_path: str
    read_only: bool = False


@dataclass
class VolumeMount:
    """Mounts a persistent volume into the worker container."""

    src: str
    dst_path: str
    read_only: bool = False


@dataclass
class DeviceMount:
    """Mounts a host device into the container."""

    src_path: str
    dst_path: str
    permissions: str = "rwm"


@dataclass
class Workspace:
    """
    Maps local project directories to remote workspace locations. At submit-time,
    files are copied/synced so that the remote job mirrors local code changes.

    .. doctest::

        >>> from torchx.specs import Workspace
        >>> # copies ~/github/torch/** into $REMOTE_ROOT/torch/**
        >>> ws = Workspace(projects={"~/github/torch": "torch"})
        >>> # copies ~/github/torch/** into $REMOTE_ROOT/** (no sub-dir)
        >>> ws = Workspace(projects={"~/github/torch": ""})

    The exact ``$REMOTE_ROOT`` is implementation-dependent. See
    :py:class:`~torchx.workspace.api.WorkspaceMixin` and scheduler docs.

    Args:
        projects: ``{local_path: remote_subdir}`` mapping.
    """

    projects: dict[str, str]

    def __bool__(self) -> bool:
        return bool(self.projects)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Workspace):
            return False
        return self.projects == other.projects

    def __hash__(self) -> int:
        return hash(frozenset(self.projects.items()))

    def is_unmapped_single_project(self) -> bool:
        """``True`` if this is a single-project workspace with no target sub-directory."""
        return len(self.projects) == 1 and not next(iter(self.projects.values()))

    def merge_into(self, outdir: str | pathlib.Path) -> None:
        """Copies each project into ``{outdir}/{target}``."""

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
        # Logging-only representation; not symmetric with from_str().
        if self.is_unmapped_single_project():
            return next(iter(self.projects))
        else:
            return ";".join(
                k if not v else f"{k}:{v}" for k, v in self.projects.items()
            )


@dataclass
class Role:
    """
    A set of nodes that perform a specific duty within an :py:class:`AppDef`.

    * DDP app — single role (``trainer``)
    * Parameter-server app — multiple roles (``trainer``, ``ps``)

    .. doctest::

        >>> from torchx.specs import Role, Resource
        >>> trainer = Role(
        ...     name="trainer",
        ...     image="pytorch/torch:latest",
        ...     entrypoint="train.py",
        ...     args=["--lr", "0.01"],
        ...     num_replicas=4,
        ...     resource=Resource(cpu=4, gpu=1, memMB=8192),
        ... )

    Args:
        name: name of the role
        image: software bundle installed on the container (docker image, fbpkg, tar-ball, etc.)
        entrypoint: command to invoke inside the container
        args: arguments to the entrypoint
        env: environment variable mappings
        num_replicas: number of container replicas
        min_replicas: minimum replicas for elastic scaling. If unset or unsupported
            by the scheduler, the job runs at ``num_replicas``.
        max_retries: max number of retries before giving up
        retry_policy: retry behavior upon failures
        resource: resource requirements per replica
        port_map: named port mappings (e.g. ``{"tensorboard": 8081}``)
        metadata: scheduler-specific data. Keys should follow ``$scheduler.$key``.
        mounts: bind, volume, or device mounts
        workspace: local project directories to mirror on the remote job.
            The ``workspace`` argument on :py:class:`~torchx.runner.api.Runner`
            APIs overrides this on ``roles[0]``.
    """

    name: str
    image: str
    min_replicas: int | None = None
    entrypoint: str = MISSING
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    max_retries: int = 0
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION
    resource: Resource = field(default_factory=_null_resource)
    port_map: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    mounts: list[BindMount | VolumeMount | DeviceMount] = field(default_factory=list)
    workspace: Workspace | None = None

    # DEPRECATED DO NOT SET, WILL BE REMOVED SOON
    overrides: dict[str, Any] = field(default_factory=dict)

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
        """Hook for role-specific scheduler request modifications.

        Called per-role during :py:meth:`Scheduler.submit_dryrun <torchx.schedulers.api.Scheduler.submit_dryrun>`,
        in the order they appear in :py:attr:`AppDef.roles`.
        """
        return dryrun_info


@dataclass
class AppDef:
    """A distributed application composed of one or more :py:class:`Role` s.

    .. doctest::

        >>> from torchx.specs import AppDef, Role
        >>> app = AppDef(
        ...     name="my_train",
        ...     roles=[Role(name="trainer", image="my_image:latest")],
        ... )

    Args:
        metadata: scheduler-specific metadata (treatment varies by scheduler)
    """

    name: str
    roles: list[Role] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


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


_TERMINAL_STATES: list[AppState] = [
    AppState.SUCCEEDED,
    AppState.FAILED,
    AppState.CANCELLED,
]

_STARTED_STATES: list[AppState] = _TERMINAL_STATES + [
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
    """Status of a single replica during job execution.

    Args:
        id: node rank (not worker rank)
        hostaddr: DNS name or IP of the container. Defaults to ``hostname``.
    """

    id: int
    state: ReplicaState
    role: str
    hostname: str
    structured_error_msg: str = NONE
    hostaddr: str | None = None

    def __post_init__(self) -> None:
        if self.hostaddr is None:
            self.hostaddr = self.hostname


@dataclass
class RoleStatus:
    """Status of all replicas within a role."""

    role: str
    replicas: list[ReplicaStatus]

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "replicas": [asdict(replica) for replica in self.replicas],
        }


@dataclass
class AppStatus:
    """Runtime status of an :py:class:`AppDef`.

    ``roles`` contains replica statuses for the most recent retry only.
    """

    state: AppState
    num_restarts: int = 0
    msg: str = ""
    structured_error_msg: str = NONE
    ui_url: str | None = None
    roles: list[RoleStatus] = field(default_factory=list)

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
        """Raises :py:class:`AppStatusError` if state is not ``SUCCEEDED``."""
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
        self, roles: list[RoleStatus], filter_roles: list[str] | None = None
    ) -> list[RoleStatus]:
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

    def to_json(self, filter_roles: list[str] | None = None) -> dict[str, Any]:
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
        filter_roles: list[str] | None = None,
    ) -> str:
        """Human-readable status string."""
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
    """Raised by :py:meth:`AppStatus.raise_for_status` when state is not ``SUCCEEDED``."""

    def __init__(self, status: AppStatus, *args: object) -> None:
        super().__init__(*args)

        self.status = status


# valid run cfg values; only support primitives (str, int, float, bool, list[str], dict[str, str])
CfgVal = str | int | float | bool | list[str] | dict[str, str] | None


T = TypeVar("T")


class AppDryRunInfo(Generic[T]):
    """Returned by :py:meth:`Scheduler.submit_dryrun <torchx.schedulers.api.Scheduler.submit_dryrun>`.

    Wraps the scheduler ``request`` that *would* have been submitted.
    ``print(info)`` yields a human-readable representation.
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
        self._app: AppDef | None = None
        self._cfg: Mapping[str, CfgVal] = {}
        self._scheduler: str | None = None

    def __repr__(self) -> str:
        return self._fmt(self.request)


def get_type_name(tp: Type[CfgVal]) -> str:
    """Returns a human-readable name for ``tp`` (handles generic types like ``list[str]``)."""
    if tp.__module__ != "typing" and hasattr(tp, "__name__"):
        return tp.__name__
    else:
        return str(tp)


class cases:
    """Case conversion utilities."""

    @staticmethod
    def snake_to_camel(name: str) -> str:
        """Convert snake_case to camelCase."""
        components = name.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    @staticmethod
    def camel_to_snake(name: str) -> str:
        """Convert camelCase to snake_case."""
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


@dataclass
class runopt:
    """Metadata for a single scheduler run option."""

    default: CfgVal
    opt_type: Type[CfgVal]
    is_required: bool
    help: str

    @property
    def is_type_list_of_str(self) -> bool:
        return self.opt_type in (List[str], list[str])

    @property
    def is_type_dict_of_str(self) -> bool:
        return self.opt_type in (Dict[str, str], dict[str, str])

    def cast_to_type(self, value: str) -> CfgVal:
        """Casts the given `value` (in its string representation) to the type of this run option.
        Below are the cast rules for each option type and value literal:

        1. opt_type=str, value="foo" -> "foo"
        1. opt_type=bool, value="True"/"False" -> True/False
        1. opt_type=int, value="1" -> 1
        1. opt_type=float, value="1.1" -> 1.1
        1. opt_type=list[str]/List[str], value="a,b,c" or value="a;b;c" -> ["a", "b", "c"]
        1. opt_type=dict[str,str]/Dict[str,str], value="a:1,b:2" or value="a:1;b:2" -> {"a": "1", "b": "2"}

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
    Schema for scheduler run configuration.

    Holds accepted config keys, defaults, and help strings. Constructed by
    :py:meth:`Scheduler.run_opts() <torchx.schedulers.api.Scheduler.run_opts>`
    and validated at submit time.

    .. doctest::

        >>> from torchx.specs import runopts
        >>> opts = runopts()
        >>> opts.add("cluster_id", type_=int, help="cluster to submit the job", required=True)
        >>> opts.add("priority", type_=float, default=0.5, help="job priority")
        >>> opts.add("preemptible", type_=bool, default=False, help="is the job preemptible")

    .. note:: For new schedulers, prefer :py:class:`~torchx.schedulers.api.StructuredOpts`
              which auto-generates ``runopts`` from typed dataclass fields.
    """

    def __init__(self) -> None:
        self._opts: dict[str, runopt] = {}

    def __iter__(self) -> Iterator[tuple[str, runopt]]:
        return self._opts.items().__iter__()

    def __len__(self) -> int:
        return len(self._opts)

    @staticmethod
    def is_type(obj: CfgVal, tp: Type[CfgVal]) -> bool:
        """Like ``isinstance()`` but supports generic types (e.g. ``list[str]``)."""
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

    def get(self, name: str) -> runopt | None:
        """Returns the registered option, or ``None``.

        Accepts camelCase names (e.g. ``"clusterName"`` resolves ``"cluster_name"``).
        """
        # _opts maps names to runopt instances (never None), so a None
        # result unambiguously means the key does not exist.
        result = self._opts.get(name)
        if result is None:
            snake = cases.camel_to_snake(name)
            if snake != name:
                result = self._opts.get(snake)
        return result

    def resolve(self, cfg: Mapping[str, CfgVal]) -> dict[str, CfgVal]:
        """Validates ``cfg`` against registered options, filling defaults.

        Raises :py:class:`InvalidRunConfigException` for missing required options
        or type mismatches. Accepts camelCase keys.
        """

        resolved_cfg: dict[str, CfgVal] = {**cfg}

        for cfg_key, runopt in self._opts.items():
            val = resolved_cfg.get(cfg_key)

            # Fallback: try camelCase version of the registered key in cfg
            if val is None and cfg_key not in resolved_cfg:
                camel_key = cases.snake_to_camel(cfg_key)
                if camel_key != cfg_key and camel_key in resolved_cfg:
                    val = resolved_cfg.pop(camel_key)
                    resolved_cfg[cfg_key] = val

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
            if val is None and cfg_key not in resolved_cfg:
                resolved_cfg[cfg_key] = runopt.default
        return resolved_cfg

    def cfg_from_str(self, cfg_str: str) -> dict[str, CfgVal]:
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

        cfg: dict[str, CfgVal] = {}
        for key, val in to_dict(cfg_str).items():
            opt = self.get(key)
            if opt:
                cfg[key] = opt.cast_to_type(val)
            else:
                logger.warning(
                    f"{YELLOW_BOLD}Unknown run option passed to scheduler: {key}={val}{RESET}"
                )
        return cfg

    def cfg_from_json_repr(self, json_repr: str) -> dict[str, CfgVal]:
        """
        Converts the given dict to a valid cfg for this ``runopts`` object.
        """
        cfg: dict[str, CfgVal] = {}
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
    ) -> None:
        """Registers a config option. Required options must not have a default."""
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
        )
        self._opts[cfg_key] = opt

    def update(self, other: "runopts") -> None:
        self._opts.update(other._opts)

    # pyre-fixme[15]: Inconsistent override - __or__ returns runopts, not UnionType
    def __or__(self, other: "runopts") -> "runopts":
        """Merge two runopts, returning a new runopts.

        Example:
            .. doctest::

                >>> opts1 = runopts()
                >>> opts1.add("foo", type_=str, default="a", help="foo option")
                >>> opts2 = runopts()
                >>> opts2.add("bar", type_=int, default=1, help="bar option")
                >>> merged = opts1 | opts2
                >>> sorted(k for k, _ in merged)
                ['bar', 'foo']
        """
        merged = runopts()
        merged.update(self)
        merged.update(other)
        return merged

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
    """Raised when run cfg is missing required options or has type mismatches."""

    def __init__(
        self, invalid_reason: str, cfg_key: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        given = str(cfg) if cfg else "<EMPTY>"
        super().__init__(f"{invalid_reason}. Given: {given}")
        self.cfg_key = cfg_key


class MalformedAppHandleException(Exception):
    """Raised when an :py:data:`AppHandle` is not a valid URI."""

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
    """Parsed components of an :py:data:`AppHandle`."""

    scheduler_backend: str
    session_name: str
    app_id: str


class UnknownAppException(Exception):
    """Raised when the application does not exist or has been purged."""

    def __init__(self, app_handle: "AppHandle") -> None:
        super().__init__(
            f"Unknown app = {app_handle}. Did you forget to call session.run()?"
            f" Otherwise, the app may have already finished and purged by the scheduler"
        )


def parse_app_handle(app_handle: AppHandle) -> ParsedAppHandle:
    """Parses ``{scheduler}://{session_name}/{app_id}`` into its components.

    .. doctest::

        >>> from torchx.specs import parse_app_handle
        >>> parse_app_handle("k8s://default/foo_bar")
        ParsedAppHandle(scheduler_backend='k8s', session_name='default', app_id='foo_bar')
        >>> parse_app_handle("k8s:///foo_bar")
        ParsedAppHandle(scheduler_backend='k8s', session_name='', app_id='foo_bar')

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
