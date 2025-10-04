# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import fnmatch
import posixpath
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from torchx.specs import AppDef, CfgVal, Role, runopts, Workspace

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

TORCHX_IGNORE = ".torchxignore"

T = TypeVar("T")

PackageType = TypeVar("PackageType")
WorkspaceConfigType = TypeVar("WorkspaceConfigType")


@dataclass
class PkgInfo(Generic[PackageType]):
    """
    Convenience class used to specify information regarding the built workspace
    """

    img: str
    lazy_overrides: Dict[str, Any]
    metadata: PackageType

    def __post_init__(self) -> None:
        msg = (
            f"{self.__class__.__name__} is deprecated and will be removed in the future."
            " Consider forking this class if your project depends on it."
        )
        warnings.warn(
            msg,
            FutureWarning,
            stacklevel=2,
        )


@dataclass
class WorkspaceBuilder(Generic[PackageType, WorkspaceConfigType]):
    cfg: WorkspaceConfigType

    def __post_init__(self) -> None:
        msg = (
            f"{self.__class__.__name__} is deprecated and will be removed in the future."
            " Consider forking this class if your project depends on it."
        )
        warnings.warn(
            msg,
            FutureWarning,
            stacklevel=2,
        )

    @abc.abstractmethod
    def build_workspace(self, sync: bool = True) -> PkgInfo[PackageType]:
        """
        Builds the specified ``workspace`` with respect to ``img``.
        In the simplest case, this method builds a new image.
        Certain (more efficient) implementations build
        incremental diff patches that overlay on top of the role's image.

        """
        pass


class WorkspaceMixin(abc.ABC, Generic[T]):
    """
    Note: (Prototype) this interface may change without notice!

    A mix-in that can be attached to a Scheduler that adds the ability to
    builds a workspace. A workspace is the local checkout of the codebase/project
    that builds into an image. The workspace scheduler adds capability to
    automatically rebuild images or generate diff patches that are
    applied to the ``Role``, allowing the user to make local code
    changes to the application and having those changes be reflected
    (either through a new image or an overlaid patch) at runtime
    without a manual image rebuild. The exact semantics of what the
    workspace build artifact is, is implementation dependent.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

    def workspace_opts(self) -> runopts:
        """
        Returns the run configuration options expected by the workspace.
        Basically a ``--help`` for the ``run`` API.
        """
        return runopts()

    def build_workspace_and_update_role2(
        self,
        role: Role,
        workspace: Union[Workspace, str],
        cfg: Mapping[str, CfgVal],
    ) -> None:
        """
        Same as :py:meth:`build_workspace_and_update_role` but operates
        on :py:class:`Workspace` (supports multi-project workspaces)
        as well as ``str`` (for backwards compatibility).

        If ``workspace`` is a ``str`` this method simply calls
        :py:meth:`build_workspace_and_update_role`.

        If ``workspace`` is :py:class:`Workspace` then the default
        impl copies all the projects into a tmp directory and passes the tmp dir to
        :py:meth:`build_workspace_and_update_role`

        Subclasses can override this method to customize multi-project
        workspace building logic.
        """
        if isinstance(workspace, Workspace):
            if not workspace.is_unmapped_single_project():
                with tempfile.TemporaryDirectory(suffix="torchx_workspace_") as outdir:
                    for src, dst in workspace.projects.items():
                        dst_path = Path(outdir) / dst
                        if Path(src).is_file():
                            shutil.copy2(src, dst_path)
                        else:  # src is dir
                            shutil.copytree(src, dst_path, dirs_exist_ok=True)

                    self.build_workspace_and_update_role(role, outdir, cfg)
                    return
            else:  # single project workspace with no target mapping (treat like a str workspace)
                workspace = str(workspace)

        self.build_workspace_and_update_role(role, workspace, cfg)

    @abc.abstractmethod
    def build_workspace_and_update_role(
        self,
        role: Role,
        workspace: str,
        cfg: Mapping[str, CfgVal],
    ) -> None:
        """
        Builds the specified ``workspace`` with respect to ``img``
        and updates the ``role`` to reflect the built workspace artifacts.
        In the simplest case, this method builds a new image and updates
        the role's image. Certain (more efficient) implementations build
        incremental diff patches that overlay on top of the role's image.

        Note: this method mutates the passed ``role``.
        """
        ...

    def dryrun_push_images(self, app: AppDef, cfg: Mapping[str, CfgVal]) -> T:
        """
        dryrun_push does a dryrun of the image push and updates the app to have
        the final values. Only called for remote jobs.

        ``push`` must be called before scheduling the job.
        """
        raise NotImplementedError("dryrun_push is not implemented")

    def push_images(self, images_to_push: T) -> None:
        """
        push pushes any images to the remote repo if required.
        """
        raise NotImplementedError("push is not implemented")


def _ignore(s: str, patterns: Iterable[str]) -> Tuple[int, bool]:
    last_matching_pattern = -1
    match = False
    if s in (".", "Dockerfile.torchx"):
        return last_matching_pattern, match
    s = posixpath.normpath(s)
    for i, pattern in enumerate(patterns):
        if pattern.startswith("!") and fnmatch.fnmatch(s, pattern[1:]):
            match = False
            last_matching_pattern = i
        elif fnmatch.fnmatch(s, pattern):
            match = True
            last_matching_pattern = i
    return last_matching_pattern, match


def walk_workspace(
    fs: "AbstractFileSystem",
    path: str,
    ignore_name: str = TORCHX_IGNORE,
) -> Iterable[Tuple[str, Iterable[str], Mapping[str, Mapping[str, object]]]]:
    """
    walk_workspace walks the filesystem path and applies the ignore rules
    specified via ``ignore_name``.
    This follows the rules for ``.dockerignore``.
    """
    ignore_patterns = []
    ignore_path = posixpath.join(path, ignore_name)
    if fs.exists(ignore_path):
        with fs.open(ignore_path, "rt") as f:
            lines = f.readlines()
        for line in lines:
            line, _, _ = line.partition("#")
            line = line.strip()
            if len(line) == 0 or line == ".":
                continue
            ignore_patterns.append(line)

    paths_to_walk = [(0, path)]
    while paths_to_walk:
        first_pattern_to_use, current_path = paths_to_walk.pop()
        for dir, dirs, files in fs.walk(current_path, detail=True, maxdepth=1):
            assert isinstance(dir, str), "path must be str"
            relpath = posixpath.relpath(dir, path)

            if _ignore(relpath, ignore_patterns[first_pattern_to_use:])[1]:
                continue
            filtered_dirs = []
            last_matching_pattern_index = []
            for d in dirs:
                index, match = _ignore(
                    posixpath.join(relpath, d), ignore_patterns[first_pattern_to_use:]
                )
                if not match:
                    filtered_dirs.append(d)
                    last_matching_pattern_index.append(first_pattern_to_use + index)
            dirs = filtered_dirs
            files = {
                file: info
                for file, info in files.items()
                if not _ignore(
                    posixpath.join(relpath, file) if relpath != "." else file,
                    ignore_patterns[first_pattern_to_use:],
                )[1]
            }
            yield dir, dirs, files
            for i, d in zip(last_matching_pattern_index, dirs):
                paths_to_walk.append((i + 1, posixpath.join(dir, d)))
