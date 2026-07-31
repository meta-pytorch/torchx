# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import fnmatch
import logging
import posixpath
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Mapping, TYPE_CHECKING, TypeVar

from torchx.specs import AppDef, CfgVal, Role, runopts, Workspace

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

TORCHX_IGNORE = ".torchxignore"

T = TypeVar("T")

PackageType = TypeVar("PackageType")
WorkspaceConfigType = TypeVar("WorkspaceConfigType")


@dataclass
class PkgInfo(Generic[PackageType]):
    """
    .. deprecated::
        Will be removed in a future release. Fork if your project depends on it.

    Metadata for a built workspace package.
    """

    img: str
    lazy_overrides: dict[str, Any]
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
        """Builds the workspace, producing either a new image or an incremental patch."""
        pass


class WorkspaceMixin(abc.ABC, Generic[T]):
    """Scheduler mix-in that auto-builds a local workspace into a deployable image or patch.

    .. warning::
        Prototype -- this interface may change without notice.

    Attach to a :py:class:`~torchx.schedulers.api.Scheduler` so that local code
    changes in the workspace are automatically reflected at runtime (via a rebuilt
    image or an overlaid diff patch) without a manual image rebuild.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

    def workspace_opts(self) -> runopts:
        """Returns the :py:class:`~torchx.specs.api.runopts` accepted by this workspace."""
        return runopts()

    def build_workspaces(self, roles: list[Role], cfg: Mapping[str, CfgVal]) -> None:
        """Builds workspaces for each role and updates ``role.image`` in-place.

        .. important::
            Mutates the passed *roles*. May also add env vars (e.g. ``WORKSPACE_DIR``)
            to ``role.env``.
        """

        build_cache: dict[object, object] = {}

        for i, role in enumerate(roles):
            if role.workspace:
                old_img = role.image
                self.caching_build_workspace_and_update_role(role, cfg, build_cache)

                if old_img != role.image:
                    logger.info(
                        "role[%d]=%s updated with new image to include workspace changes",
                        i,
                        role.name,
                    )

    def caching_build_workspace_and_update_role(
        self,
        role: Role,
        cfg: Mapping[str, CfgVal],
        build_cache: dict[object, object],
    ) -> None:
        """Like :py:meth:`build_workspace_and_update_role` but with a per-call *build_cache*.

        Subclasses should implement this method instead of
        :py:meth:`build_workspace_and_update_role`. The cache avoids redundant
        builds when multiple roles share the same image and workspace.

        .. important::
            *build_cache* lifetime is scoped to a single
            :py:meth:`build_workspaces` call. What gets cached is up to the
            implementation.

        The default implementation delegates to the (deprecated)
        :py:meth:`build_workspace_and_update_role`, merging multi-dir
        workspaces into a single tmpdir first.
        """

        workspace = role.workspace

        if not workspace:
            return

        if workspace.is_unmapped_single_project():
            # single-dir workspace with no target map; no need to copy to a tmp dir
            self.build_workspace_and_update_role(role, str(workspace), cfg)
        else:
            # multi-dirs or single-dir with a target map;
            # copy all dirs to a tmp dir and treat the tmp dir as a single-dir workspace
            with tempfile.TemporaryDirectory(suffix="torchx_workspace_") as outdir:
                workspace.merge_into(outdir)
                self.build_workspace_and_update_role(role, outdir, cfg)

    def build_workspace_and_update_role(
        self,
        role: Role,
        workspace: str,
        cfg: Mapping[str, CfgVal],
    ) -> None:
        """Build *workspace* and mutate *role* to reference the resulting artifact.

        .. deprecated::
            Implement :py:meth:`caching_build_workspace_and_update_role` instead.
        """
        raise NotImplementedError("implement `caching_build_workspace_and_update_role`")

    def dryrun_push_images(self, app: AppDef, cfg: Mapping[str, CfgVal]) -> T:
        """Dry-run the image push: updates *app* with final image names.

        Only called for remote jobs. :py:meth:`push_images` must be called
        with the return value before scheduling.
        """
        raise NotImplementedError("dryrun_push is not implemented")

    def push_images(self, images_to_push: T) -> None:
        """Pushes images (returned by :py:meth:`dryrun_push_images`) to the remote repo."""
        raise NotImplementedError("push is not implemented")


def _ignore(s: str, patterns: Iterable[str]) -> tuple[int, bool]:
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
) -> Iterable[tuple[str, Iterable[str], Mapping[str, Mapping[str, object]]]]:
    """Walks *path* on *fs*, filtering entries via ``.dockerignore``-style rules
    read from *ignore_name*.
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
