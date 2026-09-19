# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import abc
import fnmatch
import logging
import posixpath
import tempfile
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Iterable, Mapping, TypeVar

from torchx.specs import AppDef, CfgVal, Role, Workspace, runopts

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
            to ``role.env``. ``role.workspace`` is left set, so a role that has
            been through this method is not distinguishable by inspection from
            one that has not -- compare ``role.image`` instead.

        Called by :py:meth:`~torchx.runner.api.Runner.dryrun` with the roles of
        its private copy, so the caller's :py:class:`~torchx.specs.AppDef` is
        unaffected there; :py:meth:`~torchx.schedulers.api.Scheduler.submit`
        passes the caller's own roles.
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
        with the return value before scheduling. Leave this method
        unoverridden when the builder has no push step (its build already
        put the artifact where the job reads it); :py:class:`MultiWorkspaceMixin`
        then pushes nothing for it.
        """
        raise NotImplementedError("dryrun_push is not implemented")

    def push_images(self, images_to_push: T) -> None:
        """Pushes images (returned by :py:meth:`dryrun_push_images`) to the remote repo."""
        raise NotImplementedError("push is not implemented")


def _has_push_step(builder: WorkspaceMixin[Any]) -> bool:
    # A builder that leaves the base ``dryrun_push_images`` in place has no
    # separate push step: its build already put the artifact where the job
    # reads it (a shared directory, say).
    return type(builder).dryrun_push_images is not WorkspaceMixin.dryrun_push_images


class MultiWorkspaceMixin(WorkspaceMixin[tuple[str, Any]]):
    """Scheduler mix-in that offers several workspace builders; the
    ``workspace_type`` run option picks one.

    A scheduler whose backend accepts more than one kind of image (a shared
    directory and a container image, say) returns one builder per kind from
    :py:meth:`workspace_builders`. Every :py:class:`WorkspaceMixin` call is
    forwarded to the builder the run config selects, and
    :py:meth:`workspace_opts` is the union of the builders' options plus the
    selector, so ``torchx runopts <scheduler>`` lists them all.

    .. doctest::

        >>> from typing import Any, Mapping
        >>> from torchx.workspace import MultiWorkspaceMixin, WorkspaceMixin
        >>> from torchx.workspace.dir_workspace import DirWorkspaceMixin, TmpDirWorkspaceMixin
        >>> class TwoWaysMixin(MultiWorkspaceMixin):
        ...     def __init__(self) -> None:
        ...         self.builders = {"dir": DirWorkspaceMixin(), "tmpdir": TmpDirWorkspaceMixin()}
        ...     def workspace_builders(self) -> Mapping[str, WorkspaceMixin[Any]]:
        ...         return self.builders
        >>> mixin = TwoWaysMixin()
        >>> mixin.workspace_builder({})[0]              # the first builder is the default
        'dir'
        >>> mixin.workspace_builder({"workspace_type": "tmpdir"})[0]
        'tmpdir'
        >>> sorted(mixin.workspace_opts()._opts)
        ['workspace_type']

    The first builder is the default, so adding this mix-in to a scheduler
    that used one builder changes nothing until a user sets
    ``workspace_type``.
    """

    #: the run option that picks the builder
    WORKSPACE_TYPE_OPT: str = "workspace_type"

    @abc.abstractmethod
    def workspace_builders(self) -> Mapping[str, WorkspaceMixin[Any]]:
        """Returns ``{name: builder}``. The first entry is the default.

        Return the same builder instances on every call (build the mapping
        once, in ``__init__``): the push step looks the builder up again by
        name and expects the instance that ran the build.
        """
        ...

    def workspace_opts(self) -> runopts:
        """The selector plus the union of every builder's options.

        Two builders declaring the same option name must mean the same thing
        by it; the later builder's declaration wins.
        """
        builders = self.workspace_builders()
        if not builders:
            raise ValueError(
                f"{type(self).__name__}.workspace_builders() returned no builders"
            )
        names = list(builders)
        opts = runopts()
        opts.add(
            self.WORKSPACE_TYPE_OPT,
            type_=str,
            default=names[0],
            help=f"which workspace builder patches the image: one of {', '.join(names)}",
        )
        for builder in builders.values():
            opts.update(builder.workspace_opts())
        return opts

    def workspace_builder(
        self, cfg: Mapping[str, CfgVal]
    ) -> tuple[str, WorkspaceMixin[Any]]:
        """Returns ``(name, builder)`` for ``cfg["workspace_type"]``, or the
        default builder when the option is unset.

        Raises:
            ValueError: the requested type is not one this scheduler offers;
                the message lists the names it does offer.
        """
        builders = self.workspace_builders()
        name = cfg.get(self.WORKSPACE_TYPE_OPT)
        if name is None:
            name = next(iter(builders))
        if not isinstance(name, str) or name not in builders:
            raise ValueError(
                f"unknown {self.WORKSPACE_TYPE_OPT} `{name}`;"
                f" this scheduler offers: {', '.join(builders)}"
            )
        return name, builders[name]

    def caching_build_workspace_and_update_role(
        self,
        role: Role,
        cfg: Mapping[str, CfgVal],
        build_cache: dict[object, object],
    ) -> None:
        """Builds *role*'s workspace with the selected builder."""
        _, builder = self.workspace_builder(cfg)
        builder.caching_build_workspace_and_update_role(role, cfg, build_cache)

    def dryrun_push_images(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> tuple[str, Any]:
        """Runs the selected builder's push dry-run.

        Returns ``(name, images)`` so :py:meth:`push_images` can find the
        same builder again; *images* is ``None`` for a builder with no push
        step, and :py:meth:`push_images` then pushes nothing.
        """
        name, builder = self.workspace_builder(cfg)
        if not _has_push_step(builder):
            return name, None
        return name, builder.dryrun_push_images(app, cfg)

    def push_images(self, images_to_push: tuple[str, Any]) -> None:
        """Pushes with the builder named by :py:meth:`dryrun_push_images`."""
        name, images = images_to_push
        if images is None:
            return
        self.workspace_builders()[name].push_images(images)


def pin_workspace_images(
    app: AppDef, images: Mapping[tuple[str, Workspace], str]
) -> None:
    """Points each role at its already-built image so submitting *app* skips the rebuild.

    Pairs with :py:meth:`~torchx.runner.Runner.build_workspace`, whose return
    value *images* is. Mutates *app*: a pinned role gets ``image`` set and
    ``workspace`` cleared, which is what makes the rebuild a no-op.

    A role matches on its whole ``(image, workspace)`` pair, since that is what
    the build ran on: one sharing only the workspace is a different build and is
    left alone, as is a role with no workspace (it arrived with a pre-built image
    that must not be clobbered) or one whose pair nobody built.
    """
    for role in app.roles:
        if not role.workspace:
            continue
        image = images.get((role.image, role.workspace))
        if image is not None:
            role.image = image
            role.workspace = None


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
