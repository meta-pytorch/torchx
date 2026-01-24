# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import logging
import posixpath
import stat
import sys
import tarfile
import tempfile
from typing import IO, Iterable, Mapping, TextIO, TYPE_CHECKING

import fsspec
import torchx
from docker.errors import BuildError
from torchx.specs import AppDef, CfgVal, Role, runopts
from torchx.workspace.api import walk_workspace, WorkspaceMixin

if TYPE_CHECKING:
    from docker import DockerClient

log: logging.Logger = logging.getLogger(__name__)


TORCHX_DOCKERFILE = "Dockerfile.torchx"

DEFAULT_DOCKERFILE = b"""
ARG IMAGE
FROM $IMAGE

COPY . .
"""


class DockerWorkspaceMixin(WorkspaceMixin[dict[str, tuple[str, str]]]):
    """
    DockerWorkspaceMixin will build patched docker images from the workspace. These
    patched images are docker images and can be either used locally via the
    docker daemon or pushed using the helper methods to a remote repository for
    remote jobs.

    This requires a running docker daemon locally and for remote pushing
    requires being authenticated to those repositories via ``docker login``.

    If there is a ``Dockerfile.torchx`` file present in the workspace that will
    be used instead to build the container.

    The docker build is provided with some extra build arguments that can be
    used in the Dockerfile.torchx:

    * IMAGE: the image string from the first Role in the AppDef
    * WORKSPACE: the full workspace path

    To exclude files from the build context you can use the standard
    `.dockerignore` file.

    See more:

    * https://docs.docker.com/engine/reference/commandline/login/
    * https://docs.docker.com/get-docker/
    """

    LABEL_VERSION: str = "torchx.pytorch.org/version"

    def __init__(
        self,
        *args: object,
        docker_client: "DockerClient | None" = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__docker_client = docker_client

    @property
    def _docker_client(self) -> "DockerClient":
        client = self.__docker_client
        if client is None:
            import docker

            client = docker.from_env()
            self.__docker_client = client
        return client

    def workspace_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "image_repo",
            type_=str,
            help="(remote jobs) the image repository to use when pushing patched images, must have push access. Ex: example.com/your/container",
        )
        opts.add(
            "quiet",
            type_=bool,
            default=False,
            help="whether to suppress verbose output for image building. Defaults to ``False``.",
        )
        return opts

    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """
        Builds a new docker image using the ``role``'s image as the base image
        and updates the ``role``'s image with this newly built docker image id

        Args:
            role: the role whose image (a Docker image) is to be used as the base image
            workspace: a fsspec path to a directory with contents to be overlaid
        """

        old_imgs = [
            image.id
            for image in self._docker_client.images.list(name=cfg["image_repo"])
        ]
        context = _build_context(role.image, workspace)

        try:
            try:
                self._docker_client.images.pull(role.image)
            except Exception as e:
                log.warning(
                    f"failed to pull image {role.image}, falling back to local: {e}"
                )
            log.info("Building workspace docker image (this may take a while)...")
            build_events = self._docker_client.api.build(
                fileobj=context,
                custom_context=True,
                dockerfile=TORCHX_DOCKERFILE,
                buildargs={
                    "IMAGE": role.image,
                    "WORKSPACE": workspace,
                },
                pull=False,
                rm=True,
                decode=True,
                labels={
                    self.LABEL_VERSION: torchx.__version__,
                },
            )
            image_id = None
            for event in build_events:
                if message := event.get("stream"):
                    if not cfg.get("quiet", False):
                        message = message.strip("\r\n").strip("\n")
                        if message:
                            log.info(message)
                if aux := event.get("aux"):
                    image_id = aux["ID"]
                if error := event.get("error"):
                    raise BuildError(reason=error, build_log=None)
            if len(old_imgs) == 0 or role.image not in old_imgs:
                assert image_id, "image id was not found"
                role.image = image_id

        finally:
            context.close()

    def dryrun_push_images(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> dict[str, tuple[str, str]]:
        """
        _update_app_images replaces the local Docker images (identified via
        ``sha256:...``) in the provided ``AppDef`` with the remote path that they will be uploaded to and
        returns a mapping of local to remote names.

        ``push`` must be called with the returned mapping before
        launching the job.

        Returns:
            A dict of [local image name, (remote repo, tag)].
        """
        HASH_PREFIX = "sha256:"
        image_repo = cfg.get("image_repo")

        images_to_push = {}
        for role in app.roles:
            if role.image.startswith(HASH_PREFIX):
                if not image_repo:
                    raise KeyError(
                        f"must specify the image repository via `image_repo` config to be able to upload local image {role.image}"
                    )
                assert isinstance(image_repo, str), "image_repo must be str"

                image_hash = role.image[len(HASH_PREFIX) :]
                remote_image = image_repo + ":" + image_hash
                images_to_push[role.image] = (
                    image_repo,
                    image_hash,
                )
                role.image = remote_image
        return images_to_push

    def push_images(self, images_to_push: dict[str, tuple[str, str]]) -> None:
        """
        _push_images pushes the specified images to the remote container
        repository with the specified tag. The docker daemon must be
        authenticated to the remote repository using ``docker login``.

        Args:
            images_to_push: A dict of [local image name, (remote repo, tag)].
        """

        if len(images_to_push) == 0:
            return

        client = self._docker_client
        for local, (repo, tag) in images_to_push.items():
            log.info(f"pushing image {repo}:{tag}...")
            img = client.images.get(local)
            img.tag(repo, tag=tag)
            print_push_events(
                client.images.push(repo, tag=tag, stream=True, decode=True)
            )


def print_push_events(
    events: Iterable[dict[str, str]],
    stream: TextIO = sys.stderr,
) -> None:
    ID_KEY = "id"
    ERROR_KEY = "error"
    STATUS_KEY = "status"
    PROG_KEY = "progress"
    LINE_CLEAR = "\033[2K"
    BLUE = "\033[34m"
    ENDC = "\033[0m"
    HEADER = f"{BLUE}docker push {ENDC}"

    def lines_up(lines: int) -> str:
        return f"\033[{lines}F"

    def lines_down(lines: int) -> str:
        return f"\033[{lines}E"

    ids = []
    for event in events:
        if ERROR_KEY in event:
            raise RuntimeError(f"failed to push docker image: {event[ERROR_KEY]}")

        id = event.get(ID_KEY)
        status = event.get(STATUS_KEY)

        if not status:
            continue

        if id:
            msg = f"{HEADER}{id}: {status} {event.get(PROG_KEY, '')}"
            if id not in ids:
                ids.append(id)
                stream.write(f"{msg}\n")
            else:
                lineno = len(ids) - ids.index(id)
                stream.write(f"{lines_up(lineno)}{LINE_CLEAR}{msg}{lines_down(lineno)}")
        else:
            stream.write(f"{HEADER}{status}\n")


def _build_context(img: str, workspace: str) -> IO[bytes]:
    # f is closed by parent, NamedTemporaryFile auto closes on GC
    f = tempfile.NamedTemporaryFile(  # noqa P201
        prefix="torchx-context",
        suffix=".tar",
    )

    with tarfile.open(fileobj=f, mode="w") as tf:
        _copy_to_tarfile(workspace, tf)
        if TORCHX_DOCKERFILE not in tf.getnames():
            info = tarfile.TarInfo(TORCHX_DOCKERFILE)
            info.size = len(DEFAULT_DOCKERFILE)
            tf.addfile(info, io.BytesIO(DEFAULT_DOCKERFILE))
    f.seek(0)
    return f


def _copy_to_tarfile(workspace: str, tf: tarfile.TarFile) -> None:
    fs, path = fsspec.core.url_to_fs(workspace)
    log.info(f"Workspace `{workspace}` resolved to filesystem path `{path}`")
    assert isinstance(path, str), "path must be str"

    for dir, dirs, files in walk_workspace(fs, path, ".dockerignore"):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        for file, info in files.items():
            with fs.open(info["name"], "rb") as f:
                filepath = posixpath.join(relpath, file) if relpath != "." else file
                tinfo = tarfile.TarInfo(filepath)
                size = info["size"]
                assert isinstance(size, int), "size must be an int"
                tinfo.size = size

                # preserve unix mode for supported filesystems; fsspec.filesystem("memory") for example does not support
                # unix file mode, hence conditional check here
                if "mode" in info:
                    mode = info["mode"]
                    assert isinstance(mode, int), "mode must be an int"
                    tinfo.mode = stat.S_IMODE(mode)

                tf.addfile(tinfo, f)
