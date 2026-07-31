# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
from abc import ABC, abstractmethod

import torchx.components.dist as dist_components
import torchx.components.serve as serve_components
import torchx.components.utils as utils_components
from torchx.specs import AppDef


class ComponentProvider(ABC):
    """
    Abstract class that represents generic component provider.
    """

    def __init__(self, scheduler: str, image: str) -> None:
        self._scheduler = scheduler
        self._image = image

    @abstractmethod
    def get_app_def(self) -> AppDef:
        pass

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass


class DDPComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return dist_components.ddp(
            m="torchx.examples.apps.compute_world_size.main",
            name="ddp-trainer",
            image=self._image,
            cpu=1,
            j="2x2",
            max_retries=3,
            env={
                "LOGLEVEL": "INFO",
            },
        )


class ServeComponentProvider(ComponentProvider):
    # TODO(aivanou): Remove dryrun and test e2e serve component+app
    def get_app_def(self) -> AppDef:
        return serve_components.torchserve(
            model_path="dummy_path",
            management_api="dummy_api",
            image=self._image,
            dryrun=True,
        )


class BoothComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return utils_components.booth(
            x1=1.0,
            x2=2.0,
            image=self._image,
        )


class ShComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return utils_components.sh(
            *["echo", "test"],
            image=self._image,
        )


class TouchComponentProvider(ComponentProvider):
    def __init__(self, image: str, scheduler: str) -> None:
        super(TouchComponentProvider, self).__init__(image, scheduler)
        self._file_path = "<None>"

    def setUp(self) -> None:
        fname = "torchx_touch_test.txt"
        if self._scheduler == "local_cwd":
            self._file_path: str = os.path.join(tempfile.gettempdir(), fname)
        else:
            self._file_path: str = fname

    def get_app_def(self) -> AppDef:
        return utils_components.touch(
            file=self._file_path,
            image=self._image,
        )

    def tearDown(self) -> None:
        if os.path.exists(self._file_path):
            os.remove(self._file_path)


class CopyComponentProvider(ComponentProvider):
    def __init__(self, image: str, scheduler: str) -> None:
        super(CopyComponentProvider, self).__init__(image, scheduler)
        self._src_path = "<None>"
        self._dst_path = "<None>"

    def setUp(self) -> None:
        if self._scheduler in ["local_cwd"]:
            fname = "torchx_copy_test.txt"
            self._src_path: str = os.path.join(tempfile.gettempdir(), fname)
            self._dst_path: str = os.path.join(tempfile.gettempdir(), f"{fname}.copy")
            self._process_local_sched()
        else:
            self._src_path: str = "README.md"
            self._dst_path: str = "README.md.copy"

    def _process_local_sched(self) -> None:
        if not os.path.exists(self._src_path):
            with open(self._src_path, "w") as f:
                f.write("test data")

    def tearDown(self) -> None:
        if os.path.exists(self._dst_path):
            os.remove(self._dst_path)
        if self._scheduler in ["local_cwd"] and os.path.exists(self._dst_path):
            os.remove(self._dst_path)

    def get_app_def(self) -> AppDef:
        return utils_components.copy(
            src=self._src_path, dst=self._dst_path, image=self._image
        )
