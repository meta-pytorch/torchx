#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Optional, Union


class SourceType(str, Enum):
    UNKNOWN = "<unknown>"
    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"


@dataclass
class TorchxEvent:
    """
    The class represents the event produced by ``torchx.runner`` api calls.

    Arguments:
        session: Session id of the current run
        scheduler: Scheduler that is used to execute request
        api: Api name
        app_id: Unique id that is set by the underlying scheduler
        app_image: Image/container bundle that is used to execute request.
        app_metadata: metadata to the app (treatment of metadata is scheduler dependent)
        runcfg: Run config that was used to schedule app.
        source: Type of source the event is generated.
        cpu_time_usec: CPU time spent in usec
        wall_time_usec: Wall time spent in usec
        start_epoch_time_usec: Epoch time in usec when runner event starts
        Workspace: Track how different workspaces/no workspace affects build and scheduler
    """

    session: str
    scheduler: str
    api: str
    app_id: Optional[str] = None
    app_image: Optional[str] = None
    app_metadata: Optional[Dict[str, str]] = None
    runcfg: Optional[str] = None
    raw_exception: Optional[str] = None
    source: SourceType = SourceType.UNKNOWN
    cpu_time_usec: Optional[int] = None
    wall_time_usec: Optional[int] = None
    start_epoch_time_usec: Optional[int] = None
    workspace: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_source_location: Optional[str] = None

    def __str__(self) -> str:
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "TorchxEvent"]) -> "TorchxEvent":
        if isinstance(data, TorchxEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
            if "source" in data_dict:
                # Convert string to enum
                try:
                    data_dict["source"] = SourceType(data_dict["source"])
                except ValueError:
                    data_dict.pop("source", None)

        # pyre-fixme[61]: `data_dict` may not be initialized here.
        return TorchxEvent(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
