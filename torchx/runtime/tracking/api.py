#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import json

import fsspec


KeyType = int | str
ResultType = int | float | str


class ResultTracker(abc.ABC):
    """
    Base result tracker, which should be sub-classed to implement trackers.
    Typically there exists a tracker implementation per backing store.

    Usage:

    .. code-block:: python

     # get and put APIs can be used directly or in map-like API
     # the following are equivalent
     tracker.put("foo", l2norm=1.2)
     tracker["foo"] = {"l2norm": 1.2}

     # so are these
     tracker.get("foo")["l2norm"] == 1.2
     tracker["foo"]["l2norm"] == 1.2

    Valid ``result`` types are:

    1. numeric: int, float
    2. literal:str (1kb size limit when utf-8 encoded)

    Valid ``key`` types are:

    1. ``int``
    2. ``str``

    As a convention, "slashes" can be used in the key to store results that
    are statistical. For instance, to store the mean and sem of l2norm:

    .. code-block:: python

     tracker[key] = {"l2norm/mean" : 1.2, "l2norm/sem": 3.4}
     tracker[key]["l2norm/mean"] # returns 1.2
     tracker[key]["l2norm/sem"] # returns 3.4

    Keys are assumed to be unique within the scope of the tracker's backing
    store. For example, if a tracker is backed by a local directory and the
    ``key`` is the file within directory where the results are saved, then

    .. code-block:: python

      # same key, different backing directory -> results are not overwritten
      FsspecResultTracker("/tmp/foo")["1"] = {"l2norm":1.2}
      FsspecResultTracker("/tmp/bar")["1"] = {"l2norm":3.4}

    The tracker is NOT a central entity hence no strong consistency guarantees
    (beyond what the backing store provides) are made between ``put`` and ``get``
    operations on the same key. Similarly no strong consistency guarantees are
    made between two consecutive ``put`` or ``get`` operations on the same key.

    For example:

    .. code-block:: python

      tracker[1] = {"l2norm":1.2}
      tracker[1] = {"l2norm":3.4}
      tracker[1] # NOT GUARANTEED TO BE 3.4!

      sleep(1*MIN)
      tracker[1] # more likely to be 3.4 but still not guaranteed!

    It is STRONGLY advised that a unique id is used as the key. This id is
    often the job id for simple jobs or can be a concatenation of
    (experiment_id, trial_number) or (job id, replica/worker rank)
    for iterative applications like hyper-parameter optimization.

    """

    def __getitem__(self, key: KeyType) -> dict[str, ResultType]:
        return self.get(key)

    def __setitem__(self, key: KeyType, results: dict[str, ResultType]) -> None:
        self.put(key, **results)

    @abc.abstractmethod
    def put(self, key: KeyType, **results: ResultType) -> None:
        """
        Records the given results by associating them with the provided key.
        The key is implicitly converted to a string by calling ``str(key)``.

        Calling this API on the same key multiple times overwrites
        the results BUT not necessarily with the last call's results.
        The exact semantics of consistency depends on the backing store.

        .. note:: It is recommended this API is only called once per unique key

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get(self, key: KeyType) -> dict[str, ResultType]:
        """
        Returns the results that have been recorded (put) with the key or
        an empty map if no such key exists.
        The key is implicitly converted to a string by calling ``str(key)``.

        Note that if the backing store is not strongly consistent, there may be
        a delay in the presence of the key after the ``put`` API has been called.
        In this case, this method DOES NOT block until the key becomes available.
        To account for this, the caller may chose to retry-get-with-timeout.
        """
        raise NotImplementedError()


class FsspecResultTracker(ResultTracker):
    """
    Tracker that uses fsspec under the hood to save results.

    Usage:

    .. testcode:: [tracking_fsspec_result_tracker]

     from torchx.runtime.tracking import FsspecResultTracker

     # PUT: in trainer.py
     tracker_base = "/tmp/foobar" # also supports URIs (e.g. "s3://bucket/trainer/123")
     tracker = FsspecResultTracker(tracker_base)
     tracker["attempt_1/out"] = {"accuracy": 0.233}

     # GET: anywhere outside trainer.py
     tracker = FsspecResultTracker(tracker_base)
     print(tracker["attempt_1/out"]["accuracy"])

    .. testoutput:: [tracking_fsspec_result_tracker]

      0.233

    """

    def __init__(self, tracker_base: str) -> None:
        self._tracker_base = tracker_base

    def put(self, key: KeyType, **results: ResultType) -> None:
        mapper = fsspec.get_mapper(self._tracker_base, create=True)
        # save results in pretty-print format so that the file is human readable
        mapper[key] = json.dumps(results, indent=2).encode("utf-8")

    def get(self, key: KeyType) -> dict[str, ResultType]:
        mapper = fsspec.get_mapper(self._tracker_base)
        try:
            results = mapper[key]
            return json.loads(results.decode("utf-8"))
        except KeyError:
            return {}
