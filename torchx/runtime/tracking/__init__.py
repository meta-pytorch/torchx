#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
.. note:: EXPERIMENTAL, USE AT YOUR OWN RISK, APIs SUBJECT TO CHANGE

In TorchX applications are binaries (executables),
hence there is no built-in way to "return" results from applications.
The ``torchx.runtime.tracking`` module allows applications
to return simple results (note the keyword "simple"). The return types
that the tracker module supports are intentionally constrained. For instance,
attempting to return the trained model weights, which can be hundreds of GB in size,
is not allowed. This module is NOT intended nor tuned to pass around large
quantities of data or binary blobs.

When apps are launched as part of a higher level coordinated effort
(e.g. workflow, pipeline, hyper-parameter optimization) often times,
the result of the app needs to be accessible to the coordinator or
other apps in the workflow.

Suppose App1 and App2 are launched sequentially with the output of App1
feeding as input of App2. Since these are binaries the typical way to
chain input/outputs between apps is by passing the output file path
of App1 as the input file path of App2:

.. code-block:: shell-session

 $ app1 --output-file s3://foo/out/app1.out
 $ app2 --input-file s3://foo/out/app1.out

As easy as this may seem, there are a few things one needs to worry about:

1. The format of the file ``app1.out`` (app1 needs to write it in the format app2 understands)
2. Actually parsing the url and writing/reading the output file

So the application's main ends up looking like this (pseudo-code for demonstrative purposes):

.. code-block:: python

 # in app1.py
 if __name__ == "__main__":
    accuracy = do_something()
    s3client = ...
    out = {"accuracy": accuracy}

    with open("/tmp/out", "w") as f:
        f = json.dumps(out).encode("utf-8")

    s3client.put(args.output_file, f)

 # in app2.py
 if __name__ == "__main__":
    s3client = ...
    with open("/tmp/out", "w") as f:
        s3client.get(args.input_file, f)

    with open("/tmp/out", "r") as f:
        in = json.loads(f.read().decode("utf-8"))

    do_something_else(in["accuracy"])


Instead with the tracker a tracker with the same ``tracker_base``
can be used across apps to make the return values of one app
available to another without the need to chain output file paths of
one app with the input file path of another and deal with custom
serialization and file writing.

.. code-block:: python

 # in app1.py
 if __name__ == "__main__":
    accuracy = do_something()
    tracker = FsspecResultTracker(args.tracker_base)
    tracker["app1_out"] = {"accuracy": accuracy}

 # in app2.py
 if __name__ == "__main__":
    tracker = FsspecResultTracker(args.tracker_base)
    app1_accuracy = tracker["app1_out"]
    do_something_else(app1_accuracy)

"""

from .api import FsspecResultTracker, ResultTracker  # noqa: F401
