# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Training machine learning models often requires custom train loop and custom
code. As such, we don't provide an out of the box training loop app. We do
however have examples for how you can construct your training app as well as
generic components you can use to run your custom training app.


1. :ref:`component_best_practices:Component Best Practices`
3. See :py:mod:`torchx.components` to learn more about authoring components
4. For more information on distributed training see :py:mod:`torchx.components.dist`.

Embedded Train Script
#######################

For simple apps you can use the :py:meth:`torchx.components.utils.python`
component to embed the training script as a command line argument to the Python
command. This has a size limitation but works for many smaller apps.

>>> from torchx.components.utils import python
>>> app = \"\"\"
... import sys
... print(f"Hello, {sys.argv[0]}")
... \"\"\"
>>> python("TorchX user", c=app)
AppDef(..., entrypoint='python', ...)

"""
