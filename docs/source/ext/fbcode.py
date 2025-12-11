#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


class FbcodeDirective(SphinxDirective):
    """
    Includes the content of this directive if running in fbcode.

    Used to add fb-specific (internal) documentation to display in
    StaticDocs (https://www.internalfb.com/intern/staticdocs/torchx).

    To exclude the contents in the directive (e.g. for oss-only documentation)
    when building docs in fbcode, use the ``:exclude:`` option (see example below).

    Usage:

    ```
    List of supported components:

     * ``utils.python``

     .. fbcode::
        :exclude:

        * ``utils.echo``

     .. fbcode::

        * ``fb.dist.hpc``
    ```

    In the example above, ``utils.echo`` will be listed only when building outside of fbcode.
    Similarly ``fb.dist.hpc`` will be listed only when buildincg in fbcode.
    """

    # this enables content in the directive
    has_content = True
    option_spec = {
        "exclude": directives.flag,  # if present, includes contents EXCEPT when in fbcode
    }

    def run(self):
        exclude_in_fbcode = "exclude" in self.options
        is_fbcode = "fbcode" in os.getcwd()

        if is_fbcode ^ exclude_in_fbcode:
            node = nodes.section()
            node.document = self.state.document
            nested_parse_with_titles(self.state, self.content, node)
            return node.children
        else:
            return []


def setup(app):
    app.add_directive("fbcode", FbcodeDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
