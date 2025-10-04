#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generates ``versions.html`` listing out the available doc versions
similar to: https://raw.githubusercontent.com/pytorch/pytorch.github.io/site/docs/versions.html

This list of versions is displayed on the top LHS dropdown of
https://meta-pytorch.org/torchx.

Usage (also see ``doc_push.sh``):

::
 # NOTE: YOU HAVE TO RUN THE SCRIPT FROM THE ROOT of gh-pages branch checkout
 git clone -b gh-pages --single-branch https://github.com/meta-pytorch/torchx.git /tmp/torchx-gh-pages
 cd /tmp/torchx-gh-pages
 $torchx_repo_root/docs/versions_html.py`
"""

import os
from string import Template
from typing import List, Optional

from packaging.version import InvalidVersion, Version

VERSIONS_HTML_TEMPLATE = Template(
    """
<html>
  <head>
    <meta charset="utf-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="main/_static/css/theme.css" type="text/css" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Lato" type="text/css" />
    <link rel="stylesheet" href="main/_static/css/pytorch_theme.css" type="text/css" />
    <script src="main/_static/js/modernizr.min.js"></script>


  </head>
  <body>
    <div class="wy-nav-content">
      <div class="rst-content">
    <h1> TorchX Documentation </h1>
    <div class="toctree-wrapper compound">
      <p class="caption"><span class="caption-text">Pick a version</span></p>
      <ul>
$ver_list
      </ul>
    </div>
    </div></div>
  </body>
</html>
"""
)
# fmt: off
VERSION_TEMPLATE = Template("""
        <li class="toctree-l1">
          <a class="reference internal" href="$ver/">$desc</a>
        </li>
""")
# fmt: on

TAGS = {
    "main": "(unstable)",
    "latest": "(pre-release)",
    "stable": "(stable release)",
}

# map tags to faux versions for sorting
TAGS_VER_PROXY = {
    "main": Version("0.0.0.dev2"),
    "latest": Version("0.0.0.dev1"),
    "stable": Version("0.0.0.dev0"),
}
VERSIONS_HTML = "versions.html"


def parse_ver(version: str) -> Optional[Version]:
    if version in TAGS_VER_PROXY:
        return TAGS_VER_PROXY[version]
    try:
        return Version(version)
    except InvalidVersion:
        return None


def versions_list(versions: List[str]) -> List[str]:
    tag_list: List[(Version, str)] = []
    ver_list: List[(Version, str)] = []

    for ver in versions:
        v = parse_ver(ver)
        if not v:
            continue

        if ver in TAGS:
            desc = f"{ver} {TAGS[ver]} -> {os.readlink(ver)}"
            tag_list.append(
                (v, VERSION_TEMPLATE.substitute(ver=ver, desc=desc).strip("\n"))
            )
        else:
            desc = f"v{ver}"
            ver_list.append(
                (v, VERSION_TEMPLATE.substitute(ver=ver, desc=desc).strip("\n"))
            )

    tag_list.sort(key=lambda e: e[0], reverse=True)
    ver_list.sort(key=lambda e: e[0], reverse=True)
    return [e[1] for e in tag_list + ver_list]


def gen_versions_html() -> None:
    # cwd is expected to be https://github.com/meta-pytorch/torchx/tree/gh-pages
    # most top level subdirs are versioned docs
    print(f"Generating {VERSIONS_HTML}")

    subdirs = [d for d in os.listdir() if os.path.isdir(d)]
    ver_list = versions_list(versions=subdirs)
    versions_html = VERSIONS_HTML_TEMPLATE.substitute(ver_list="\n".join(ver_list))
    with open(VERSIONS_HTML, "w") as f:
        f.write(versions_html)

    versions_html_abspath = os.path.join(os.getcwd(), VERSIONS_HTML)
    print(f"Wrote {len(ver_list)} versions to {versions_html_abspath}")


if __name__ == "__main__":
    gen_versions_html()
