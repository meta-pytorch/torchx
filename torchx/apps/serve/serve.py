#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import binascii
import os
import socket
import sys
import tempfile
import threading
from functools import partial
from getpass import getuser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import fsspec
import requests

TORCHSERVE_PARAMS = (
    "model_name",
    "handler",
    "runtime",
    "batch_size",
    "max_batch_delay",
    "initial_workers",
    "synchronous",
    "response_timeout",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="uploads the provided model to torchserve",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model to serve",
        required=True,
    )
    parser.add_argument(
        "--management_api",
        type=str,
        help="address of the management api. e.g. http://localhost:8081",
        required=True,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout for requests to management api",
        default=60,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--port",
        type=int,
        help="""port for the HTTP file server to listen on when torchserve is loading the model.
          This must be accessible from the torchserve instance.""",
        default=8222,
    )

    # arguments from https://pytorch.org/serve/management_api.html#register-a-model
    for param in TORCHSERVE_PARAMS:
        parser.add_argument(
            f"--{param}",
            type=str,
            help=f"""torchserve parameter {param}.
            See https://pytorch.org/serve/management_api.html#register-a-model""",
        )
    return parser.parse_args(argv)


def get_routable_ip_to(addr: str) -> str:
    """
    get_routable_ip_to opens a dummy connection to the target HTTP URL and
    returns the IP address used to connect to it.
    """
    parsed = urlparse(addr)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((parsed.hostname, parsed.port or 80))
        return s.getsockname()[0]
    finally:
        s.close()


def rand_id() -> str:
    id = binascii.b2a_hex(os.urandom(8)).decode("utf-8")
    return f"{getuser()}_{id}"


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    if args.dryrun:
        print("App serve started successfully")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        model_name = args.model_name or "model"
        model_file = f"{model_name}_{rand_id()}.mar"
        model_path = os.path.join(tmpdir, model_file)
        print(f"downloading model from {args.model_path} to {model_path}...")
        fs, _, rpaths = fsspec.get_fs_token_paths(args.model_path)
        assert len(rpaths) == 1, "must have single path"
        fs.get(rpaths[0], model_path)

        addr = ("", args.port)
        print(f"starting HTTP server at {addr}...")

        handler_class = partial(SimpleHTTPRequestHandler, directory=tmpdir)
        server: ThreadingHTTPServer = ThreadingHTTPServer(addr, handler_class)

        try:

            def serve() -> None:
                server.serve_forever()

            t = threading.Thread(target=serve)
            t.start()

            ip_address = get_routable_ip_to(args.management_api)
            model_url = f"http://{ip_address}:{server.server_port}/{model_file}"
            print(f"serving file at {model_url}")

            url = f"{args.management_api}/models"
            print(f"POST {url}")
            payload = {
                "url": model_url,
            }
            for param in TORCHSERVE_PARAMS:
                v = getattr(args, param)
                if v is not None:
                    payload[param] = v
            r = requests.post(url, params=payload, timeout=args.timeout)
            print(r.text)
            r.raise_for_status()

        finally:
            print("shutting down...")
            server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
