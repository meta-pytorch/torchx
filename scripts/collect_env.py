# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# This script uses https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
# and collects additional information on top of it to output relevant system
# environment info.
# Run it with `python collect_env.py`.
import re
import subprocess
import sys
import tempfile
from os import getenv
from os.path import exists
from urllib import request

PYTORCH_COLLECT_ENV_URL = "https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py"
TORCHX_PACKAGES = (
    "https://raw.githubusercontent.com/pytorch/torchx/main/dev-requirements.txt"
)


def run(
    command: str, filter_output_regexp: str | None = None
) -> tuple[int, bytes, bytes] | None:
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        args=command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    raw_output, raw_err = p.communicate()
    raw_output, raw_err = raw_output.strip().decode("utf-8"), raw_err.strip().decode(
        "utf-8"
    )
    rc = p.returncode
    if rc != 0:
        return None

    if filter_output_regexp:
        match = re.search(filter_output_regexp, raw_output)
        if match is None:
            return None
        return match.group(1)

    # pyre-fixme[7]: Expected `Optional[Tuple[int, bytes, bytes]]` but got
    #  `Tuple[typing.Any, str, str]`.
    return rc, raw_output, raw_err


def get_pip_packages() -> str:
    """Returns versions of packages that match torchx dev requirements"""
    user_packages = subprocess.run(
        f"{sys.executable + ' -mpip'} list --format=freeze",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    torchx_packages, _ = request.urlretrieve(TORCHX_PACKAGES)
    with open(torchx_packages, "r") as packages:
        torchx_deps = [
            re.split(r"==|>=|<=|!=|!=|===|<|>", package.strip())[0]
            for package in packages.readlines()
            if package.strip() and not package.startswith("#")
        ]
        assert torchx_deps is not None

    user_deps = [
        re.split(r"==|>=|<=|!=|!=|===|<|>", line)
        for line in user_packages.stdout.decode("utf-8").splitlines()
    ]

    return "\n".join(
        f"{udeps[0]}:{udeps[1]}"
        for udeps in user_deps
        if any(tdeps in udeps[0] for tdeps in torchx_deps)
    )


def get_torchx_config() -> str:
    torchxconfig = None
    if exists(".torchxconfig"):
        torchxconfig = ".torchxconfig"
    elif exists(f"{getenv('HOME')}/.torchxconfig"):
        torchxconfig = f"{getenv('HOME')}/.torchxconfig"
    else:
        return "N/A"

    with open(torchxconfig, "r") as f:
        return f.read()


def run_pytorch_collect_env() -> tuple[int, bytes]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as temp:
        request.urlretrieve(PYTORCH_COLLECT_ENV_URL, temp.name)
        out = subprocess.run(
            f"{sys.executable} {temp.name}", stderr=subprocess.PIPE, shell=True
        )
        return out.returncode, out.stderr


def get_cli_info() -> None:
    print(f"AWS CLI: {get_aws_version()}")
    print(f"gCloud CLI: {get_gcp_version()}")
    print(f"AZ CLI: {get_azure_version()}")
    print(f"Slurm: {get_slurm_version()}")
    print(f"Docker: {get_docker_version()}")
    print(f"kubectl: {get_kubectl_version()}")


def get_aws_version() -> str:
    result = run("aws --version")
    if result:
        return str(result[1])
    return "N/A"


def get_gcp_version() -> str:
    result = run("gcloud --version", r"Google Cloud (.*)")
    return str(result)


def get_azure_version() -> str:
    result = run("az version", r"\"azure-cli\": (.*)")
    return str(result)


def get_slurm_version() -> str:
    result = run("slurmd --version")
    if result:
        return str(result[1])
    return "N/A"


def get_docker_version() -> str:
    result = run("docker --version", r"Docker version (.*)")
    return str(result)


def get_kubectl_version() -> str:
    result = run("kubectl version --client", r"Client Version: (.*)")
    return str(result)


def main() -> None:
    status = run_pytorch_collect_env()
    if status[0] != 0:
        print(f"Could not run Pytorch collect_env script: {status[1]}")

    print("\nVersions of CLIs:")
    get_cli_info()

    print("\ntorchx dev package versions:")
    print(get_pip_packages())

    print("\ntorchx config:")
    print(f"{get_torchx_config()}")


if __name__ == "__main__":
    main()
