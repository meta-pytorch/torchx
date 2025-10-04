---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Quickstart

This is a self contained guide on how to write a simple app and start launching
distributed jobs on local and remote clusters.

## Installation

First thing we need to do is to install the TorchX python package which includes
the CLI and the library.

<!-- #md -->
```sh
# install torchx with all dependencies
$ pip install "torchx[dev]"
```
<!-- #endmd -->

See the [README](https://github.com/meta-pytorch/torchx) for more
information on installation.

```sh
torchx --help
```

## Hello World

Lets start off with writing a simple "Hello World" python app. This is just a
normal python program and can contain anything you'd like.

<div class="admonition note">
<div class="admonition-title">Note</div>
This example uses Jupyter Notebook `%%writefile` to create local files for
example purposes. Under normal usage you would have these as standalone files.
</div>

```python
%%writefile my_app.py

import sys

print(f"Hello, {sys.argv[1]}!")
```

## Launching

We can execute our app via `torchx run`. The
`local_cwd` scheduler executes the app relative to the current directory.

For this we'll use the `utils.python` component:

```sh
torchx run --scheduler local_cwd utils.python --help
```

The component takes in the script name and any extra arguments will be passed to
the script itself.

```sh
torchx run --scheduler local_cwd utils.python --script my_app.py "your name"
```

We can run the exact same app via the `local_docker` scheduler. This scheduler
will package up the local workspace as a layer on top of the specified image.
This provides a very similar environment to the container based remote
schedulers.

<div class="admonition note">
<div class="admonition-title">Note</div>
This requires Docker installed and won't work in environments such as Google
Colab. See the Docker install instructions:
[https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)</a>
</div>

```sh
torchx run --scheduler local_docker utils.python --script my_app.py "your name"
```

TorchX defaults to using the
[ghcr.io/pytorch/torchx](https://ghcr.io/pytorch/torchx) Docker container image
which contains the PyTorch libraries, TorchX and related dependencies.

## Distributed

TorchX's `dist.ddp` component uses
[TorchElastic](https://pytorch.org/docs/stable/distributed.elastic.html)
to manage the workers. This means you can launch multi-worker and multi-host
jobs out of the box on all of the schedulers we support.

```sh
torchx run --scheduler local_docker dist.ddp --help
```

Lets create a slightly more interesting app to leverage the TorchX distributed
support.

```python
%%writefile dist_app.py

import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo")
print(f"I am worker {dist.get_rank()} of {dist.get_world_size()}!")

a = torch.tensor([dist.get_rank()])
dist.all_reduce(a)
print(f"all_reduce output = {a}")
```

Let launch a small job with 2 nodes and 2 worker processes per node:

```sh
torchx run --scheduler local_docker dist.ddp -j 2x2 --script dist_app.py
```

## Workspaces / Patching

For each scheduler there's a concept of an `image`. For `local_cwd` and `slurm`
it uses the current working directory. For container based schedulers such as
`local_docker`, `kubernetes` and `aws_batch` it uses a docker container.

To provide the same environment between local and remote jobs, TorchX CLI uses
workspaces to automatically patch images for remote jobs on a per scheduler
basis.

When you launch a job via `torchx run` it'll overlay the current directory on
top of the provided image so your code is available in the launched job.

For `docker` based schedulers you'll need a local docker daemon to build and
push the image to your remote docker repository.

## `.torchxconfig`

Arguments to schedulers can be specified either via a command line flag to
`torchx run -s <scheduler> -c <args>` or on a per scheduler basis via a
`.torchxconfig` file.

```python
%%writefile .torchxconfig

[kubernetes]
queue=torchx
image_repo=<your docker image repository>

[slurm]
partition=torchx
```

## Remote Schedulers

TorchX supports a large number of schedulers.
Don't see yours?
[Request it!](https://github.com/meta-pytorch/torchx/issues/new?assignees=&labels=&template=feature-request.md)

Remote schedulers operate the exact same way the local schedulers do. The same
run command for local works out of the box on remote.

<!-- #md -->
```sh
$ torchx run --scheduler slurm dist.ddp -j 2x2 --script dist_app.py
$ torchx run --scheduler kubernetes dist.ddp -j 2x2 --script dist_app.py
$ torchx run --scheduler aws_batch dist.ddp -j 2x2 --script dist_app.py
```
<!-- #endmd -->

Depending on the scheduler there may be a few extra configuration parameters so
TorchX knows where to run the job and upload built images. These can either be
set via `-c` or in the `.torchxconfig` file.


All config options:

```sh
torchx runopts
```


## Custom Images

### Docker-based Schedulers

If you want more than the standard PyTorch libraries you can add custom
Dockerfile or build your own docker container and use it as the base image for
your TorchX jobs.


```python
%%writefile timm_app.py

import timm

print(timm.models.resnet18())
```

```python
%%writefile Dockerfile.torchx

FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN pip install timm

COPY . .
```

Once we have the Dockerfile created we can launch as normal and TorchX will
automatically build the image with the newly provided Dockerfile instead of the
default one.

```sh
torchx run --scheduler local_docker utils.python --script timm_app.py
```

### Slurm

The `slurm` and `local_cwd` use the current environment so you can use `pip` and
`conda` as normal.


## Next Steps

1. Checkout other features of the [torchx CLI](cli.rst)
2. Take a look at the [list of schedulers](schedulers.rst) supported by the runner
3. Browse through the collection of [builtin components](components/overview.rst)
5. See a [training app example](examples_apps/index.rst)
