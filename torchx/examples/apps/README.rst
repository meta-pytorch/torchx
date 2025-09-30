Application Examples
====================

This contains the example applications that demonstrates how to use TorchX
for various styles of applications (e.g. single node, distributed, etc).
These apps can be launched by themselves or part of a pipeline. It is important
to note that TorchX's job is to launch the apps. You'll notice that the apps
are implemented without any TorchX imports.

See the Pipelines Examples for how to use the components in a pipeline.

Prerequisites
################

Before executing examples, install TorchX and dependencies necessary to run examples:

```
$ pip install torchx
$ git clone https://github.com/meta-pytorch/torchx.git
$ cd torchx/examples/apps
$ TORCHX_VERSION=$(torchx --version | sed 's/torchx-//')
$ git checkout v$TORCHX_VERSION
$ pip install -r dev-requirements.txt
```
