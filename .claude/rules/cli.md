# CLI Usage

## Main Command: `torchx run`

```bash
torchx run [options] <component> [component_args]
```

### Options

| Flag              | Description                                    |
|-------------------|------------------------------------------------|
| `-s, --scheduler` | Scheduler backend (local, docker, kubernetes)  |
| `--cfg`           | Scheduler configuration (key=value)            |
| `-n, --name`      | Application name                               |
| `--dryrun`        | Print job definition without submitting        |

### Examples

```bash
# Local execution
torchx run -s local dist.ddp -j 1x4 --script train.py

# Kubernetes with 2 nodes × 4 procs
torchx run -s kubernetes dist.ddp -j 2x4 --script main.py

# Docker execution
torchx run -s docker dist.ddp -j 1x2 --script train.py

# Dryrun (inspect without submitting)
torchx run -s local dist.ddp -j 1x4 --script main.py --dryrun
```

## Other Commands

```bash
torchx status <app_id>     # Check app status
torchx log <app_id>        # View logs
torchx cancel <app_id>     # Cancel running app
torchx describe <app_id>   # Detailed app info
torchx list                # List apps
```

## Component Resolution

Components can be specified as:

```bash
# Built-in component
torchx run dist.ddp ...

# File-based component
torchx run ~/components.py:my_component ...

# Module path
torchx run my_package.components:train ...
```

## Common Component Parameters

| Param   | Description                                      |
|---------|--------------------------------------------------|
| `j`     | Job topology: `{nnodes}x{nproc_per_node}`        |
| `m`     | Python module to run                             |
| `h`     | Named resource (gpu type, instance type)         |
| `env`   | Environment variables                            |
| `image` | Container image                                  |
| `script`| Script or binary to run                          |
