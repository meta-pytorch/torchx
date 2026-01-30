# Docker

TorchX provides Docker scheduler and container support for local container-based execution.

## Container Image

**Dockerfile**: `runtime/container/Dockerfile`
**Base**: `pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime`
**Registry**: `ghcr.io/meta-pytorch/torchx`

### Build Pattern (Layer Optimization)

```dockerfile
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy deps first (cached layer)
COPY pyproject.toml uv.lock /app/

# Install deps without project (cached)
RUN uv sync --frozen --no-install-project --extra dev

# Copy code
COPY . /app

# Install project
RUN uv sync --frozen --extra dev

ENV PATH="/app/.venv/bin:$PATH"
```

**Key Pattern**: Separate dependency installation from code copy for layer caching.

### Building

```bash
# Build
runtime/container/build.sh

# Tag
docker tag torchx ghcr.io/meta-pytorch/torchx:VERSION

# Push
docker push ghcr.io/meta-pytorch/torchx:VERSION
```

## Docker Scheduler

**Module**: `schedulers/docker_scheduler.py`

### Usage

```bash
torchx run -s docker dist.ddp -j 1x2 --script train.py
```

### Key Implementation Details

**Network**: Uses `torchx` bridge network for container communication

**Container Labels**:
- `torchx.pytorch.org/app-id`
- `torchx.pytorch.org/role-name`
- `torchx.pytorch.org/replica-id`

**State Mapping**:
| Docker State | TorchX State |
|--------------|--------------|
| created      | SUBMITTED    |
| restarting   | PENDING      |
| running      | RUNNING      |
| paused       | PENDING      |
| dead         | FAILED       |

**Docker Detection**:
```python
from torchx.schedulers.docker_scheduler import has_docker
if has_docker():
    # Docker is available
```

**Network Creation** (multi-process safe with filelock):
```python
from torchx.schedulers.docker_scheduler import ensure_network
ensure_network()  # Creates 'torchx' network if needed
```

## Docker Workspace

**Module**: `workspace/docker_workspace.py`

Builds patched Docker images from workspace directories:
- Overlays local files onto base image
- Supports `.dockerignore`
- Pushes to remote registries for remote schedulers

**Options**:
- `image_repo`: Target repository for pushing
- `quiet`: Suppress verbose build output

## Common Issues

**Docker not available**: Check `has_docker()` before using Docker scheduler

**Build context too large**: Use `.dockerignore` to exclude `.venv/`, `__pycache__/`, `.git/`

**Network race condition**: TorchX uses `filelock` for safe network creation

**Authentication for push**: Run `docker login ghcr.io` before pushing
