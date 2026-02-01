# Architecture

## Core Data Structures

All defined in `specs/api.py` as dataclasses:

| Type       | Description                                           |
|------------|-------------------------------------------------------|
| `AppDef`   | Application definition (name + list of Roles)         |
| `Role`     | Set of nodes performing a duty (trainer, PS, etc.)    |
| `Resource` | Resource requirements (cpu, gpu, memMB, capabilities) |
| `AppState` | State enum: UNSUBMITTED → SUBMITTED → PENDING → RUNNING → terminal |

```python
@dataclass
class AppDef:
    name: str
    roles: list[Role]
    metadata: Metadata = field(default_factory=Metadata)

@dataclass
class Role:
    name: str
    image: str
    entrypoint: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    resource: Resource = field(default_factory=lambda: Resource(cpu=1, memMB=128))
    num_replicas: int = 1
```

## Directory Structure

```
torchx/
├── specs/           # Core data structures & validation
│   ├── api.py       # AppDef, Role, Resource, AppState
│   ├── builders.py  # AppDef builder utilities
│   └── file_linter.py  # Component validation
├── runner/          # Runner API
│   ├── api.py       # Runner interface
│   └── config.py    # Configuration
├── schedulers/      # Scheduler implementations
│   ├── api.py       # Base Scheduler class
│   ├── local_scheduler.py
│   ├── docker_scheduler.py
│   ├── kubernetes_*.py
│   ├── aws_*.py
│   ├── slurm_scheduler.py
│   └── fb/          # Meta-internal schedulers
├── components/      # Built-in components
│   ├── dist.py      # Distributed training (DDP)
│   ├── serve.py     # Model serving
│   └── fb/          # Meta-internal components
├── cli/             # CLI interface
│   ├── main.py      # Entry point
│   └── cmd_*.py     # Subcommands
└── workspace/       # Workspace management
```

## Scheduler Plugin Pattern

All schedulers inherit from `schedulers/api.py`:

```python
class Scheduler(Generic[T]):
    def schedule(self, dryrun_info: AppDryRunInfo[T]) -> str: ...
    def describe(self, app_id: str) -> AppDef: ...
    def log_iter(self, app_id: str, role: str, index: int) -> Iterator[str]: ...
    def cancel(self, app_id: str) -> None: ...
```

Factory function pattern: `create_scheduler(session_name: str, **kwargs) -> Scheduler`

## Available Schedulers (OSS)

| Scheduler        | Module                          | Use Case                |
|------------------|---------------------------------|-------------------------|
| local            | `local_scheduler.py`            | Local process execution |
| docker           | `docker_scheduler.py`           | Container execution     |
| kubernetes       | `kubernetes_scheduler.py`       | K8s cluster             |
| kubernetes_mcad  | `kubernetes_mcad_scheduler.py`  | K8s with MCAD           |
| aws_batch        | `aws_batch_scheduler.py`        | AWS Batch               |
| aws_sagemaker    | `aws_sagemaker_scheduler.py`    | SageMaker               |
| slurm            | `slurm_scheduler.py`            | SLURM HPC               |
| lsf              | `lsf_scheduler.py`              | IBM LSF HPC             |

## Component Pattern

Components are functions returning `AppDef`:

```python
def my_component(
    script: str,
    j: str = "1x4",      # Job topology: {nnodes}x{nproc_per_node}
    image: str = "pytorch:latest",
    m: str | None = None,  # Python module
    env: dict[str, str] | None = None,
    h: str | None = None,  # Named resource
) -> AppDef:
    """Component docstring."""
    return AppDef(
        name="my_app",
        roles=[Role(name="worker", image=image, entrypoint=script, ...)],
    )
```

## Component Validation

`specs/file_linter.py` contains `ComponentFunctionValidator`:
- Validates component function signatures
- Checks return type is `AppDef`
- Accepts both old (`Dict`, `List`) and new (`dict`, `list`) type hints
- **Important**: Must continue accepting both styles for user-provided components
