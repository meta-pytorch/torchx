# UV Package Manager

TorchX uses [UV](https://github.com/astral-sh/uv) for Python dependency management.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Core Commands

| Command                              | Purpose                                    |
|--------------------------------------|--------------------------------------------|
| `uv sync --extra dev`                | Install all dev dependencies               |
| `uv sync --frozen --extra dev`       | Install from lock file (CI/reproducible)   |
| `uv run <command>`                   | Execute in virtual environment             |
| `uv lock`                            | Generate/update lock file                  |
| `uv lock --upgrade`                  | Upgrade all dependencies                   |
| `uv lock --upgrade-package <name>`   | Upgrade specific package                   |
| `uv pip install <pkg>`               | Install additional package                 |

## Development Workflow

```bash
# Initial setup
uv sync --extra dev

# Run tests
uv run pytest --cov=./ --cov-report=xml

# Lint and type check
uv run lintrunner init      # Sets up pyre
uv run lintrunner -a        # Auto-fix linting issues

# Verify CLI
uv run torchx --help
```

## Lock File Management

**Files**: `pyproject.toml` (dependencies) + `uv.lock` (locked versions)

**When to update `uv.lock`**:
1. After modifying dependencies in `pyproject.toml`
2. When user requests dependency updates
3. If lock file is missing or corrupted
4. Before merging PRs that change dependencies

**Workflow**:
```bash
# 1. Edit pyproject.toml
# 2. Regenerate lock
uv lock

# 3. Install and test
uv sync --extra dev
uv run pytest

# 4. Commit both files together
git add pyproject.toml uv.lock
```

## CI/CD Pattern

All GitHub Actions workflows use the `--frozen` flag for reproducibility:

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4
  with:
    python-version: "3.12"
    enable-cache: true

- name: Install dependencies
  run: uv sync --frozen --extra dev
```

## Optional Dependency Groups

| Extra        | Packages                          | Use Case              |
|--------------|-----------------------------------|-----------------------|
| `dev`        | pytest, boto3, kubernetes, etc.   | Development/testing   |
| `docs`       | sphinx, nbsphinx, matplotlib      | Documentation build   |
| `kubernetes` | kubernetes>=11                    | K8s scheduler         |
| `aws_batch`  | boto3                             | AWS Batch scheduler   |

Install multiple extras: `uv sync --extra dev --extra docs`

## Common Issues

**Lock file conflicts**: Regenerate with `uv lock`

**ModuleNotFoundError**: Use `uv run <command>` instead of running Python directly

**Stale environment**: Delete `.venv/` and run `uv sync --extra dev`
