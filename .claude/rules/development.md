---
oncalls: ['torchx_core']
apply_to_regex: '.*'
---

# Development (Git Checkout)

## Quick Start

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Verify
uv run torchx --help
```

## Common Commands

| Task            | Command                          |
|-----------------|----------------------------------|
| Install deps    | `uv sync --extra dev`            |
| Run tests       | `uv run pytest`                  |
| Run with cov    | `uv run pytest --cov`            |
| Lint + autofix  | `uv run lintrunner -a`           |
| Type check      | `uv run pyre check`              |
| Setup pyre      | `uv run lintrunner init`         |
| Run CLI         | `uv run torchx --help`           |

Or activate the venv first:

```bash
source .venv/bin/activate
pytest
lintrunner -a
pyre check
torchx --help
```

## CI Workflows

See `.github/workflows/` for CI commands:

| Workflow           | File                                  | Purpose              |
|--------------------|---------------------------------------|----------------------|
| Lint               | `lint.yaml`                           | Code quality         |
| Type check         | `pyre.yaml`                           | Pyre type checking   |
| Unit tests         | `python-unittests.yaml`               | pytest (3.10-3.12)   |
| Integration        | `components-integration-tests.yaml`   | Local + K8s tests    |
| Docs               | `doc-build.yaml`                      | Sphinx documentation |
| Container          | `container.yaml`                      | Docker image build   |

## Related

- **UV details**: `rules/uv.md`
- **Docker**: `rules/docker.md`
- **fbsource dev**: `rules/fb-development.md`
