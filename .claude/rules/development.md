# Development (Git Checkout)

See `.github/workflows/` for CI commands. Common commands:

```bash
uv sync --extra dev    # install dev deps
uv run lintrunner -a   # lint + auto-fix
uv run pyre check      # type-check
uv run pytest          # test
uv run torchx --help   # verify CLI works
```

Or activate the venv first:

```bash
source .venv/bin/activate
lintrunner -a
pyre check
pytest
torchx --help
```
