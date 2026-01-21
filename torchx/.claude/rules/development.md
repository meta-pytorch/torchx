# Development (Git Checkout)

See `.github/workflows/` for CI commands. Common commands:

```bash
pip install -e ".[dev]"  # install dev deps
lintrunner -a            # lint + auto-fix
pyre check               # type-check
pytest                   # test
```
