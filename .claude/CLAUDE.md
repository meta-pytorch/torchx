# TorchX

Job launcher and orchestration library for PyTorch. Core types in `specs/api.py`: `AppDef`, `Role`, `Resource`, `AppState`.

## Development

| Environment | Tests | Lint + Format | Type Check |
|-------------|-------|---------------|------------|
| OSS (`github/`) | `uv run pytest` | `uv run lintrunner -a` | `uv run pyre check` |
| fbsource | `buck2 test //torchx/...` | `arc lint -a` / `arc f` | `arc pyre check-changed-targets` |

OSS repo root is `github/`. Setup: `cd github && uv sync --all-extras`.

Run `uv lock` after changing `pyproject.toml`. Commit both together.

## Conventions

**Headers**: OSS: `# Copyright (c) Meta Platforms, Inc. and affiliates.` + BSD license + `# pyre-strict`. fb/: `# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.`

**Docstrings**: Google Style, Sphinx/Napoleon compatible. Succinct — skip Args/Returns when obvious. Prefer `.. doctest::` over code blocks. Components lead with CLI examples. Dataclasses use `Args:`. Use `:py:class:`/`:py:func:` cross-references.

**Type hints**: built-in types (`list`, `dict`, `X | None`). Component validation accepts both old and new styles.

**Imports**: stdlib → stdlib-from → local (alphabetical within each group).

**Logging**: `%s` formatting (not f-strings), lowercase, no trailing period, backtick variable names.

**Tests**: `module/test/module_test.py` | Class: `ModuleTest` | Method: `test_*`

**Runopts**: avoid adding. They reduce AppDef portability. Prefer metadata overlays or extending core APIs.
