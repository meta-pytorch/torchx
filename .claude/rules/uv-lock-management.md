# UV Lock File Management

## When to Update uv.lock

Automatically run `uv lock` whenever:

1. **Dependencies change in pyproject.toml** - After adding, removing, or modifying any dependency in `[project.dependencies]` or `[project.optional-dependencies]`

2. **User requests dependency updates** - When asked to update dependencies, run `uv lock --upgrade` (all) or `uv lock --upgrade-package <name>` (specific)

3. **uv.lock is missing** - If the lock file doesn't exist, generate it with `uv lock`

4. **Lock file conflicts** - If `uv sync` fails due to lock file issues, regenerate with `uv lock`

## Commands

```bash
# Regenerate lock after pyproject.toml changes
uv lock

# Update all dependencies to latest compatible versions
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>

# Install in editable mode with dev dependencies
uv sync --extra dev

# Verify installation works
uv run torchx --help
```

## Workflow

1. Edit `pyproject.toml` to add/modify dependencies
2. Run `uv lock` to update the lock file
3. Run `uv sync --extra dev` to install
4. Test that the changes work
5. Commit both `pyproject.toml` and `uv.lock` together

## Notes

- The `uv.lock` file should always be committed alongside `pyproject.toml` changes
- If network access to PyPI is unavailable, inform the user to run `uv lock` manually
- The lock file ensures reproducible builds across all environments
