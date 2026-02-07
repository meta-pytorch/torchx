---
oncalls: ['torchx_core']
apply_to_regex: '.*'
---

# Coding Conventions

## Docstrings

Google Style. See `python-docstring-google-style.md` for full guide.

### Code Examples: Prefer Doctest

**Use `.. doctest::` over plain code blocks** for Python examples in docstrings.
Doctest examples are validated by Sphinx during doc builds.

```python
def parse_resource(resource_str: str) -> Resource:
    """Parse a resource string into a Resource object.

    Example:
        .. doctest::

            >>> parse_resource("cpu=2,gpu=1,memMB=1024")
            Resource(cpu=2, gpu=1, memMB=1024)
            >>> parse_resource("gpu=4")
            Resource(cpu=1, gpu=4, memMB=128)

    """
```

**When to use each:**
- `.. doctest::` - Python code examples (validated)
- `.. code:: shell-session` - CLI/shell examples (not validated)
- `.. code-block:: python` - Python snippets not suitable for doctest (setup code, partial examples)

### Component Docstrings

Components lead with CLI usage examples:

```python
def ddp(*args, script: str, j: str = "1x2") -> AppDef:
    """Distributed data parallel application.

    .. code:: shell-session

        $ torchx run dist.ddp -j 1x4 --script main.py
    """
```

## Markdown Tables

Align columns so they line up in raw markdown (not just when rendered):

```markdown
| Command   | Description                |
|-----------|----------------------------|
| `/lint`   | Run lintrunner with fixes  |
| `/test`   | Run pytest                 |
```

## File Headers

```python
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
```

For `fb/` files: `# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.`

## Type Hints

```python
# Prefer built-in types; component validation supports both dict/Dict, list/List
def get_user(user_id: int) -> User | None: ...
def process(items: list[str], mapping: dict[str, int]) -> tuple[str, ...]: ...
```

## Imports

Order: stdlib → stdlib-from → local (alphabetical)

## Logging & Errors

```python
logger: logging.Logger = logging.getLogger(__name__)

# Use %s (not f-strings); lowercase, no trailing period; backtick variables
logger.debug("fetching app: `%s` from scheduler: `%s`", app_id, scheduler)
raise ValueError(f"unknown scheduler: `{scheduler}`")
```

## Component Parameters

`j` (job topology), `m` (module), `h` (host), `env`, `image`

## Tests

Location: `module/test/module_test.py` | Class: `ModuleTest` | Method: `test_*`

## Runopts (Scheduler Configuration)

**Avoid adding runopts to implement features.** This should be a last resort.

**Why:**
1. Runopts reduce AppDef portability—the AppDef becomes tightly coupled to specific schedulers
2. Runopts obscure scheduler behavior, making it hard for users to reason about the resulting job

**Preferred alternatives:**
- **Overlays** in AppDef or Role metadata (search scheduler source for logic that applies user-specified overrides on top of the generated request)
- **Workspace type checking**: `isinstance(self, SomeWorkspace)`
- **Scheduler/workspace composition**: Compose schedulers and workspaces to encapsulate behavior
- **Extend core APIs**: Add attributes/methods to `torchx.specs`, `torchx.schedulers`, `torchx.workspace`, or `torchx.runner`

**Anti-patterns (avoid these):**
- **Resource capabilities**: `resource.capabilities["key"] = "value"` — overloads an unrelated field

Only add runopts when no alternative exists and the option is fundamental to scheduler operation.
