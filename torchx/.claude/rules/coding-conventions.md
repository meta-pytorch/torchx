# Coding Conventions

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
