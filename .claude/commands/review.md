---
description: Senior engineer code review with TorchX-specific checks
---

# /review - TorchX Code Review

Senior engineer code review with TorchX-specific checks.

## Instructions

### Step 1: Identify changed files

Detect the VCS and get changed files:
- Sapling: `sl status --no-status`
- Git: `git diff --name-only HEAD`

Filter to files under the `torchx/` tree.

### Step 2: Validate modified Python files

Detect the environment and run the appropriate validation tools on each modified `.py` file. Collect all failures — these are **blocking** issues.

**Sapling/fbsource** (has `arc` and `.hg/`):
- `arc f <files>` (format)
- `arc lint -a <files>` (lint + autofix)
- `arc pyre check-owning-targets <files>` (type check)

**Git checkout** (has `pyproject.toml` and `uv`):
- `uv run lintrunner -a` (lint + format)
- `uv run pyre check` (type check)

### Step 3: Review changes

Spawn a subagent (Task tool, subagent_type="general-purpose") to review the diff. Pass the list of changed files and instruct it to read each file and the corresponding diff (`sl diff` or `git diff`), then evaluate against **all** of the following checklists.

#### Generic review

- Correctness and edge cases
- API design and usability
- Performance implications
- Security considerations (OWASP top 10)
- Consistency with project patterns
- Tech debt introduced

#### Backwards compatibility

TorchX has two audiences: **users** (who write components, submit jobs, use the Runner API) and **plugin implementors** (who write schedulers and workspaces by subclassing `Scheduler`/`WorkspaceMixin`). Check changes against both:

- **User-facing**: changes to `specs.AppDef`, `specs.Role`, `specs.Resource`, `Runner` API, component signatures, CLI arguments, and config keys must not break existing usage without a deprecation path
- **Plugin-facing**: changes to `schedulers/api.py` base class (abstract methods, method signatures, return types), `WorkspaceMixin` interface, `SchedulerBackend` registration, or `runopts` must not break existing scheduler/workspace implementations
- Renamed or removed public symbols need re-exports or deprecation warnings — not silent removal
- New required fields on core types (`AppDef`, `Role`, `Resource`) must have defaults
- Changes to serialization formats (e.g. `AppDef` to/from JSON) must remain compatible with existing persisted data

#### Tests

Tests are reference usage documentation — each test case should read like an example of how users interact with the API.

- Tests exist for new/modified code (expected at `module/test/module_test.py`)
- **Refactoring**: when code moves from one module to another, tests must move too — from the source module's test file to the destination module's test file
- After moving or adding tests, review neighboring test cases in the destination file as a whole: do they still represent how users would use these APIs? Consolidate where it makes sense
- Prefer adding an assertion to an existing test case over creating a new test case when the new assertion fits the existing test's usage scenario — don't fragment coherent API usage into separate tests just for coverage
- Test classes: `{Feature}Test(unittest.TestCase)`, methods: `test_{behavior}(self) -> None`
- Tests construct real objects (`AppDef`, `Role`, `Resource`) and assert behavior, not implementation details
- Edge cases tested: invalid inputs with `assertRaises`, boundary conditions
- Component tests extend `ComponentTestCase` and call `self.validate(module, fn)`
- Non-obvious assertions have messages

#### Docstrings

- Google Style, Sphinx/Napoleon compatible
- Succinct — skip Args/Returns when obvious from names and types
- Use `.. doctest::` over `.. code-block:: python` for Python examples
- Use `.. code:: shell-session` for CLI examples
- Components lead with CLI usage examples
- Dataclasses use `Args:` (not `Attributes:`) for fields
- Use `:py:class:`/`:py:func:`/`:py:meth:` cross-references instead of backtick-only references
- Show over tell — prefer examples over prose

#### Spelling

- Check comments, docstrings, error messages, and variable/function names for typos
- Flag misspelled words as **nit** unless they appear in user-facing strings (error messages, CLI help text, docstrings) — those are **suggestion**

#### Conventions

- File headers: BSD license + `# pyre-strict` for OSS files, `# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.` for `fb/` files
- **ShipIt boundary**: OSS code (outside `fb/` directories) must never import from `fb/` paths — this breaks the open-source build
- Every `assert` statement must have a message explaining the invariant
- Dead code: if something is unused, delete it completely — no `_var` renames, no `# removed` comments, no stale re-exports
- Component signatures must accept both `Dict`/`List` and `dict`/`list` (validated by `specs/file_linter.py`)
- Import order: stdlib → stdlib-from → local (alphabetical within each group)
- Logger formatting: `%s` not f-strings, lowercase messages, no trailing period, backtick variable names
- Error messages: lowercase, no trailing period (PEP 8), include relevant variable values, and always suggest a fix or point to documentation — don't just state what went wrong. Example: `raise ValueError(f"unknown scheduler: \`{name}\`, choose from: {', '.join(SCHEDULERS)}")`
- Exception handling: use `raise ... from e` to chain when translating exception types, bare `raise` to re-raise, `raise ... from None` only to intentionally suppress context. Never swallow exceptions silently. Choose the most specific exception type (`ValueError`, `TypeError`, `KeyError`, etc.) over generic `Exception` or `RuntimeError`
- Log messages: consistent with error message style — lowercase, descriptive, include context from local variables
- No new runopts without justification (they reduce AppDef portability)
- Type hints use built-in types (`list`, `dict`, `X | None`)
- Module imports for non-stdlib: `from torchx import specs; specs.AppDef(...)` not `from torchx.specs import AppDef`

### Step 4: Present findings

Organize review feedback by severity:

- **Blocking**: Must fix before landing (lint/format/pyre failures, missing tests, correctness bugs, security issues)
- **Suggestion**: Should fix, improves quality (missing docstrings, convention violations, test gaps)
- **Nit**: Optional polish (style preferences, minor naming)

For each issue provide: what's wrong, why it matters, and a suggested fix with file path and line number.

### Step 5: Offer to fix

After presenting feedback, offer to address blocking issues.
