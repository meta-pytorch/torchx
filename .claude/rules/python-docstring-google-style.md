# Python Docstring Guidelines for TorchX

## Scope

This rule applies to:
- `torchx/` directories in git checkouts (github)
- `fbcode/torchx/` or `torchx/github/` directories in hg checkouts (fbcode/fbsource)

## Overview

TorchX uses **Sphinx** with the **Napoleon extension** to generate documentation. Docstrings are automatically included in the documentation using `automodule`, `autoclass`, `automethod`, and `autofunction` directives.

All docstrings MUST be:
1. **Sphinx compatible** - parseable by sphinx.ext.napoleon
2. **Google Style formatted** - following the Google Python Style Guide
3. **Succinct** - document only what is not obvious from the code

## Prefer Succinct Documentation

**Document what is NOT obvious. Skip what IS obvious.**

### When to Skip Docstrings Entirely

- **Trivial getters/setters**: `def get_name(self)` or `def set_name(self, name)` need no docstring
- **Simple property accessors**: Properties that just return an attribute
- **Obvious helper methods**: Private methods with self-explanatory names

### When to Skip the Args Section

Skip `Args:` when parameter names and types are self-descriptive:

```python
# GOOD - no Args needed, parameters are obvious
def connect(self, host: str, port: int, timeout: int = 30):
    """Establish connection to the remote server."""

# BAD - redundant documentation
def connect(self, host: str, port: int, timeout: int = 30):
    """Establish connection to the remote server.

    Args:
        host: The host to connect to.
        port: The port to connect to.
        timeout: The timeout in seconds.
    """
```

### When to Skip the Returns Section

Skip `Returns:` when the return value is obvious from the function name and inputs:

```python
# GOOD - return value is obvious
def get_user_by_id(self, user_id: int) -> User:
    """Fetch user from the database."""

def is_valid(self) -> bool:
    """Check if the configuration is valid."""

def count_active_jobs(self) -> int:
    """Count jobs currently in the active state."""

# BAD - redundant return documentation
def get_user_by_id(self, user_id: int) -> User:
    """Fetch user from the database.

    Returns:
        The user with the given ID.
    """
```

### When Documentation IS Needed

- Non-obvious behavior, side effects, or edge cases
- Complex algorithms or business logic
- Parameters with specific constraints or formats
- Return values that differ from what the name suggests
- Exceptions that callers should handle

```python
def retry_with_backoff(self, fn: Callable, max_attempts: int = 3) -> Any:
    """Execute function with exponential backoff on failure.

    Args:
        max_attempts: Must be >= 1. Backoff doubles after each failure,
            starting at 1 second.

    Returns:
        The return value of `fn` on success.

    Raises:
        RetryExhaustedError: If all attempts fail. Contains the last exception.

    """
```

## Prefer Examples Over Verbiage

**A good example is worth more than paragraphs of explanation.**

When behavior can be demonstrated with code, use doctest examples instead of lengthy prose. Use the `.. doctest::` directive for examples that should be validated by Sphinx.

### Doctest Format

```python
def parse_duration(duration_str: str) -> int:
    """Parse a duration string into seconds.

    Example:
        .. doctest::

            >>> parse_duration("1h30m")
            5400
            >>> parse_duration("2d")
            172800
            >>> parse_duration("45s")
            45

    """
```

### When to Prefer Examples

- **Format specifications**: Show input/output instead of describing the format
- **Edge cases**: Demonstrate behavior rather than explaining it
- **Complex return types**: Show actual output structure

```python
# GOOD - example shows exactly what to expect
def get_job_status(job_id: str) -> dict:
    """Query the status of a submitted job.

    Example:
        .. doctest::

            >>> get_job_status("job-123")
            {'state': 'RUNNING', 'progress': 0.75, 'workers': ['w1', 'w2']}

    """

# BAD - verbose description of the same thing
def get_job_status(job_id: str) -> dict:
    """Query the status of a submitted job.

    Returns:
        A dictionary containing:
        - 'state': A string representing the job state (PENDING, RUNNING, etc.)
        - 'progress': A float between 0 and 1 indicating completion percentage
        - 'workers': A list of worker identifiers assigned to this job

    """
```

## Cross-References

Use Sphinx cross-reference roles to create hyperlinks to other documented items. This produces clickable links in the generated documentation.

### Syntax

| Role | Usage | Example |
|------|-------|---------|
| `:py:mod:` | Module | `:py:mod:\`torchx.specs\`` |
| `:py:class:` | Class | `:py:class:\`AppDef\`` |
| `:py:func:` | Function | `:py:func:\`get_runner\`` |
| `:py:meth:` | Method | `:py:meth:\`Runner.run\`` |
| `:py:attr:` | Attribute | `:py:attr:\`AppDef.name\`` |
| `:py:exc:` | Exception | `:py:exc:\`InvalidAppError\`` |

### Examples in Docstrings

```python
def run_app(app: AppDef, scheduler: str = "local") -> AppHandle:
    """Submit an application for execution.

    Similar to :py:meth:`Runner.run` but uses the default runner.

    Args:
        app: The application to run. See :py:class:`AppDef` for details.
        scheduler: Scheduler name. See :py:mod:`torchx.schedulers` for options.

    Raises:
        :py:exc:`InvalidAppError`: If the app definition is malformed.

    See Also:
        :py:func:`get_runner`, :py:class:`Runner`

    """
```

### Shorthand References

For items in the same module, use `~` prefix to show only the final name:

```python
"""
Returns a :py:class:`~torchx.specs.AppDef` instance.
"""
# Renders as: "Returns a AppDef instance." (with link)
```

## Google Style Docstring Format

### Module Docstrings

```python
"""Short one-line summary of the module.

Longer description spanning multiple lines if needed. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using the ``Example`` or ``Examples`` sections::

        $ python example.py

Attributes:
    module_level_variable (int): Module level variables documented here
        or inline after the variable.

"""
```

### Function Docstrings

```python
def function_name(param1, param2=None, *args, **kwargs):
    """Short one-line summary of the function.

    Longer description if needed. Function parameters should be documented
    in the ``Args`` section.

    Args:
        param1 (int): The first parameter.
        param2 (str, optional): The second parameter. Defaults to None.
            Multi-line descriptions should be indented.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        bool: True if successful, False otherwise.

        The return description may span multiple lines and paragraphs.

    Raises:
        ValueError: If `param2` is equal to `param1`.
        AttributeError: Description of when this is raised.

    Example:
        >>> result = function_name(1, "test")
        >>> print(result)
        True

    """
```

### Function with PEP 484 Type Annotations

When using type annotations, types can be omitted from the docstring:

```python
def function_name(param1: int, param2: str) -> bool:
    """Short one-line summary.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True for success, False otherwise.

    """
```

### Generator Docstrings

Use `Yields` instead of `Returns`:

```python
def example_generator(n):
    """Short summary of the generator.

    Args:
        n (int): The upper limit of the range.

    Yields:
        int: The next number in the range.

    """
```

### Class Docstrings

```python
class ExampleClass:
    """Short one-line summary of the class.

    Longer description if needed. Public attributes may be documented here
    in an ``Attributes`` section.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (int, optional): Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Initialize the class.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (int, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (list[str]): Description of `param3`.

        """
```

### Method Docstrings

```python
def example_method(self, param1, param2):
    """Short summary of the method.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    """
```

### Property Docstrings

Document properties in their getter method:

```python
@property
def readonly_property(self) -> str:
    """str: Properties should be documented in their getter method."""
    return self._value

@property
def readwrite_property(self) -> list[str]:
    """list[str]: Properties with getter and setter.

    Document only in the getter. If the setter has notable behavior,
    mention it here.
    """
    return self._values
```

### Exception Docstrings

```python
class ExampleError(Exception):
    """Short summary of the exception.

    The __init__ method may be documented in either the class level
    docstring or as a docstring on the __init__ method itself.

    Args:
        msg (str): Human readable string describing the exception.
        code (int, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
```

## TorchX-Specific Patterns

The following patterns are derived from analyzing the TorchX codebase.

### Module Docstrings with Context

Module docstrings often provide context about how the module fits into the TorchX ecosystem, with links to external documentation and shell examples:

```python
"""
For distributed training, TorchX relies on the scheduler's gang scheduling
capabilities to schedule ``n`` copies of nodes. Once launched, the application
is expected to be written in a way that leverages this topology, for instance,
with PyTorch's
`DDP <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.

DDP Builtin
----------------

.. code:: shell-session

    # locally, 1 node x 4 workers
    $ torchx run -s local_cwd dist.ddp -j 1x4 --script main.py

    # remote (kubernetes)
    $ torchx run -s kubernetes -cfg queue=default dist.ddp \\
        -j 2x4 --script main.py

Components APIs
-----------------
"""
```

### Dataclass Docstrings

For dataclasses, use `Args:` to document constructor parameters (the fields):

```python
@dataclass
class Resource:
    """
    Represents resource requirements for a ``Role``.

    Args:
        cpu: number of logical cpu cores
        gpu: number of gpus
        memMB: MB of ram
        capabilities: additional hardware specs (interpreted by scheduler)

    Note: you should prefer to use named_resources instead of specifying the raw
    resource requirement directly.
    """

    cpu: int
    gpu: int
    memMB: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
```

### Component Function Docstrings

Component functions (used with `torchx run`) should lead with usage examples:

```python
def ddp(
    *script_args: str,
    script: Optional[str] = None,
    m: Optional[str] = None,
    image: str = torchx.IMAGE,
    j: str = "1x2",
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate PyTorch worker processes.

    Note: (cpu, gpu, memMB) parameters are mutually exclusive with ``h`` (named resource) where
          ``h`` takes precedence if specified for setting resource requirements.

    Args:
        script_args: arguments to the main module
        script: script or binary to run within the image
        m: the python module path to run
        image: image (e.g. docker)
        j: [{min_nnodes}:]{nnodes}x{nproc_per_node}, for gpu hosts, nproc_per_node must not exceed num gpus
    """
```

### Enum Docstrings

Document enum values with a numbered list in the class docstring:

```python
class AppState(int, Enum):
    """
    State of the application. An application starts from an initial
    ``UNSUBMITTED`` state and moves through ``SUBMITTED``, ``PENDING``,
    ``RUNNING`` states finally reaching a terminal state:
    ``SUCCEEDED``,``FAILED``, ``CANCELLED``.

    1. UNSUBMITTED - app has not been submitted to the scheduler yet
    2. SUBMITTED - app has been successfully submitted to the scheduler
    3. PENDING - app has been submitted to the scheduler pending allocation
    4. RUNNING - app is running
    5. SUCCEEDED - app has successfully completed
    6. FAILED - app has unsuccessfully completed
    7. CANCELLED - app was cancelled before completing
    """

    UNSUBMITTED = 0
    SUBMITTED = 1
    # ...
```

### Parameter Format Strings

Document format patterns inline with the parameter:

```python
"""
Args:
    j: [{min_nnodes}:]{nnodes}x{nproc_per_node}, for gpu hosts,
       nproc_per_node must not exceed num gpus
    name: job name in format: ``{experimentname}/{runname}`` or
          ``{experimentname}/`` or ``/{runname}``
    mounts: mounts in format: type=<bind/volume>,src=/host,dst=/job[,readonly]
"""
```

### Mutual Exclusivity Notes

Document mutually exclusive parameters with a `Note:` at the start:

```python
"""
Note: (cpu, gpu, memMB) parameters are mutually exclusive with ``h`` (named resource) where
      ``h`` takes precedence if specified for setting resource requirements.
      See `registering named resources <https://...>`_.

Args:
    cpu: number of cpus per replica
    gpu: number of gpus per replica
    memMB: cpu memory in MB per replica
    h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
"""
```

### Code Block Directives

Use `.. code-block:: python` for non-doctest code examples:

```python
"""
Usage:

.. code-block:: python

    runner.run_component("distributed.ddp", ...)

    # File-based component
    runner.run_component("~/home/components.py:my_component", ...)
"""
```

Use `.. code:: shell-session` for CLI examples:

```python
"""
.. code:: shell-session

    $ torchx run -s local_cwd dist.ddp -j 1x4 --script main.py
"""
```

### External Links

Use RST link syntax for external URLs:

```python
"""
Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
to launch and coordinate PyTorch worker processes.
"""
```

### Warning and Note Directives

Use RST directives for important callouts:

```python
"""
.. warning:: Macros used in fields of :py:class:`Role` other than the ones
             mentioned above, are NOT substituted.

.. note:: sub-classes of ``Runner`` should implement ``schedule`` method
          rather than overriding this method directly.
"""
```

### Inline Variable Documentation

Module-level variables can be documented with a docstring immediately after:

```python
_TORCH_DEBUG_FLAGS: Dict[str, str] = {
    "CUDA_LAUNCH_BLOCKING": "1",
    "NCCL_DESYNC_DEBUG": "1",
}
"""
These are commonly set environment variables to debug PyTorch execution.

* ``CUDA_LAUNCH_BLOCKING``: Read more `here <https://docs.nvidia.com/...>`__.
* ``TORCH_DISTRIBUTED_DEBUG``: Read more `here <https://pytorch.org/...>`__.
"""
```

## Section Reference

| Section | Purpose |
|---------|---------|
| `Args:` | Function/method parameters |
| `Returns:` | Return value description |
| `Yields:` | Generator yield description |
| `Raises:` | Exceptions that may be raised |
| `Attributes:` | Class/module attributes |
| `Example:` / `Examples:` | Usage examples (doctest format) |
| `Note:` / `Notes:` | Additional notes |
| `Warning:` / `Warnings:` | Warning information |
| `See Also:` | Related functions/classes |
| `Todo:` | Future improvements |

## Key Rules

1. **Be succinct**: Document only what is not obvious from names and types
2. **Skip trivial docstrings**: Getters, setters, and obvious methods need none
3. **Skip obvious sections**: Omit Args/Returns when self-explanatory
4. **Prefer examples**: Use `.. doctest::` examples over verbose prose
5. **Use cross-references**: Link to classes/functions with `:py:class:`, `:py:func:`, etc.
6. **First line**: Always a short imperative summary on one line
7. **Blank line**: Separate summary from body with a blank line
8. **Indentation**: Use 4 spaces for continuation lines in sections
9. **Types**: Include types in parentheses after parameter names, OR use PEP 484 annotations (not both)
10. **Optional params**: Mark with `optional` after the type
11. **Defaults**: Document default values with "Defaults to X" only if non-obvious
12. **Self/cls**: Never include `self` or `cls` in the Args section
13. **Dataclasses**: Use `Args:` for fields, not `Attributes:`
14. **Format strings**: Document parameter formats inline (e.g., `j: {nnodes}x{nproc_per_node}`)
15. **External links**: Use RST syntax: `` `text <url>`_ ``
16. **Shell examples**: Use `.. code:: shell-session` with `$` prefix

## Reference

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Sphinx Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [TorchX Documentation](https://pytorch.org/torchx/)
