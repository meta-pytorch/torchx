# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import re
import sys
import threading
from dataclasses import dataclass, replace
from inspect import getmembers, isfunction
from pathlib import Path
from types import ModuleType
from typing import Callable, Generator

from torchx.specs import AppDef
from torchx.specs.file_linter import (
    ComponentFunctionValidator,
    get_fn_docstring,
    validate,
)
from torchx.util import entrypoints
from torchx.util.io import read_conf_file
from torchx.util.types import none_throws

logger: logging.Logger = logging.getLogger(__name__)


class ComponentValidationException(Exception):
    pass


class ComponentNotFoundException(Exception):
    pass


@dataclass
class _Component:
    """
    Definition of the component

    Args:
        name: The name of the component, which usually MODULE_PATH.FN_NAME
        description: The description of the component, taken from the desrciption
            of the function that creates component. In case of no docstring, description
            will be the same as name
        fn_name: Function name that creates component
        fn: Function that creates component
        validation_errors: Validation errors
    """

    name: str
    description: str
    fn_name: str

    fn: Callable[..., AppDef]

    validation_errors: list[str]


class ComponentsFinder(abc.ABC):
    @abc.abstractmethod
    def find(
        self, validators: list[ComponentFunctionValidator] | None
    ) -> list[_Component]:
        """
        Retrieves a set of components. A component is defined as a python
        function that conforms to ``torchx.specs.file_linter`` linter.

        Returns:
            List of components
        """


def is_namespace_package(module: ModuleType) -> bool:
    """
    Returns:
        Whether the ``module`` is a
        `namespace package <https://packaging.python.org/en/latest/guides/packaging-namespace-packages/>`_.

    """
    # namespace package modules have no or empty __file__ attribute
    return (not hasattr(module, "__file__")) or (module.__file__ is None)


def is_package(module: ModuleType) -> bool:
    """
    Note that this function returns ``True`` if ``module`` is either a
    regular (has an ``__init__.py`` file) or namespace package (does not have an ``__init__.py`` file).
    To disambiguate between a regular and namespace package use :py:func:`is_namespace_package`.

    Returns:
        Whether the ``module`` is a python module (maps to a python file) or a package
        (maps to a dir with an ``__init__.py`` file).

    """
    # packages have the __path__ attribute set
    # see https://docs.python.org/3/tutorial/modules.html#packages-in-multiple-directories
    return hasattr(module, "__path__")


def module_relname(module: ModuleType, relative_to: ModuleType) -> str:
    """
    Example:

        .. doctest::

            >>> from torchx.specs.finder import module_relname
            >>> import torchx.components as c
            >>> import torchx.components.dist as d

            >>> module_relname(d, relative_to=c)
            'dist'

            >>> module_relname(d, relative_to=d)
            ''

            >>> module_relname(c, relative_to=d)
            Traceback (most recent call last):
            ...
            ValueError: `torchx.components` is not a submodule of `torchx.components.dist`

    Returns:
        The ``module``'s name relative to the ``relative_to`` module.

    Raises:
        ValueError: if ``module`` is not a submodule of ``relative_to``
    """

    # use pathlib.Path's relative_to function by converting the module name to a path, then back
    modname = module.__name__
    reltoname = relative_to.__name__
    if modname == reltoname:
        return ""

    p = Path(modname.replace(".", os.sep))
    rp = Path(reltoname.replace(".", os.sep))
    return str(p.relative_to(rp)).replace(os.sep, ".")


class ModuleComponentsFinder(ComponentsFinder):
    """Retrieves components from the directory associated with module.

    Finds all components in the given module and submodules in a recursive manner.
    The ``module`` can be specified as a string (e.g. ``foo.bar``) or as a loaded module.

    If a non-empty ``group`` is passed, then the module name is replaced with the ``group``.
    This can be used to either alias the component name different to the component's function name
    or to group the components into an arbitrary logical namespace.

    For example, for the following directory structure:

    ::

      foo/
       |- __init__.py
       |- bar/
           |- __init__.py
           |- baz.py


    Where ``baz.py`` defines the component ``echo`` as such:

    ::

      # contents of baz.py
      def echo(msg: str) -> AppDef:
        ...

    Then depending on the ``module`` and ``group`` params the component ``echo`` is named as:

    1. ``ModuleComponentsFinder(module="foo.bar", group="")`` -> ``baz.echo``
    1. ``ModuleComponentsFinder(module="foo.bar", group="abc")`` -> ``abc.echo``
    1. ``ModuleComponentsFinder(module="foo.bar.baz", group="")`` -> ``echo``
    1. ``ModuleComponentsFinder(module="foo.bar.baz", group="my_echo")`` -> ``my_echo``

    """

    def __init__(self, module: str | ModuleType, group: str) -> None:
        self.base_module: ModuleType = self._try_import(module)
        self.group = group

    def _iter_modules_recursive(
        self, module: str | ModuleType
    ) -> Generator[ModuleType, None, None]:
        """
        Given a module name (e.g. "a.b") recursively finds and loads the sub-modules and itself
        as a generator.
        """

        # load itself first only if it is a package or module but not namespace
        module = self._try_import(module)
        if not is_namespace_package(module):
            yield module

        # module may be a module or a package
        # only recurse if the module_name is a package
        if is_package(module):
            # recurse through the sub-modules
            for module_info in pkgutil.iter_modules(
                module.__path__, prefix=f"{module.__name__}."
            ):
                if module_info.ispkg:
                    for submodule in self._iter_modules_recursive(module_info.name):
                        yield submodule
                else:
                    yield self._try_import(module_info.name)

    def find(
        self, validators: list[ComponentFunctionValidator] | None
    ) -> list[_Component]:
        components = []
        for m in self._iter_modules_recursive(self.base_module):
            components += self._get_components_from_module(m, validators)
        return components

    def _try_import(self, module: str | ModuleType) -> ModuleType:
        """
        If the module is a module name (e.g. ``"foo.bar"``) as a string, then this function
        imports the module and returns the loaded module. If it is already a module type then
        it just returns the module.
        """

        if isinstance(module, str):
            return importlib.import_module(module)
        else:
            return module

    def _get_components_from_module(
        self, module: ModuleType, validators: list[ComponentFunctionValidator] | None
    ) -> list[_Component]:
        functions = getmembers(module, isfunction)
        component_defs = []

        module_path = module.__file__
        assert module_path, f"module must have __file__: {module_path}"
        module_path = os.path.abspath(module_path)
        rel_module_name = module_relname(module, relative_to=self.base_module)
        for function_name, function in functions:
            linter_errors = validate(module_path, function_name, validators)
            component_desc, _ = get_fn_docstring(function)

            # remove empty string to deal with group=""
            component_name = ".".join(
                [p for p in [self.group, rel_module_name, function_name] if p]
            )
            component_def = _Component(
                name=component_name,
                description=component_desc,
                fn_name=function_name,
                fn=function,
                validation_errors=[
                    linter_error.description for linter_error in linter_errors
                ],
            )
            component_defs.append(component_def)
        return component_defs


def _package_identity(filepath: str) -> tuple[str, str] | None:
    """
    Derives the dotted module name of the python file at ``filepath`` from its
    enclosing package (the chain of parent directories carrying ``__init__.py``).

    Returns:
        ``(module_name, sys_path_root)`` where ``sys_path_root`` is the directory
        that must be on ``sys.path`` for ``module_name`` (and absolute imports of
        the file's package siblings) to be importable.
        ``None`` if ``filepath`` is not part of a package. A dir or stem whose
        name is not a valid identifier cannot appear in a dotted import, so it
        bounds the walk.
    """
    stem = Path(filepath).stem
    if not stem.isidentifier():
        return None
    directory = os.path.dirname(filepath)
    parts = [stem]
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        basename = os.path.basename(directory)
        if not basename.isidentifier():
            break
        parent = os.path.dirname(directory)
        parts.append(basename)
        if parent == directory:
            break
        directory = parent
    if len(parts) == 1:
        return None
    return ".".join(reversed(parts)), directory


def _is_same_file(path1: str, path2: str) -> bool:
    try:
        return os.path.samefile(path1, path2)
    except OSError:
        return False


# RLock: a loading component file can itself trigger a same-thread load
_LOAD_LOCK = threading.RLock()


def _is_module_path(path: str) -> bool:
    """
    Whether ``path`` reads as a dotted module path (``foo.bar.baz``) rather
    than a file path. A ``.py`` suffix or a path separator makes it a file path.
    """
    return not path.endswith(".py") and all(
        part.isidentifier() for part in path.split(".")
    )


def _names_missing_module(module_path: str, e: ModuleNotFoundError) -> bool:
    """
    Whether ``e`` reports ``module_path`` itself (or one of its ancestor
    packages) as missing, as opposed to an import failing *inside* an
    existing module (which is the caller's bug and must propagate as-is).
    """
    missing = e.name
    return bool(missing) and (
        module_path == missing or module_path.startswith(f"{missing}.")
    )


def _exec_src_as_module(filepath: str) -> ModuleType:
    """
    Loads a component file that does not exist on the local filesystem but whose
    source is resolvable through ``read_conf_file`` (e.g. served by a
    ``torchx.file`` entrypoint) by exec-ing the source into a synthetic module.
    """
    abspath = os.path.abspath(filepath)
    # the path digest disambiguates distinct paths that collapse to the same
    # name under the non-word substitution (`/a/b-c.py` vs `/a/b_c.py`)
    modname = (
        "torchx_component_file_"
        + re.sub(r"\W", "_", abspath).strip("_")
        + "_"
        + hashlib.blake2b(abspath.encode(), digest_size=4).hexdigest()
    )
    with _LOAD_LOCK:
        existing = sys.modules.get(modname)
        if existing is not None:
            return existing

        file_source = read_conf_file(filepath)
        module = ModuleType(modname)
        module.__file__ = abspath
        sys.modules[modname] = module
        try:
            exec(compile(file_source, abspath, "exec"), module.__dict__)  # noqa: P204
        except BaseException:
            sys.modules.pop(modname, None)
            raise
        return module


def _load_file_as_module(filepath: str) -> ModuleType:
    """
    Loads the python file at ``filepath`` as a regular module, mirroring how the
    interpreter itself would load it:

    #. a file inside a package (parent dirs carry ``__init__.py``) loads under
       its dotted module name with the package root's parent dir appended to
       ``sys.path`` (as with ``python -m pkg.mod``), so absolute imports of the
       file's package siblings resolve;
    #. a standalone file loads under its stem with its directory appended to
       ``sys.path`` (as with ``python file.py``), so imports of neighboring
       modules resolve;
    #. a path not present on the local filesystem falls back to
       :py:func:`_exec_src_as_module`.

    The module is registered in ``sys.modules``, so functions and classes it
    defines carry their real ``__module__`` (instead of the finder's) and
    machinery that resolves ``sys.modules[obj.__module__]`` (``typing``,
    ``dataclasses``, ``pickle``) works on them. For a package file the parent
    package is imported first (as a real import would), so relative imports
    inside the component file resolve; if the package name is shadowed by a
    different root earlier on ``sys.path``, a warning names both roots.
    Loading the same file again returns the already-loaded module.

    Limitation: when the derived module name is taken by a DIFFERENT file
    (e.g. the same dotted name reachable from another ``sys.path`` root), the
    module loads under a ``_<n>``-suffixed name. That name resolves through
    ``sys.modules`` but is not importable by the import system, so
    re-resolution that round-trips through an import (e.g. unpickling in a
    fresh process) does not work for such modules.
    """
    abspath = os.path.abspath(filepath)
    if not os.path.isfile(abspath):
        return _exec_src_as_module(filepath)

    identity = _package_identity(abspath)
    if identity:
        modname, sys_path_root = identity
    else:
        modname, sys_path_root = Path(abspath).stem, os.path.dirname(abspath)

    base_modname = modname
    with _LOAD_LOCK:
        collision = 0
        while (existing := sys.modules.get(modname)) is not None:
            existing_file = getattr(existing, "__file__", None)
            if existing_file and _is_same_file(existing_file, abspath):
                return existing
            collision += 1
            modname = f"{base_modname}_{collision}"

        spec = importlib.util.spec_from_file_location(modname, abspath)
        if spec is None or spec.loader is None:
            raise ComponentNotFoundException(
                f"cannot create a module spec for `{abspath}`;"
                " make sure the file is a regular python source file"
            )
        loader = spec.loader
        module = importlib.util.module_from_spec(spec)

        appended_sys_path = sys_path_root not in sys.path
        if appended_sys_path:
            sys.path.append(sys_path_root)
        parent_name, _, leaf_name = modname.rpartition(".")
        parent: ModuleType | None = None
        sys.modules[modname] = module
        try:
            if parent_name:
                parent = importlib.import_module(parent_name)
                parent_dir = os.path.dirname(getattr(parent, "__file__", "") or "")
                if parent_dir and not _is_same_file(
                    parent_dir, os.path.dirname(abspath)
                ):
                    logger.warning(
                        "package `%s` resolves to `%s` (earlier on sys.path),"
                        " which shadows this component file's own package root"
                        " `%s`; sibling imports inside `%s` will resolve"
                        " against the shadowing package",
                        parent_name,
                        parent_dir,
                        os.path.dirname(abspath),
                        abspath,
                    )
                    parent = None
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(modname, None)
            if appended_sys_path and sys_path_root in sys.path:
                sys.path.remove(sys_path_root)
            raise
        if parent is not None:
            setattr(parent, leaf_name, module)
        return module


class CustomComponentsFinder(ComponentsFinder):
    """
    Finds a single component addressed as ``PATH:FUNCTION_NAME``, where ``PATH``
    is either a path to a python file (``path/to/comp.py:fn``) or a dotted
    module path (``pkg.module:fn``). A ``PATH`` that exists as a file wins over
    the module interpretation; the component must be defined in the named
    file/module (not merely imported into it).
    """

    def __init__(self, filepath: str, function_name: str) -> None:
        self._filepath = filepath
        self._function_name = function_name

    def _get_validation_errors(
        self,
        path: str,
        function_name: str,
        validators: list[ComponentFunctionValidator] | None,
    ) -> list[str]:
        linter_errors = validate(path, function_name, validators)
        return [linter_error.description for linter_error in linter_errors]

    def _load(self) -> tuple[ModuleType, str]:
        """Loads the addressed file/module; returns it with its validation path.

        A missing target module raises :py:class:`ComponentNotFoundException`;
        a missing import *inside* an existing module propagates as-is.
        """
        if not os.path.isfile(self._filepath) and _is_module_path(self._filepath):
            try:
                module = importlib.import_module(self._filepath)
            except ModuleNotFoundError as e:
                if _names_missing_module(self._filepath, e):
                    raise ComponentNotFoundException(
                        f"Module `{self._filepath}` not found on the python path"
                    ) from e
                raise
            module_file = module.__file__
            if module_file is None:
                raise ComponentNotFoundException(
                    f"`{self._filepath}` is a namespace package, not a module;"
                    " components must be addressed by the module that defines them"
                )
            return module, module_file
        return _load_file_as_module(self._filepath), self._filepath

    def find(
        self, validators: list[ComponentFunctionValidator] | None
    ) -> list[_Component]:
        module, validation_path = self._load()
        validation_errors = self._get_validation_errors(
            validation_path, self._function_name, validators
        )

        if self._function_name not in vars(module):
            raise ComponentNotFoundException(
                f"Function {self._function_name} does not exist in {self._filepath}"
            )
        app_fn = getattr(module, self._function_name)
        fn_desc, _ = get_fn_docstring(app_fn)
        return [
            _Component(
                name=f"{self._filepath}:{self._function_name}",
                description=fn_desc,
                fn_name=self._function_name,
                fn=app_fn,
                validation_errors=validation_errors,
            )
        ]


def _load_custom_components(
    validators: list[ComponentFunctionValidator] | None,
) -> list[_Component]:
    component_modules = {
        name: load_fn()
        for name, load_fn in
        # load_group() defers the module load so you have to call
        # the deferred load_fn to actually load the module
        entrypoints.load_group("torchx.components").items()
    }

    components: list[_Component] = []
    for group, module in component_modules.items():
        # using "_" prefix for entrypoint name allows users to
        # specify component names without a prefix
        # we use "_" since this is consistent with ignored function params in python
        # e.g.
        # [torchx.components]
        # _0 = torchx.components.dist
        # _1 = torchx.components.utils
        assert isinstance(module, (ModuleType, str)), (
            f"the `{group}` entry point in group `torchx.components` must load"
            f" a module or a module name, got {type(module).__name__}"
        )
        group = "" if group.startswith("_") else group
        components += ModuleComponentsFinder(module, group).find(validators)
    return components


def _load_components(
    validators: list[ComponentFunctionValidator] | None,
) -> dict[str, _Component]:
    """
    Loads either the custom component defs from the entrypoint ``[torchx.components]``
    or the default builtins from ``torchx.components`` module.

    .. note::
        If the custom components exist then, the default builtins are not loaded
        since the user can add the ones from ``torchx.components`` in their entrypoint

    """

    components = _load_custom_components(validators)
    if not components:
        components = ModuleComponentsFinder("torchx.components", "").find(validators)
    return {c.name: c for c in components}


_components: dict[str, _Component] | None = None


def _find_components(
    validators: list[ComponentFunctionValidator] | None,
) -> dict[str, _Component]:
    global _components
    if not _components:
        _components = _load_components(validators)
    return none_throws(_components)


def _is_custom_component(component_name: str) -> bool:
    return ":" in component_name


def _find_custom_components(
    name: str, validators: list[ComponentFunctionValidator] | None
) -> dict[str, _Component]:
    if ":" not in name:
        raise ValueError(
            f"Invalid custom component: {name}, valid template : `FILEPATH`:`FUNCTION_NAME`"
        )
    filepath, component_name = name.split(":")
    components = CustomComponentsFinder(filepath, component_name).find(validators)
    return {component.name: component for component in components}


def _find_module_components(
    name: str, validators: list[ComponentFunctionValidator] | None
) -> dict[str, _Component] | None:
    """
    Resolves a colon-less dotted ``name`` (``pkg.module.fn``) as the component
    function ``fn`` in module ``pkg.module``. Returns ``None`` only when
    ``name`` does not read as a dotted path or the module does not exist; a
    module that exists but lacks ``fn``, a namespace-package target, and a
    missing import *inside* an existing module all raise with their specific
    error (matching what the colon form ``pkg.module:fn`` reports).
    """
    module_path, _, function_name = name.rpartition(".")
    if not module_path or not _is_module_path(module_path):
        return None
    try:
        if importlib.util.find_spec(module_path) is None:
            return None
    except ValueError:
        return None
    except ModuleNotFoundError as e:
        if _names_missing_module(module_path, e):
            return None
        raise
    (component,) = CustomComponentsFinder(module_path, function_name).find(validators)
    return {name: replace(component, name=name)}


def get_components(
    validators: list[ComponentFunctionValidator] | None = None,
) -> dict[str, _Component]:
    """
    Returns all custom components registered via ``[torchx.components]`` entrypoints
    OR builtin components that ship with TorchX (but not both).

    When registering custom components via entrypoints, each line is a key-value pair:

    ::

      [torchx.components]
      foo = test.bar
      hello = test.world


    Where ``test.bar`` is a valid path to the python module and ``foo`` is
    the prefix alias for all the components found in the module ``test.bar``.
    TorchX recursively finds all components
    (functions that return :py:class:`torchx.specs.AppDef`) in the given module.

    In the example above, components found in the ``test.bar`` module (and its
    sub-modules) will have the name ``foo.<component_fn_name>``, where ``<component_fn_name>``
    is the path to the component function relative to the registered base module.
    Similarly, components found in ``test.world`` will have the ``hello.`` prefix in their names.

    .. note::
        TorchX will NOT recurse through sub-namespace packages!
        Make sure to drop an ``__init__.py`` to have TorchX discover components
        recursively, or explicitly map the namespace packages in ``[torchx.components]``
        section of your entrypoints.

    If no ``[torchx.components]`` have been registered by the user, then this function
    load the builtin components, which is equivalent to loading:

    ::

      [torchx.components]
      dist = torchx.components.dist
      util = torchx.components.util
      # ... and so on for all modules in torchx.components.* ...


    Returns:
        Components in a format : {ALIAS: LIST_OF_COMPONENTS}

    """

    valid_components: dict[str, _Component] = {}
    for component_name, component in _find_components(validators).items():
        if len(component.validation_errors) == 0:
            valid_components[component_name] = component
    return valid_components


def get_component(
    name: str, validators: list[ComponentFunctionValidator] | None = None
) -> _Component:
    """
    Retrieves components by the provided name, which is one of:

    #. a registered component name (builtin or ``[torchx.components]``
       entrypoint), e.g. ``utils.echo``
    #. a path to a python file and a function in it, e.g. ``path/to/comp.py:fn``
    #. a dotted module path and a function in it, e.g. ``pkg.module:fn``
       (equivalently ``pkg.module.fn`` when no registered component has
       that name)

    Returns:
        The component with the given ``name``.

    Raises:
        ComponentNotFoundException: if no component with ``name`` exists
    """
    if _is_custom_component(name):
        components = _find_custom_components(name, validators)
    else:
        components = _find_components(validators)
        if name not in components:
            components = _find_module_components(name, validators) or components
    if name not in components:
        raise ComponentNotFoundException(
            f"Component `{name}` not found. Please make sure it is one of the "
            "builtins: `torchx builtins`. Or registered via `[torchx.components]` "
            "entry point (see: https://meta-pytorch.org/torchx/latest/configure.html). "
            "Or addressable as `path/to/file.py:fn` or `pkg.module:fn`"
        )

    component = components[name]
    if len(component.validation_errors) > 0:
        validation_msg = "\n".join(component.validation_errors)
        raise ComponentValidationException(
            f"Component {name} has validation errors: \n {validation_msg}"
        )
    return component


def get_builtin_source(
    name: str, validators: list[ComponentFunctionValidator] | None = None
) -> str:
    """
    Returns a string of the the builtin component's function source code
    with all the import statements. Intended to be used to make a copy
    of the builtin component to use as a template for further customization.

    For simplicity import statements are read literally from the python file
    where the builtin component is defined. All lines that start with
    "import " and "from " preceding the function declaration
    (e.g. ``def builtin_name(...):`` are considered necessary import statements
    and hence included in the returned string.

    Therefore, it is possible to get additional unused import statements,
    which can happen if multiple builtins are defined in the same file.
    Make sure to pass the copy through a linter so that import statements
    are optimized and formatting adheres to your organization's standards.
    """

    component = get_component(name, validators)
    fn = component.fn
    fn_name = component.name.split(".")[-1]

    # grab only the literal import statements BEFORE the builtin function def
    with open(inspect.getfile(component.fn), "r") as f:
        import_stmts = []
        for line in f.readlines():
            if line.startswith("import ") or line.startswith("from "):
                import_stmts.append(line.rstrip("\n"))
            elif line.startswith(f"def {fn_name}("):
                break

    fn_src = inspect.getsource(fn)

    return "\n".join([*import_stmts, "\n", fn_src, "\n"])
