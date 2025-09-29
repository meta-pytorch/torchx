# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import copy
import importlib
import inspect
import logging
import os
import pkgutil
from dataclasses import dataclass
from inspect import getmembers, isfunction
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Generator, List, Optional, Union

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

    validation_errors: List[str]


class ComponentsFinder(abc.ABC):
    @abc.abstractmethod
    def find(
        self, validators: Optional[List[ComponentFunctionValidator]]
    ) -> List[_Component]:
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

    def __init__(self, module: Union[str, ModuleType], group: str) -> None:
        self.base_module: ModuleType = self._try_import(module)
        self.group = group

    def _iter_modules_recursive(
        self, module: Union[str, ModuleType]
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
        self, validators: Optional[List[ComponentFunctionValidator]]
    ) -> List[_Component]:
        components = []
        for m in self._iter_modules_recursive(self.base_module):
            components += self._get_components_from_module(m, validators)
        return components

    def _try_import(self, module: Union[str, ModuleType]) -> ModuleType:
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
        self, module: ModuleType, validators: Optional[List[ComponentFunctionValidator]]
    ) -> List[_Component]:
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


class CustomComponentsFinder(ComponentsFinder):
    def __init__(self, filepath: str, function_name: str) -> None:
        self._filepath = filepath
        self._function_name = function_name

    def _get_validation_errors(
        self,
        path: str,
        function_name: str,
        validators: Optional[List[ComponentFunctionValidator]],
    ) -> List[str]:
        linter_errors = validate(path, function_name, validators)
        return [linter_error.description for linter_error in linter_errors]

    def find(
        self, validators: Optional[List[ComponentFunctionValidator]]
    ) -> List[_Component]:
        validation_errors = self._get_validation_errors(
            self._filepath, self._function_name, validators
        )

        file_source = read_conf_file(self._filepath)
        namespace = copy.copy(globals())
        # so that __file__ used inside the component points to the correct file
        namespace["__file__"] = os.path.abspath(self._filepath)
        exec(file_source, namespace)  # noqa: P204
        if self._function_name not in namespace:
            raise ComponentNotFoundException(
                f"Function {self._function_name} does not exist in file {self._filepath}"
            )
        app_fn = namespace[self._function_name]
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
    validators: Optional[List[ComponentFunctionValidator]],
) -> List[_Component]:
    component_modules = {
        name: load_fn()
        for name, load_fn in
        # load_group() defers the module load so you have to call
        # the deferred load_fn to actually load the module
        entrypoints.load_group("torchx.components", default={}).items()
    }

    components: List[_Component] = []
    for group, module in component_modules.items():
        # using "_" prefix for entrypoint name allows users to
        # specify component names without a prefix
        # we use "_" since this is consistent with ignored function params in python
        # e.g.
        # [torchx.components]
        # _0 = torchx.components.dist
        # _1 = torchx.components.utils
        group = "" if group.startswith("_") else group
        components += ModuleComponentsFinder(module, group).find(validators)
    return components


def _load_components(
    validators: Optional[List[ComponentFunctionValidator]],
) -> Dict[str, _Component]:
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


_components: Optional[Dict[str, _Component]] = None


def _find_components(
    validators: Optional[List[ComponentFunctionValidator]],
) -> Dict[str, _Component]:
    global _components
    if not _components:
        _components = _load_components(validators)
    return none_throws(_components)


def _is_custom_component(component_name: str) -> bool:
    return ":" in component_name


def _find_custom_components(
    name: str, validators: Optional[List[ComponentFunctionValidator]]
) -> Dict[str, _Component]:
    if ":" not in name:
        raise ValueError(
            f"Invalid custom component: {name}, valid template : `FILEPATH`:`FUNCTION_NAME`"
        )
    filepath, component_name = name.split(":")
    components = CustomComponentsFinder(filepath, component_name).find(validators)
    return {component.name: component for component in components}


def get_components(
    validators: Optional[List[ComponentFunctionValidator]] = None,
) -> Dict[str, _Component]:
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

    valid_components: Dict[str, _Component] = {}
    for component_name, component in _find_components(validators).items():
        if len(component.validation_errors) == 0:
            valid_components[component_name] = component
    return valid_components


def get_component(
    name: str, validators: Optional[List[ComponentFunctionValidator]] = None
) -> _Component:
    """
    Retrieves components by the provided name.

    Returns:
        Component or None if no component with ``name`` exists
    """
    if _is_custom_component(name):
        components = _find_custom_components(name, validators)
    else:
        components = _find_components(validators)
    if name not in components:
        raise ComponentNotFoundException(
            f"Component `{name}` not found. Please make sure it is one of the "
            "builtins: `torchx builtins`. Or registered via `[torchx.components]` "
            "entry point (see: https://meta-pytorch.org/torchx/latest/configure.html)"
        )

    component = components[name]
    if len(component.validation_errors) > 0:
        validation_msg = "\n".join(component.validation_errors)
        raise ComponentValidationException(
            f"Component {name} has validation errors: \n {validation_msg}"
        )
    return component


def get_builtin_source(
    name: str, validators: Optional[List[ComponentFunctionValidator]] = None
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
