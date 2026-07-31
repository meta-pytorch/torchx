# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[3, 2, 16]

"""Central plugin discovery, caching, and diagnostics.

This is an internal module — import from :py:mod:`torchx.plugins` instead.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib
import logging
import os
import pathlib
import pkgutil
from enum import auto
from types import ModuleType
from typing import Any, Callable, overload

from torchx.util import entrypoints

logger: logging.Logger = logging.getLogger(__name__)

#: Module-level attribute where named-resource registrations are stored.
NAMED_RESOURCES_ATTR: str = "NAMED_RESOURCES"


class PluginType(str, enum.Enum):
    """Type of TorchX plugin.

    Values are the entry-point group names (e.g. ``"torchx.schedulers"``).

    This enum is used internally to tag decorated functions and to filter
    plugins during discovery.  It is part of the public API only to support
    advanced subclassing of :py:class:`register`.
    """

    SCHEDULER = "torchx.schedulers"
    NAMED_RESOURCE = "torchx.named_resources"
    TRACKER = "torchx.tracker"


class PluginSource(enum.IntFlag):
    """Bitmask of plugin discovery channels.

    Combine with ``|`` and test with ``in``.  The value of the
    ``TORCHX_PLUGINS_SOURCE`` env var is parsed as the integer
    representation of this flag.
    """

    NONE = 0
    NAMESPACE_PKG = auto()
    ENTRYPOINT = auto()


# ── Public API ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RegistrationError:
    """An error encountered during plugin registration or discovery.

    Import-level errors (module fails to import) only have *module* and
    *error*.  Plugin-level errors (type mismatch, duplicate name)
    additionally carry *name* and *plugin_type*.
    """

    module: str
    error: str
    name: str | None = None
    plugin_type: str | None = None


class PluginRegistry:
    """Immutable, lazily-populated plugin registry.

    Created by :py:func:`registry`.  Each plugin group is discovered on first
    access via :py:meth:`get` and cached for subsequent calls.

    Usage::

        from torchx import plugins

        reg = plugins.registry()
        scheds = reg.get(plugins.PluginType.SCHEDULER)
        print(reg)

    Discovery channels are selected via *plugin_sources*, a :py:class:`PluginSource`
    bitmask.  Defaults to all channels enabled; pass :py:attr:`PluginSource.NONE`
    for an empty registry or any combination of ``NAMESPACE_PKG`` / ``ENTRYPOINT``
    to enable a subset.

    Args:
        plugin_sources: Bitmask of discovery channels to enable.
            Defaults to ``NAMESPACE_PKG | ENTRYPOINT``.
    """

    def __init__(
        self,
        *,
        plugin_sources: PluginSource = (
            PluginSource.NAMESPACE_PKG | PluginSource.ENTRYPOINT
        ),
    ) -> None:
        self._plugin_sources: PluginSource = plugin_sources
        # pyre-ignore[4]: plugin factories are heterogeneously typed
        self._cache: dict[PluginType, dict[str, Callable[..., Any]]] = {}
        self._errors: list[RegistrationError] = []

    # ── Namespace discovery ──────────────────────────────────────────────

    @staticmethod
    def _namespace_for_type(pt: PluginType) -> str:
        """Derive the ``torchx_plugins.*`` package name for *pt*.

        >>> PluginRegistry._namespace_for_type(PluginType.SCHEDULER)
        'torchx_plugins.schedulers'
        """
        return f"torchx_plugins.{pt.value.removeprefix('torchx.')}"

    def _collect_plugins_from_module(
        self,
        mod: ModuleType,
        fqn: str,
        expected_type: PluginType | None,
        discovered: dict[str, Any],
    ) -> None:
        """Collect ``@register``-decorated plugins from a single module.

        Any callable with a ``_plugin_type`` attribute whose ``__module__``
        matches *fqn* is collected into *discovered*.  Filters out
        re-imported symbols and validates that the plugin type matches
        *expected_type* (when set).
        """
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if not callable(obj):
                continue
            actual_type = getattr(obj, "_plugin_type", None)
            if actual_type is None:
                continue
            if getattr(obj, "__module__", None) != fqn:
                continue
            plugin_name = getattr(obj, "_plugin_name", attr_name)
            if expected_type is not None and actual_type != expected_type:
                msg = (
                    f"is a {actual_type.name.lower()} but is under"
                    f" the {expected_type.name.lower()} namespace"
                    f" — use @register.{expected_type.name.lower()}()"
                    f" or move to `{self._namespace_for_type(actual_type)}`"
                )
                logger.warning("`%s` in `%s` %s", plugin_name, fqn, msg)
                self._errors.append(
                    RegistrationError(
                        name=plugin_name,
                        module=fqn,
                        plugin_type=actual_type.name.lower(),
                        error=msg,
                    )
                )
                continue
            if plugin_name in discovered:
                msg = "duplicate — already discovered, keeping first occurrence"
                logger.warning("duplicate plugin `%s` in `%s`", plugin_name, fqn)
                self._errors.append(
                    RegistrationError(
                        name=plugin_name,
                        module=fqn,
                        plugin_type=(expected_type or actual_type).name.lower(),
                        error=msg,
                    )
                )
                continue
            discovered[plugin_name] = obj

    def _scan_namespace_pkg(
        self,
        pkg: ModuleType,
        namespace: str,
        expected_type: PluginType | None,
        discovered: dict[str, Any],
    ) -> None:
        """Recursively scan *pkg* for plugin submodules, populating *discovered*.

        Public submodules (not starting with ``_``) are imported.  Sub-packages
        are recursed into automatically (e.g. ``msl/``, ``fair/``).

        After ``pkgutil.iter_modules``, also probes for **implicit namespace
        sub-packages** — directories without ``__init__.py`` that contain
        ``.py`` files.  This lets namespace packages like
        ``torchx_plugins.named_resources`` host subdirectories without
        requiring ``__init__.py`` in each.

        **Fault tolerance**: any exception raised during submodule import
        is logged as a warning and the offending module is skipped.
        """
        try:
            module_iter = list(pkgutil.iter_modules(pkg.__path__))
        except Exception as e:
            msg = f"failed to scan `{namespace}`: {e}"
            logger.warning("%s", msg)
            self._errors.append(RegistrationError(module=namespace, error=msg))
            return

        found_names: set[str] = set()
        for module_info in module_iter:
            name = module_info.name
            found_names.add(name)
            if name.startswith("_"):
                continue
            fqn = f"{namespace}.{name}"
            try:
                mod = importlib.import_module(fqn)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                logger.warning("failed to import `%s`: %s", fqn, e)
                self._errors.append(RegistrationError(module=fqn, error=msg))
                continue

            self._collect_plugins_from_module(mod, fqn, expected_type, discovered)

            if module_info.ispkg:
                self._scan_namespace_pkg(mod, fqn, expected_type, discovered)

        # Discover implicit namespace sub-packages (directories without
        # __init__.py that contain .py files).  pkgutil.iter_modules misses
        # these because it only recognises directories with __init__.py as
        # packages.
        self._scan_implicit_subpackages(
            pkg, namespace, expected_type, discovered, found_names
        )

    def _scan_implicit_subpackages(
        self,
        pkg: ModuleType,
        namespace: str,
        expected_type: PluginType | None,
        discovered: dict[str, Any],
        already_found: set[str],
    ) -> None:
        """Discover subdirectories that are implicit namespace packages.

        Walks every directory on *pkg.__path__*, looking for child
        directories that:

        1. Are not private (no leading ``_``).
        2. Were not already found by ``pkgutil.iter_modules``.
        3. Contain at least one ``.py`` file (worth scanning).

        Each qualifying directory is imported as a namespace package and
        recursed into via :py:meth:`_scan_namespace_pkg`.
        """
        for search_path in pkg.__path__:
            root = pathlib.Path(search_path)
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                name = child.name
                if name.startswith("_") or name in already_found:
                    continue
                # Only probe directories that contain at least one .py file.
                if not any(child.glob("*.py")):
                    continue
                already_found.add(name)
                fqn = f"{namespace}.{name}"
                try:
                    mod = importlib.import_module(fqn)
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    logger.warning("failed to import `%s`: %s", fqn, e)
                    self._errors.append(RegistrationError(module=fqn, error=msg))
                    continue

                self._collect_plugins_from_module(mod, fqn, expected_type, discovered)

                if hasattr(mod, "__path__"):
                    self._scan_namespace_pkg(mod, fqn, expected_type, discovered)

    def _find_namespace_plugins(
        self,
        namespace: str,
        expected_type: PluginType | None = None,
    ) -> dict[str, Any]:
        """Scan a namespace package for ``@register``-decorated plugins."""
        try:
            pkg: ModuleType = importlib.import_module(namespace)
        except ImportError:
            return {}

        if not hasattr(pkg, "__path__"):
            return {}

        discovered: dict[str, Any] = {}
        self._scan_namespace_pkg(pkg, namespace, expected_type, discovered)
        return discovered

    # ── Public API ───────────────────────────────────────────────────────

    def get(self, plugin_type: PluginType) -> dict[str, Callable[..., Any]]:
        """Discover plugins for *plugin_type*.  Cached after first call.

        Returns a ``dict`` mapping plugin names to their factory callables.
        Returns an empty ``dict`` when no plugins are found (never ``None``).
        """
        if plugin_type not in self._cache:
            self._cache[plugin_type] = self._find(plugin_type)
        return self._cache[plugin_type]

    @overload
    def info(self) -> dict[PluginType, dict[str, Callable[..., Any]]]: ...

    @overload
    def info(self, plugin_type: PluginType) -> dict[str, Callable[..., Any]]: ...

    def info(
        self, plugin_type: PluginType | None = None
    ) -> (
        dict[PluginType, dict[str, Callable[..., Any]]] | dict[str, Callable[..., Any]]
    ):
        """Return discovered plugins.

        When called with no arguments, triggers discovery for **all** plugin
        groups and returns a defensive copy of the full cache keyed by
        :py:class:`PluginType`.

        When called with a *plugin_type*, returns a defensive copy of the
        plugins dict for that single group.

        Example::

            >>> from torchx import plugins
            >>> all_plugins = plugins.registry().info()  # doctest: +SKIP
            >>> scheds = plugins.registry().info(plugins.PluginType.SCHEDULER)
        """
        if plugin_type is not None:
            return dict(self.get(plugin_type))

        # Ensure all known groups are discovered.
        for pt in PluginType:
            self.get(pt)

        return {pt: dict(self._cache.get(pt, {})) for pt in PluginType}

    def clear(self) -> None:
        """Reset this instance and invalidate the :py:func:`registry` singleton.

        Clears per-group discovery caches and errors on this instance.
        Also calls ``registry.cache_clear()`` so the next ``registry()``
        call creates a fresh :py:class:`PluginRegistry`.

        Typical usage::

            plugins.registry().clear()
        """
        self._cache.clear()
        self._errors.clear()
        registry.cache_clear()

    @property
    def errors(self) -> list[RegistrationError]:
        """Errors encountered during plugin discovery."""
        return list(self._errors)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all discovered plugins to a plain dict.

        The returned structure is suitable for ``yaml.safe_dump`` or
        ``json.dumps``.  Keyed by ``PluginType.name.lower()`` with each
        group being a list of plugin dicts.

        Plugin-level errors (type mismatches, duplicates) are merged into
        their actual plugin type's section with an ``error`` key.
        Import-level errors (no plugin to attach to) remain in ``errors``.
        """
        all_plugins = self.info()
        data: dict[str, Any] = {}

        # Partition errors: plugin-level → merge into type section,
        # import-level → keep in separate "errors" section.
        plugin_errors: dict[str, list[RegistrationError]] = {}
        import_errors: list[dict[str, str]] = []
        for err in self._errors:
            if err.plugin_type is not None:
                plugin_errors.setdefault(err.plugin_type, []).append(err)
            else:
                import_errors.append({"module": err.module, "error": err.error})

        for pt in PluginType:
            group = all_plugins.get(pt, {})
            group_key = pt.name.lower()

            # Identify children (aliases and fractionals of a base plugin).
            child_names: set[str] = set()
            base_aliases: dict[str, list[str]] = {}
            base_fracs: dict[str, list[str]] = {}
            for name, fn in group.items():
                base = getattr(fn, "_plugin_base_name", None)
                if base is None:
                    continue
                child_names.add(name)
                if getattr(fn, "_plugin_is_alias", False):
                    base_aliases.setdefault(base, []).append(name)
                else:
                    base_fracs.setdefault(base, []).append(name)

            items: list[dict[str, Any]] = []
            for name, fn in group.items():
                if name in child_names:
                    continue
                entry: dict[str, Any] = {
                    "name": name,
                    "module": getattr(fn, "__module__", "unknown"),
                }
                aliases = base_aliases.get(name)
                if aliases:
                    entry["aliases"] = aliases
                fracs = base_fracs.get(name)
                if fracs:
                    entry["fractionals"] = fracs
                items.append(entry)

            # Append plugin-level errors for this type.
            for err in plugin_errors.get(group_key, []):
                items.append(
                    {
                        "name": err.name,
                        "module": err.module,
                        "error": err.error,
                    }
                )

            data[group_key] = items

        data["errors"] = import_errors
        return data

    @staticmethod
    def _yaml_quote(value: str) -> str:
        """Double-quote a YAML scalar value.

        Error messages are free-form text that can contain any YAML special
        characters (``:``, ``#``, ``{``, ``[``, etc.).  Always quoting
        avoids fragile heuristics about which characters need escaping.
        """
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def __str__(self) -> str:
        """Render a YAML representation of all discovered plugins.

        The output is valid YAML loadable with ``yaml.safe_load``.
        Sections mirror the internal ``_cache`` and ``_errors`` state.
        """
        data = self.to_dict()
        lines: list[str] = []
        for pt in PluginType:
            key = pt.name.lower()
            items = data.get(key, [])
            if not items:
                lines.append(f"{key}: []")
                continue
            lines.append(f"{key}:")
            for item in items:
                lines.append(f"  - name: {item['name']}")
                lines.append(f"    module: {item['module']}")
                if "aliases" in item:
                    lines.append(f"    aliases: [{', '.join(item['aliases'])}]")
                if "fractionals" in item:
                    lines.append(f"    fractionals: [{', '.join(item['fractionals'])}]")
                if "error" in item:
                    lines.append(f"    error: {self._yaml_quote(item['error'])}")
        errors = data.get("errors", [])
        if not errors:
            lines.append("errors: []")
        else:
            lines.append("errors:")
            for err in errors:
                lines.append(f"  - module: {err['module']}")
                lines.append(f"    error: {self._yaml_quote(err['error'])}")
        return "\n".join(lines)

    def _find(self, plugin_type: PluginType) -> dict[str, Callable[..., Any]]:
        """Find plugins for a single *plugin_type*.

        Merge priority (highest → lowest):

        1. ``importlib.metadata`` entry points (when ``ENTRYPOINT`` is set)
        2. ``torchx_plugins.<suffix>`` namespace submodules (when
           ``NAMESPACE_PKG`` is set)
        """
        group: str = plugin_type.value
        namespace = self._namespace_for_type(plugin_type)
        plugins = (
            self._find_namespace_plugins(namespace, plugin_type)
            if PluginSource.NAMESPACE_PKG in self._plugin_sources
            else {}
        )
        if PluginSource.ENTRYPOINT in self._plugin_sources:
            plugins |= entrypoints.load_group(group) or {}
        return plugins


@functools.lru_cache(maxsize=1)
def registry() -> PluginRegistry:
    """Return the cached :py:class:`PluginRegistry` singleton.

    The registry lazily discovers plugins per-group on first
    :py:meth:`~PluginRegistry.get` access and caches the results.

    The ``TORCHX_PLUGINS_SOURCE`` environment variable selects which
    discovery channels are enabled.  Its value is parsed as the integer
    representation of a :py:class:`PluginSource` bitmask: ``0`` for
    none, ``1`` for namespace package only, ``2`` for entry points only,
    ``3`` for both.  Defaults to all channels enabled when unset.

    Returns:
        The cached :py:class:`PluginRegistry` instance.

    Example::

        from torchx import plugins

        reg = plugins.registry()
        scheds = reg.get(plugins.PluginType.SCHEDULER)
        named = reg.get(plugins.PluginType.NAMED_RESOURCE)
        print(reg)
    """
    all_sources = PluginSource.NAMESPACE_PKG | PluginSource.ENTRYPOINT
    raw = os.environ.get("TORCHX_PLUGINS_SOURCE", str(int(all_sources)))
    try:
        value = int(raw)
        if not 0 <= value <= int(all_sources):
            raise ValueError
        plugin_sources = PluginSource(value)
    except ValueError as e:
        raise ValueError(
            f"TORCHX_PLUGINS_SOURCE={raw!r} is invalid; expected one of"
            " 0 (none), 1 (namespace only), 2 (entry points only),"
            " 3 (both)."
        ) from e
    return PluginRegistry(plugin_sources=plugin_sources)
