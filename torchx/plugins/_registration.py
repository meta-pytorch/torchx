# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[2, 3, 11, 16, 21]

"""Plugin registration decorators and named-resource authoring utilities.

This is an internal module — import from :py:mod:`torchx.plugins` instead.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from torchx.specs.api import Resource  # @manual

#: TypeVar for the decorated function — ``__call__`` returns the same
#: callable type it receives so downstream code retains the original
#: function signature.
F = TypeVar("F", bound=Callable[..., object])

from torchx.plugins._registry import NAMED_RESOURCES_ATTR, PluginType


# ── Named-resource authoring utilities ────────────────────────────────────────


#: Fractional host allocation constants — what share of a host's resources
#: to request.  Each represents a fraction of **all** host resources
#: (CPU cores, memory, and GPUs) allocated to a single job.
#:
#: Use in named-resource factories and fractionals dicts::
#:
#:     from torchx.plugins import register, WHOLE, HALF, QUARTER
#:
#:     @register.named_resource(fractionals={WHOLE: "8", HALF: "4"})
#:     def my_resource(fractional: float = WHOLE) -> Resource:
#:         gpu = int(8 * fractional)
#:         ...

WHOLE: float = 1.0  #: Entire host — all CPU, memory, and GPU.
HALF: float = 0.5  #: Half of all host resources.
QUARTER: float = 0.25  #: One quarter of all host resources.
EIGHTH: float = 0.125  #: One eighth of all host resources.
SIXTEENTH: float = 0.0625  #: One sixteenth of all host resources.


class resource_tags:
    """Well-known ``Resource.tags`` keys set by :py:class:`register`.

    Named-resource factories wrapped by :py:meth:`register.named_resource`
    automatically populate these tags on every :py:class:`Resource` they
    return so that downstream code can recover the registered name and
    whether the resource is a fractional slice.

    Example::

        from torchx.plugins import resource_tags

        res = result["gpu_4"]()
        assert res.tags[resource_tags.RESOURCE_NAME] == "gpu_4"
        assert res.tags[resource_tags.IS_FRACTIONAL] is True
    """

    RESOURCE_NAME: str = "torchx/named_resources.name"
    """Registered name that produced this :py:class:`Resource`."""

    IS_FRACTIONAL: str = "torchx/named_resources.is_fractional"
    """``True`` when the resource is a fractional slice of a base resource."""


class register:
    """Decorator that tags a function as a TorchX plugin.

    Sets ``_plugin_type`` and ``_plugin_name`` attributes on the decorated
    function.  The discovery scanner (:py:class:`PluginRegistry`) imports
    each submodule under ``torchx_plugins.*`` and collects any callable
    with a ``_plugin_type`` attribute.

    Usage::

        from torchx.plugins import register

        @register.scheduler()
        def my_scheduler(session_name: str, **kwargs) -> Scheduler:
            ...

        @register.scheduler(name="custom_name")
        def create_custom(session_name: str, **kwargs) -> Scheduler:
            ...

    Each :py:class:`PluginType` has a corresponding classmethod:
    :py:meth:`scheduler`, :py:meth:`tracker`, and :py:meth:`named_resource`.

    The explicit constructor ``register(PluginType.SCHEDULER, name=...)``
    is still supported for advanced use-cases.

    Fractional resource helpers
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :py:meth:`powers_of_two_gpus` and :py:meth:`halve_mem_down_to` are
    available as staticmethods so plugin authors can use them with a
    single import::

        @register.named_resource(fractionals=register.powers_of_two_gpus)
        def my_gpu(fractional: float = 1.0) -> Resource:
            ...

        @register.named_resource(fractionals=register.halve_mem_down_to(minGiB=16))
        def my_cpu(fractional: float = 1.0) -> Resource:
            ...

    Args:
        type: The :py:class:`PluginType` declaring what kind of plugin
            this function provides.
        name: Plugin name. Defaults to the decorated function's ``__name__``.
    """

    def __init__(self, type: PluginType, name: str | None = None) -> None:
        self._type = type
        self._name = name

    def __call__(self, fn: F) -> F:
        plugin_name: str = self._name or fn.__name__
        fn._plugin_type = self._type  # type: ignore[attr-defined]
        fn._plugin_name = plugin_name  # type: ignore[attr-defined]
        return fn

    # -- convenience classmethods ------------------------------------------

    @classmethod
    def scheduler(cls, name: str | None = None) -> register:
        """Register a :py:class:`~torchx.schedulers.api.Scheduler` factory."""
        return cls(PluginType.SCHEDULER, name=name)

    @classmethod
    def tracker(cls, name: str | None = None) -> register:
        """Register a tracker factory."""
        return cls(PluginType.TRACKER, name=name)

    @classmethod
    def named_resource(
        cls,
        name: str | None = None,
        aliases: list[str] | None = None,
        fractionals: Callable[..., dict[float, str]] | dict[float, str] | None = None,
    ) -> _register_named_resource:
        """Register a named resource factory.

        Args:
            name: Resource name. Defaults to the decorated function's
                ``__name__``.
            aliases: Additional names that point to the same factory.
            fractionals: Either a callable ``(Resource) → {fraction: suffix}``
                or a literal dict.  When provided, the decorated function
                **must** accept a ``fractional: float`` parameter.
        """
        return _register_named_resource(
            name=name, aliases=aliases, fractionals=fractionals
        )

    # -- named-resource authoring utilities (staticmethods) ----------------

    @staticmethod
    def powers_of_two_gpus(resource: Any) -> dict[float, str]:
        """Return fractional specs for every power-of-two GPU slice of *resource*.

        For example, an 8-GPU resource produces::

            {1.0: "8", 0.5: "4", 0.25: "2", 0.125: "1"}

        When passed as the ``fractionals`` argument of
        :py:meth:`register.named_resource`, this auto-generates and registers
        all power-of-two fractional variants (e.g. ``my_gpu_8``, ``my_gpu_4``,
        …).

        Args:
            resource: The base (whole-host) resource.  Must have ``gpu > 0``
                and ``gpu`` must be a power of two.

        Raises:
            ValueError: if ``resource.gpu`` is zero or not a power of two.
        """

        def _is_pow2(n: int) -> bool:
            return (n > 0) and ((n & (n - 1)) == 0)

        gpus = resource.gpu
        if gpus <= 0:
            raise ValueError(
                f"resource must have gpu > 0 to generate power-of-two slices (got {gpus})"
            )
        if not _is_pow2(gpus):
            raise ValueError(
                f"resource.gpu must be a power of two to generate slices (got {gpus})"
            )

        fractionals: dict[float, str] = {}
        fraction = 1.0
        while (fractional_gpus := int(gpus * fraction)) > 0:
            fractionals[fraction] = str(fractional_gpus)
            fraction /= 2
        return fractionals

    @staticmethod
    def halve_mem_down_to(
        *,
        minGiB: int,
    ) -> Callable[..., dict[float, str]]:
        """Generate fractional suffixes by halving memory in GiB.

        Returns a callable that produces a geometric series with r=1/2
        starting from the resource's total memory in GiB down to ``minGiB``.
        Each entry maps a fractional float to the corresponding memory
        suffix string.

        Example — 64 GiB host with ``minGiB=8``::

            {1.0: "64", 0.5: "32", 0.25: "16", 0.125: "8"}

        Usage::

            @register.named_resource(fractionals=register.halve_mem_down_to(minGiB=16))
            def t1(fractional: float = WHOLE) -> Resource: ...

        Args:
            minGiB: Stop generating fractionals when memory drops below
                this threshold in GiB.  Must be >= the odd part of
                ``memGiB`` (i.e. ``memGiB`` with all factors of 2
                removed), otherwise halving would produce non-integer
                GiB values.

        Raises:
            ValueError: if ``resource.memMB`` is zero, not GiB-aligned,
                or ``minGiB`` is below the odd part of ``memGiB``.
        """

        def _odd_part(n: int) -> int:
            """Return the odd part of *n* (strip all factors of 2)."""
            while n > 0 and n % 2 == 0:
                n //= 2
            return n

        def _factory(resource: Any) -> dict[float, str]:
            mem_mb = resource.memMB
            if mem_mb <= 0:
                raise ValueError(
                    f"resource must have memMB > 0 to generate memory slices (got {mem_mb})"
                )
            if mem_mb % 1024 != 0:
                raise ValueError(
                    f"resource.memMB must be a whole number of GiB (got {mem_mb} MB)"
                )
            mem_gib = mem_mb // 1024
            odd = _odd_part(mem_gib)
            if minGiB < odd:
                raise ValueError(
                    f"`minGiB` must be >= the odd part of `memGiB` ({odd}) "
                    f"because halving {mem_gib} GiB below {odd} produces "
                    f"non-integer GiB values (got minGiB={minGiB})"
                )

            fractionals: dict[float, str] = {}
            fraction = 1.0
            while (frac_gib := int(mem_gib * fraction)) >= minGiB:
                fractionals[fraction] = str(frac_gib)
                fraction /= 2
            return fractionals

        return _factory


class _register_named_resource(register):
    """Specialised :py:class:`register` for named resources.

    At decoration time, each factory is:

    1. Tagged with ``_plugin_type`` and ``_plugin_name`` for scanner
       discovery via ``dir(mod)``.
    2. Injected as a module-level attribute so it can be imported directly
       (``from my_module import my_gpu_4``).
    3. Written to a per-module ``NAMED_RESOURCES`` dict for backward
       compatibility (``getattr(mod, "NAMED_RESOURCES")``).

    Handles the generic mechanics of named-resource registration:

    * Supports **aliases** — additional names pointing to the same factory.
      Aliases do **not** receive fractional variants; only the base name
      does.
    * Supports **fractionals** — when a ``fractionals`` callable or dict is
      provided, the decorated function is called with ``fractional=1.0`` at
      decoration time to obtain the base :py:class:`Resource`.  The
      fractionals callable maps each fraction to a human-readable suffix.
      For every ``{fraction: suffix}`` entry, a zero-arg factory named
      ``{base_name}_{suffix}`` is auto-generated and registered.

      For example, an 8-GPU resource with :py:meth:`register.powers_of_two_gpus`
      produces ``my_gpu_8``, ``my_gpu_4``, ``my_gpu_2``, ``my_gpu_1`` —
      each calling ``my_gpu(fractional=<fraction>)``.

    Subclasses override the two hook methods to inject platform-specific
    metadata:

    * :py:meth:`_make_factory` — wrap the base factory (e.g. add tags).
    * :py:meth:`_make_fractional` — wrap each fractional factory.
    """

    def __init__(
        self,
        name: str | None = None,
        aliases: list[str] | None = None,
        fractionals: Callable[..., dict[float, str]] | dict[float, str] | None = None,
    ) -> None:
        super().__init__(PluginType.NAMED_RESOURCE, name=name)
        self._aliases: list[str] = aliases or []
        self._has_fractionals: bool = bool(fractionals)

        if isinstance(fractionals, dict):
            captured: dict[float, str] = fractionals

            def _from_dict(_: Any) -> dict[float, str]:
                return captured

            self._fractionals: Callable[..., dict[float, str]] = _from_dict
        elif fractionals is not None:
            self._fractionals = fractionals
        else:
            self._fractionals = lambda _: {}

    # -- hooks for subclasses ------------------------------------------------

    def _make_factory(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        """Wrap the base factory before registration.

        The default implementation returns *fn* unchanged.  Subclasses
        override to inject metadata (e.g. whole-host flags, resource tags).
        """
        return fn

    def _make_fractional(
        self,
        fn: Callable[..., Any],
        fraction: float,
        frac_name: str,
    ) -> Callable[..., Any]:
        """Create a zero-arg factory for a fractional resource variant.

        The default implementation returns ``lambda: fn(fraction)``.
        Subclasses override to inject fractional-specific metadata.
        """

        def fractional_factory() -> Any:
            return fn(fraction)

        fractional_factory.__module__ = fn.__module__
        fractional_factory.__qualname__ = frac_name
        return fractional_factory

    # -- registration orchestration ------------------------------------------

    def __call__(self, fn: Callable[..., Resource]) -> Callable[..., Resource]:
        mod: ModuleType = sys.modules[fn.__module__]
        name: str = self._name or fn.__name__

        # Backward-compat dict: getattr(mod, "NAMED_RESOURCES").
        existing: dict[str, Any] | None = getattr(mod, NAMED_RESOURCES_ATTR, None)
        bc_reg: dict[str, Any] = existing if existing is not None else {}
        if existing is None:
            setattr(mod, NAMED_RESOURCES_ATTR, bc_reg)

        def _tag_and_register(reg_name: str, factory: Callable[..., Any]) -> None:
            if reg_name in bc_reg:
                raise ValueError(
                    f"duplicate named resource `{reg_name}` in module "
                    f"`{mod.__name__}`"
                )
            factory._plugin_type = PluginType.NAMED_RESOURCE  # type: ignore[attr-defined]
            factory._plugin_name = reg_name  # type: ignore[attr-defined]
            bc_reg[reg_name] = factory

        def _set_module_attr(attr_name: str, factory: Callable[..., Any]) -> None:
            if hasattr(mod, attr_name):
                raise AttributeError(
                    f"`{attr_name}()` already exists in `{mod.__name__}`"
                )
            setattr(mod, attr_name, factory)

        # [1] Base factory
        factory = self._make_factory(fn, name)
        _tag_and_register(name, factory)

        if name != fn.__name__:
            _set_module_attr(name, factory)

        # [2] Aliases — thin wrappers so _plugin_base_name doesn't leak
        # onto the base factory object (which is the same callable).
        for alias in self._aliases:

            def _alias_wrapper(
                *a: Any, _fn: Callable[..., Any] = factory, **kw: Any
            ) -> Any:
                return _fn(*a, **kw)

            _alias_wrapper.__module__ = factory.__module__
            _alias_wrapper.__qualname__ = alias
            _alias_wrapper._plugin_base_name = name  # type: ignore[attr-defined]
            _alias_wrapper._plugin_is_alias = True  # type: ignore[attr-defined]
            _tag_and_register(alias, _alias_wrapper)
            _set_module_attr(alias, _alias_wrapper)

        # [3] Fractionals — only when fractionals were configured.
        # Skipped when no fractionals argument was passed to avoid an eager
        # fn() call at decoration time for resources that don't need slicing.
        if self._has_fractionals:
            base_resource = fn()
            for fraction, suffix in self._fractionals(base_resource).items():
                frac_name = f"{name}_{suffix}"
                frac_factory = self._make_fractional(fn, fraction, frac_name)
                frac_factory._plugin_base_name = name  # type: ignore[attr-defined]
                _tag_and_register(frac_name, frac_factory)
                _set_module_attr(frac_name, frac_factory)

        return factory  # NB: returns _make_factory(fn), not fn


# ── Module-level aliases ─────────────────────────────────────────────────────
# Expose fractional helpers at package level so plugin authors can write
# ``from torchx.plugins import powers_of_two_gpus`` instead of reaching
# into ``register.powers_of_two_gpus``.

powers_of_two_gpus: Callable[[Resource], dict[float, str]] = register.powers_of_two_gpus
halve_mem_down_to: Callable[..., Callable[[Resource], dict[float, str]]] = (
    register.halve_mem_down_to
)
