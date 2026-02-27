# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Overlays patch the scheduler's submit-job request with fields not representable
in :py:class:`~torchx.specs.AppDef` or :py:class:`~torchx.specs.Role`.

Use :py:func:`set_overlay` / :py:func:`get_overlay` to store and retrieve overlays,
and :py:func:`apply_overlay` to apply them.

For Users
---------

Use :py:func:`set_overlay` to attach scheduler-specific fields to a
:py:class:`~torchx.specs.Role` or :py:class:`~torchx.specs.AppDef`:

.. doctest::

    >>> from torchx.specs import Role
    >>> from torchx.specs.overlays import set_overlay, get_overlay

    >>> # Kubernetes: add a node selector to a role
    >>> role = Role(name="trainer", image="my-image", entrypoint="train.py")
    >>> set_overlay(role, "kubernetes", "V1Pod", {
    ...     "spec": {"nodeSelector": {"accelerator": "a100"}},
    ... })

    >>> # Multiple set_overlay calls merge (dicts upsert, lists append)
    >>> set_overlay(role, "kubernetes", "V1Pod", {
    ...     "spec": {"tolerations": [{"key": "gpu", "operator": "Exists"}]},
    ... })
    >>> get_overlay(role, "kubernetes", "V1Pod")
    {'spec': {'nodeSelector': {'accelerator': 'a100'}, 'tolerations': [{'key': 'gpu', 'operator': 'Exists'}]}}

Operators
~~~~~~~~~

By default, :py:func:`set_overlay` merges dicts and appends lists. Use
:py:func:`PUT`, :py:func:`JOIN`, and :py:func:`DEL` as dict keys to override
per-field behavior:

.. doctest::

    >>> from torchx.specs import Role
    >>> from torchx.specs.overlays import set_overlay, PUT, JOIN, DEL

    >>> role = Role(name="trainer", image="my-image", entrypoint="train.py")

    >>> # PUT: replace a list instead of appending
    >>> set_overlay(role, "kubernetes", "V1Pod", {
    ...     "spec": {PUT("containers"): [{"name": "only"}]},
    ... })

    >>> # JOIN: strategic merge list items by key field
    >>> set_overlay(role, "kubernetes", "V1Pod", {
    ...     "spec": {JOIN("initContainers", on="name"): [
    ...         {"name": "setup", "image": "init:v2"},
    ...     ]},
    ... })

    >>> # DEL: remove a field (server uses its default)
    >>> set_overlay(role, "kubernetes", "V1Pod", {DEL("hostNetwork"): None})

Operators are stored in metadata and resolved automatically when the scheduler
calls :py:func:`apply_overlay` at submit time — users don't need to call
:py:func:`apply_overlay` directly.

.. note:: ``None`` vs ``DEL`` vs missing key

    These three states produce different results:

    - **Key missing from overlay**: field is untouched in the base
    - ``"field": None``: field is explicitly set to ``None``/null. In
      thrift/protobuf, this means "field present but null" — different from
      key is missing (never set).
    - ``DEL("field"): None``: field is **removed** from the base dict. In
      thrift/protobuf, this means "field not sent in request".

For Scheduler Implementors
--------------------------

To add overlay support to a scheduler, use :py:func:`get_overlay` to retrieve
stored overlays, :py:func:`validate_overlay` to guard against user error, and
:py:func:`apply_overlay` to apply the overlay onto the scheduler's base request dict.

.. code-block:: python

    from torchx.specs.overlays import apply_overlay, get_overlay, validate_overlay

    class MyScheduler(Scheduler):
        def _submit_dryrun(self, app, cfg):
            # Build base request from Role attributes
            base_request = build_request_from_role(app.roles[0])

            # Retrieve and validate the overlay
            overlay = get_overlay(app.roles[0], "my_scheduler", "JobSpec")
            validate_overlay(
                overlay,
                blocklist=["command", "env"],  # fields set via Role attrs
                overlay_name="JobSpec",
            )

            # Apply overlay onto the base request
            apply_overlay(base_request, overlay)

            # base_request now has user's overlay fields merged in
            return base_request

:py:func:`apply_overlay` handles operator keys (:py:func:`PUT`, :py:func:`JOIN`,
:py:func:`DEL`) automatically — the scheduler doesn't need to know about them.

.. fbcode::

    For MAST/MSL overlays, see :py:mod:`torchx.specs.fb.overlay_mast`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from torchx.specs import AppDef, Role

logger: logging.Logger = logging.getLogger(__name__)

_Overlay = dict[str, Any]

# Operator key prefixes. Each function returns a string key for use in overlay
# dicts. The prefix encodes the operation; the suffix encodes the field name
# (and merge key for JOIN).
_PUT_PREFIX = "__put__:"
_JOIN_PREFIX = "__join__:"
_DEL_PREFIX = "__del__:"


def PUT(key: str) -> str:
    """Replace a value entirely instead of merging/appending.

    Use as a dict key in overlays. For lists this replaces instead of appending;
    for dicts this replaces instead of recursive merge.

    .. doctest::

        >>> from torchx.specs.overlays import apply_overlay, PUT
        >>> base = {"containers": [{"name": "old1"}, {"name": "old2"}]}
        >>> apply_overlay(base, {PUT("containers"): [{"name": "only"}]})
        >>> base
        {'containers': [{'name': 'only'}]}

    """
    return f"{_PUT_PREFIX}{key}"


def JOIN(key: str, *, on: str) -> str:
    """Strategic merge list items by key field.

    Matched items (same value for ``on`` field) have their fields merged.
    Unmatched items are appended. Use as a dict key in overlays.

    .. doctest::

        >>> from torchx.specs.overlays import apply_overlay, JOIN
        >>> base = {"containers": [{"name": "main", "image": "v1", "cpu": "1"}]}
        >>> apply_overlay(base, {JOIN("containers", on="name"): [
        ...     {"name": "main", "memory": "1Gi"},
        ...     {"name": "sidecar", "image": "proxy"},
        ... ]})
        >>> base
        {'containers': [{'name': 'main', 'image': 'v1', 'cpu': '1', 'memory': '1Gi'}, {'name': 'sidecar', 'image': 'proxy'}]}

    Raises:
        TypeError: At apply time, if the base list contains non-dict items.

    """
    return f"{_JOIN_PREFIX}{key}:{on}"


def DEL(key: str) -> str:
    """Remove a key from the base dict.

    Use as a dict key in overlays. The value is ignored (convention: ``None``).

    This is different from setting a field to ``None`` — ``DEL`` removes the key
    entirely (thrift/protobuf: field not sent, server uses default), while
    ``"field": None`` sets it to null (thrift/protobuf: field present but null).

    .. doctest::

        >>> from torchx.specs.overlays import apply_overlay, DEL
        >>> base = {"keep": 1, "remove_me": "old"}
        >>> apply_overlay(base, {DEL("remove_me"): None})
        >>> base
        {'keep': 1}

    """
    return f"{_DEL_PREFIX}{key}"


def _field_of(key: str) -> str:
    # Extract the logical field name from a possibly-encoded operator key.
    if key.startswith(_PUT_PREFIX):
        return key[len(_PUT_PREFIX) :]
    if key.startswith(_DEL_PREFIX):
        return key[len(_DEL_PREFIX) :]
    if key.startswith(_JOIN_PREFIX):
        return key[len(_JOIN_PREFIX) :].split(":", 1)[0]
    return key


# Type alias for metadata that allows nested dicts
# Note: AppDef.metadata is typed as dict[str, str] but in practice stores dicts
# TODO: Fix AppDef.metadata type to dict[str, Any]
_Metadata = dict[str, Any]


# =============================================================================
# Core Overlay Semantics
# =============================================================================


def _strategic_merge_by_key(
    base_list: list[dict[str, Any]],
    overlay_list: list[dict[str, Any]],
    merge_key: str = "name",
) -> None:
    # Merge list items by key field (strategic merge patch semantics).
    # Matched items: overlay fields overwrite base fields (top-level only).
    # Unmatched items: appended.
    base_by_key: dict[str, dict[str, Any]] = {
        item[merge_key]: item
        for item in base_list
        if isinstance(item, dict) and merge_key in item
    }
    for overlay_item in overlay_list:
        if isinstance(overlay_item, dict) and merge_key in overlay_item:
            key_value = overlay_item[merge_key]
            if key_value in base_by_key:
                for k, v in overlay_item.items():
                    if k == merge_key:
                        continue
                    base_by_key[key_value][k] = copy.deepcopy(v)
            else:
                base_list.append(copy.deepcopy(overlay_item))
        else:
            base_list.append(copy.deepcopy(overlay_item))


def _remove_field_keys(base: _Overlay, field: str, *, keep_plain: bool = False) -> None:
    # Remove all keys from base whose logical field name is field.
    # When keep_plain is True, the plain key (no operator prefix) is kept.
    for k in [k for k in base if _field_of(k) == field]:
        if keep_plain and k == field:
            continue
        del base[k]


def _resolve_join(
    base: _Overlay,
    key: str,
    overlay_value: object,
    field: str,
) -> None:
    # Resolve a JOIN operator key into a strategic merge on base.
    _remove_field_keys(base, field, keep_plain=True)
    parts = key[len(_JOIN_PREFIX) :].split(":", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(
            f"malformed JOIN key: `{key}`, " f"expected format from JOIN(field, on=key)"
        )
    merge_key = parts[1]
    if not isinstance(overlay_value, list):
        raise TypeError(
            f"JOIN requires a list value, got "
            f"{type(overlay_value).__name__} for field `{field}`"
        )
    if not all(isinstance(item, dict) for item in overlay_value):
        raise TypeError(
            f"JOIN requires a list of dicts as overlay value, "
            f"but `{field}` overlay contains non-dict items"
        )
    if field in base:
        if not isinstance(base[field], list):
            raise TypeError(
                f"JOIN requires base[`{field}`] to be a list, "
                f"got {type(base[field]).__name__}"
            )
        if not all(isinstance(item, dict) for item in base[field]):
            raise TypeError(
                f"JOIN requires a list of dicts, but `{field}` "
                f"contains non-dict items"
            )
        _strategic_merge_by_key(base[field], overlay_value, merge_key=merge_key)
    else:
        base[field] = copy.deepcopy(overlay_value)


def _check_type_equal(key: str, o1: object, o2: object) -> None:
    # Raise TypeError if o1 and o2 have different types.
    o1_type = type(o1)
    o2_type = type(o2)
    if o1_type != o2_type:
        raise TypeError(
            f"type mismatch for attr: `{key}`. "
            f"{o1_type.__qualname__} != {o2_type.__qualname__}"
        )


def _merge_value(
    base: _Overlay,
    key: str,
    overlay_value: object,
    *,
    _resolve: bool,
) -> None:
    # Merge a single overlay value into base using default rules:
    # dict -> recursive merge, list -> append,
    # tuple -> replace list (legacy BC for YAML !!python/tuple), primitive -> overwrite
    if key in base:
        base_value = base[key]

        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            apply_overlay(base_value, overlay_value, _resolve=_resolve)
        elif isinstance(base_value, list) and isinstance(overlay_value, tuple):
            # BC: tuple replaces list (legacy alias for PUT). Supports
            # ``!!python/tuple`` tag in YAML overlay files.
            base[key] = list(overlay_value)
        elif isinstance(base_value, list) and isinstance(overlay_value, list):
            base_value.extend(copy.deepcopy(overlay_value))
        elif isinstance(base_value, (dict, list)) or isinstance(
            overlay_value, (dict, list)
        ):
            # Structural type mismatch (e.g., list vs str, dict vs int)
            _check_type_equal(key, base_value, overlay_value)
        else:
            # Primitive replacement: allows int/str mismatches since some
            # serialization formats (e.g., JSON) may represent enums as ints
            # while overlays use string names. The deserializer handles both.
            base[key] = overlay_value
    else:
        if isinstance(overlay_value, tuple):
            # BC: convert tuple to list for new keys
            base[key] = list(overlay_value)
        else:
            base[key] = copy.deepcopy(overlay_value)


def apply_overlay(
    base: _Overlay,
    overlay: _Overlay,
    *,
    _resolve: bool = True,
) -> None:
    """Merge ``overlay`` into ``base`` in-place.

    Default rules:

    1. **dict** → recursive merge (upsert keys)
    2. **list** → append overlay items
    3. **primitive** → overwrite value

    Operators (use as dict keys in overlays):

    4. :py:func:`PUT` → replace value entirely (lists, dicts, or primitives)
    5. :py:func:`JOIN` → strategic merge list items by key field
    6. :py:func:`DEL` → remove key from base

    During accumulation (multiple :py:func:`set_overlay` calls), operators for
    the same field replace earlier operations — last call wins.

    .. doctest::

        >>> from torchx.specs.overlays import apply_overlay, PUT, JOIN, DEL

        >>> # Dicts merge recursively, lists append, primitives overwrite
        >>> base = {"spec": {"cpu": "500m"}, "tags": ["prod"], "replicas": 1}
        >>> apply_overlay(base, {"spec": {"memory": "1Gi"}, "tags": ["gpu"], "replicas": 3})
        >>> base
        {'spec': {'cpu': '500m', 'memory': '1Gi'}, 'tags': ['prod', 'gpu'], 'replicas': 3}

        >>> # PUT replaces a list instead of appending
        >>> base = {"containers": [{"name": "old1"}, {"name": "old2"}]}
        >>> apply_overlay(base, {PUT("containers"): [{"name": "only"}]})
        >>> base
        {'containers': [{'name': 'only'}]}

        >>> # JOIN: match containers by name, merge their fields
        >>> base = {"containers": [{"name": "main", "image": "v1"}]}
        >>> apply_overlay(base, {JOIN("containers", on="name"): [
        ...     {"name": "main", "memory": "1Gi"},
        ... ]})
        >>> base
        {'containers': [{'name': 'main', 'image': 'v1', 'memory': '1Gi'}]}

        >>> # DEL: remove a key from the base
        >>> base = {"keep": 1, "remove": "old"}
        >>> apply_overlay(base, {DEL("remove"): None})
        >>> base
        {'keep': 1}

    Args:
        _resolve: **Internal only** — do not pass. When ``True`` (default),
            operator keys resolve to their operations on plain field names.
            When ``False``, operator keys are stored as-is (used by
            :py:func:`set_overlay` for accumulation).

    """
    for key, overlay_value in overlay.items():
        field = _field_of(key)

        # --- Resolve operators to their operations (application mode)
        if _resolve:
            if key.startswith(_DEL_PREFIX):
                _remove_field_keys(base, field)
                continue

            if key.startswith(_PUT_PREFIX):
                _remove_field_keys(base, field)
                base[field] = copy.deepcopy(overlay_value)
                continue

            if key.startswith(_JOIN_PREFIX):
                _resolve_join(base, key, overlay_value, field)
                continue

        # --- Conflict resolution: remove any existing keys for the same field
        # (accumulation mode for operators, always for plain keys)
        for k in [k for k in base if k != key and _field_of(k) == field]:
            del base[k]

        # --- Default merge behavior
        _merge_value(base, key, overlay_value, _resolve=_resolve)


# =============================================================================
# Overlay Storage API
# =============================================================================


def _load_overlay_file(uri: str) -> _Overlay:
    # Load overlay dict from a file path or URI.
    # Supports bare paths, file://, s3://, etc. via fsspec.
    # Tries JSON first, falls back to YAML.
    # Raises ValueError if the file contents are not a dict.
    import json

    import fsspec

    # Bare paths (no scheme) are treated as local file paths
    if "://" not in uri:
        uri = f"file://{uri}"

    with fsspec.open(uri, "r") as f:
        contents = f.read()

    # Try JSON first, fall back to YAML.
    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        import yaml

        # BC: Support ``!!python/tuple`` tag in YAML overlay files.
        # Tuples indicate list-replace semantics (legacy alias for PUT).
        class _TupleSafeLoader(yaml.SafeLoader):
            pass

        _TupleSafeLoader.add_constructor(
            "tag:yaml.org,2002:python/tuple",
            lambda loader, node: tuple(loader.construct_sequence(node)),
        )

        data = yaml.load(contents, Loader=_TupleSafeLoader)  # noqa: S506

    if not isinstance(data, dict):
        raise ValueError(
            f"overlay file `{uri}` must contain a dict, got {type(data).__name__}"
        )
    return data


def set_overlay(
    target: AppDef | Role,
    namespace: str,
    kind: str,
    overlay: _Overlay,
) -> None:
    """Store an overlay in ``target.metadata[namespace][kind]``.

    Multiple calls for the same ``(namespace, kind)`` accumulate via
    :py:func:`apply_overlay` (dicts merge, lists append). Use :py:func:`PUT`,
    :py:func:`JOIN`, and :py:func:`DEL` operators in the overlay dict to
    control per-field behavior.

    Args:
        namespace: Scheduler namespace (e.g., ``"kubernetes"``, ``"mast"``).
        kind: Scheduler struct type (e.g., ``"V1Pod"``,
            ``"HpcJobDefinition"``).
    """
    # Cast metadata to allow nested dicts
    # Note: AppDef.metadata is typed as dict[str, str] but overlays require nested dicts
    metadata = cast(_Metadata, target.metadata)

    # Ensure namespace dict exists
    if namespace not in metadata:
        metadata[namespace] = {}

    ns_dict = metadata[namespace]
    if not isinstance(ns_dict, dict):
        # If namespace key exists but isn't a dict, replace it
        logger.warning(
            "replacing non-dict value for namespace `%s` in metadata with "
            "nested overlay format. Previous value was: %s",
            namespace,
            type(ns_dict).__name__,
        )
        ns_dict = {}
        metadata[namespace] = ns_dict

    # Ensure kind dict exists and merge
    if kind not in ns_dict:
        ns_dict[kind] = {}

    existing = ns_dict[kind]
    if not isinstance(existing, dict):
        existing = {}
        ns_dict[kind] = existing

    apply_overlay(existing, overlay, _resolve=False)


def get_overlay(
    target: AppDef | Role,
    namespace: str,
    kind: str,
) -> _Overlay:
    """Retrieve overlay from ``target.metadata[namespace][kind]``.

    Returns ``{}`` if not found. If ``metadata[namespace]`` is a string,
    it is loaded as a file URI via ``fsspec`` (JSON or YAML).

    For backwards compatibility, if ``kind`` is not a key in
    ``metadata[namespace]``, the entire namespace dict is returned as a
    flat overlay (with a deprecation warning).
    """
    if namespace not in target.metadata:
        return {}

    ns_value = target.metadata[namespace]

    # File URI support: load overlay from file
    if isinstance(ns_value, str):
        ns_value = _load_overlay_file(ns_value)

    if not isinstance(ns_value, dict):
        return {}

    # New nested format: metadata[namespace][kind]
    if kind in ns_value:
        overlay = ns_value[kind]
        if isinstance(overlay, dict):
            return overlay

    # Backwards compat: flat format metadata[namespace] = {overlay} (no kind key).
    # Only fall back if ns_value doesn't look like nested format (multiple
    # dict-valued keys = nested, e.g. {"HpcTaskGroupSpec": {...}, ...}).
    has_multiple_dict_values = len(ns_value) > 1 and all(
        isinstance(v, dict) for v in ns_value.values()
    )
    if ns_value and not has_multiple_dict_values:
        logger.warning(
            "overlay kind `%s` not found in namespace `%s` "
            '(available kinds: %s). Treating `metadata["%s"]` as a flat '
            "overlay. Migrate to "
            '`torchx.specs.overlays.set_overlay(target, "%s", "%s", overlay)` '
            "for the nested format",
            kind,
            namespace,
            list(ns_value.keys()),
            namespace,
            namespace,
            kind,
        )
        return ns_value

    return {}


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_overlay(
    overlay: _Overlay,
    *,
    blocklist: list[str] | None = None,
    forbidden_keys: set[str] | None = None,
    overlay_name: str = "overlay",
    suggestion: str = "",
) -> None:
    """Validate that overlay doesn't contain disallowed keys.

    Used by scheduler authors to guard against user error. Operator-prefixed
    keys (e.g., ``PUT("env")``) are resolved to their logical field name
    before checking against the blocklist.

    Args:
        blocklist: Keys that should be set via Role/AppDef attributes.
        forbidden_keys: Keys that belong to a different overlay type.
        overlay_name: Overlay type name for error messages.
        suggestion: Hint appended when ``forbidden_keys`` are found.

    Raises:
        ValueError: If validation fails.

    .. doctest::

        >>> from torchx.specs.overlays import validate_overlay

        >>> # "env" is blocklisted — should be set via Role.env
        >>> try:
        ...     validate_overlay(
        ...         {"env": {"FOO": "bar"}, "nodeSelector": {"gpu": "true"}},
        ...         blocklist=["env", "command"],
        ...         overlay_name="PodSpec",
        ...     )
        ... except ValueError as e:
        ...     "env" in str(e)
        True

    """
    if blocklist:
        disallowed = [_field_of(key) for key in overlay if _field_of(key) in blocklist]
        if disallowed:
            keys_str = ", ".join(f"`{overlay_name}.{k}`" for k in disallowed)
            raise ValueError(
                f"disallowed overlay attributes {keys_str} since "
                f"they can be set directly on the role's attributes."
            )

    if forbidden_keys:
        misplaced = {_field_of(k) for k in overlay} & forbidden_keys
        if misplaced:
            msg = f"{overlay_name} overlay contains misplaced keys: {misplaced}."
            if suggestion:
                msg = f"{msg} {suggestion}"
            raise ValueError(msg)
