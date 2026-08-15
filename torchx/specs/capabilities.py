# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Typed accessors for :py:attr:`~torchx.specs.Resource.capabilities` entries.

``Resource.capabilities`` is a plain ``dict[str, Any]`` interpreted by
schedulers, so every producer/consumer pair otherwise re-spells the string
key and re-asserts the value type ad-hoc. A :py:class:`CapabilityKey` names
the key once and type-checks reads.

The wire format is unchanged: values are stored under the plain string key
``"<namespace>.<name>"`` — existing serialized resources and readers that
index ``capabilities`` directly keep working.
"""

from dataclasses import dataclass
from typing import Generic, overload, TypeVar

from torchx.specs.api import Resource

V = TypeVar("V")


@dataclass(frozen=True)
class CapabilityKey(Generic[V]):
    """A typed, namespaced key into ``Resource.capabilities``.

    .. doctest::

        >>> from torchx.specs import CapabilityKey, Resource
        >>> NETWORK = CapabilityKey("aws", "network_bandwidth_gbps", int)
        >>> NETWORK.key
        'aws.network_bandwidth_gbps'

        >>> res = Resource(cpu=4, gpu=0, memMB=1024)
        >>> NETWORK.set(res, 400)
        >>> res.capabilities
        {'aws.network_bandwidth_gbps': 400}
        >>> NETWORK.get(res)
        400
        >>> CapabilityKey("aws", "efa_enabled", bool).get(res, default=False)
        False

    Args:
        namespace: grouping prefix (e.g. a cloud provider or scheduler name).
        name: capability name within the namespace.
        type: expected value type; :py:meth:`get` raises ``TypeError`` when
            the stored value does not match.
    """

    namespace: str
    name: str
    type: type[V]

    @property
    def key(self) -> str:
        """The plain string key stored in ``Resource.capabilities``."""
        return f"{self.namespace}.{self.name}"

    def set(self, resource: Resource, value: V) -> None:
        """Store ``value`` under :py:attr:`key` in ``resource.capabilities``."""
        resource.capabilities[self.key] = value

    @overload
    def get(self, resource: Resource) -> V | None: ...

    @overload
    def get(self, resource: Resource, default: V) -> V: ...

    def get(self, resource: Resource, default: V | None = None) -> V | None:
        """Return the value stored under :py:attr:`key`, or ``default``.

        Raises:
            TypeError: if the stored value is not an instance of ``type``.
                ``bool`` does not satisfy an ``int``-typed key even though
                ``bool`` subclasses ``int``.
        """
        if self.key not in resource.capabilities:
            return default
        value = resource.capabilities[self.key]
        is_bool_where_int_expected = self.type is int and type(value) is bool
        if is_bool_where_int_expected or not isinstance(value, self.type):
            raise TypeError(
                f"capability `{self.key}` expected `{self.type.__name__}`,"
                f" got `{type(value).__name__}`: {value!r}"
            )
        return value
