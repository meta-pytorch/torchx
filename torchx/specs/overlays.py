# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Overlays are JSON structs applied to :py:class:`~torchx.specs.AppDef` and :py:class:`~torchx.specs.Role`
to specify attributes of the scheduler's submit-job request that are not currently representable
as attributes of :py:class:`~torchx.specs.AppDef` and :py:class:`~torchx.specs.Role`.

For end-uses, here are a few use-cases of overlays:

1. A new version of the scheduler has concepts/features that have not yet been added to TorchX.
2. A bespoke internal scheduler has custom features that do not generalize hence not in TorchX.
3. Re-using a pre-built ``AppDef`` but need to make a small change to the resulting scheduler request.

And for scheduler authors:

1. Scheduler setting needs to be applied to a ``Role``, which makes it hard to add as ``runopts``
   since ``runopts`` apply at the ``AppDef`` level.
2. Scheduler setting cannot be represented naturally as the types supported by ``runopts``.
3. Exposing the setting as a ``runopts`` obfuscates things.

See :py:func:`~torchx.specs.overlays.apply_overlay` for rules on how overlays are applied.
"""

from typing import Any

Json = dict[str, Any]


def apply_overlay(base: Json, overlay: Json) -> None:
    """Applies ``overlay`` on ``base``.

    .. note:: this function mutates the ``base``!

    Overlays follow these rules:

    1. Dicts, upsert key, value in base with the ones in overlay.
    2. Nested dicts, overlay recursively.
    3. Lists, append the overlay values to the base values.
    4. Nested lists DO NOT append recursively.
    5. Primitives (bool, str, int, float), replace base with the value in overlay.

    .. doctest::

        from torchx.specs.overlays import apply_overlay

        base = {
            "scheduler": {"policy": "default"},
            "resources": {"limits": {"cpu": "500m"}},
            "tolerations": [{"key": "gpu"}],
            "nodeSelectorTerms": [
                [{"matchExpressions": []}]
            ],
            "maxPods": 110,
        }
        overlay = {
            "scheduler": {"policy": "binpacking"},
            "resources": {"limits": {"memory": "1Gi"}},
            "tolerations": [{"key": "spot"}],
            "nodeSelectorTerms": [
                [{"matchExpressions": [{"key": "disk"}]}]
            ],
            "maxPods": 250,
        }

        apply_overlay(base, overlay)

        assert {
            "scheduler": {"policy": "binpacking"},
            "resources": {"limits": {"cpu": "500m", "memory": "1Gi"}},
            "tolerations": [{"key": "gpu"}, {"key": "spot"}],
            "nodeSelectorTerms": [
                [{"matchExpressions": []}],
                [{"matchExpressions": [{"key": "disk"}]}],
            ],
            "maxPods": 250,
        } == base

    """

    def assert_type_equal(key: str, o1: object, o2: object) -> None:
        o1_type = type(o1)
        o2_type = type(o2)
        assert (
            o1_type == o2_type
        ), f"Type mismatch for attr: `{key}`. {o1_type.__qualname__} != {o2_type.__qualname__}"

    for key, overlay_value in overlay.items():
        if key in base:
            base_value = base[key]

            assert_type_equal(key, base_value, overlay_value)

            if isinstance(base_value, dict) and isinstance(overlay_value, dict):
                apply_overlay(base_value, overlay_value)
            elif isinstance(base_value, list) and isinstance(overlay_value, list):
                base_value.extend(overlay_value)
            else:
                base[key] = overlay_value
        else:
            base[key] = overlay_value
