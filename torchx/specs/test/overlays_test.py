# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import json
import os
import tempfile
import unittest
from typing import Any

from torchx.specs import AppDef, Role
from torchx.specs.overlays import (
    _load_overlay_file,
    apply_overlay,
    DEL,
    get_overlay,
    JOIN,
    PUT,
    set_overlay,
    validate_overlay,
)


# =============================================================================
# apply_overlay: default merge semantics
# =============================================================================


class ApplyOverlayTest(unittest.TestCase):
    """Default merge semantics: dicts upsert, lists append, primitives replace."""

    def test_empty_base(self) -> None:
        base: dict[str, object] = {}
        overlay = {"a": {"b": "c"}, "tags": ["gpu"], "count": 1}
        apply_overlay(base, overlay)
        self.assertEqual(base, overlay)

    def test_dict_merge(self) -> None:
        base = {"spec": {"cpu": "500m"}}
        apply_overlay(base, {"spec": {"memory": "1Gi"}})
        self.assertEqual(base, {"spec": {"cpu": "500m", "memory": "1Gi"}})

    def test_dict_overwrite(self) -> None:
        base = {"spec": {"cpu": "500m"}}
        apply_overlay(base, {"spec": {"cpu": "2"}})
        self.assertEqual(base, {"spec": {"cpu": "2"}})

    def test_list_append(self) -> None:
        base = {"tags": ["prod"]}
        apply_overlay(base, {"tags": ["gpu"]})
        self.assertEqual(base, {"tags": ["prod", "gpu"]})

    def test_list_append_no_dedup(self) -> None:
        base = {"tags": ["a", "b"]}
        apply_overlay(base, {"tags": ["b", "c"]})
        self.assertEqual(base, {"tags": ["a", "b", "b", "c"]}, "lists append, no dedup")

    def test_primitive_replace(self) -> None:
        base = {"replicas": 1}
        apply_overlay(base, {"replicas": 3})
        self.assertEqual(base, {"replicas": 3})

    def test_primitive_type_coercion(self) -> None:
        """int → str is allowed for thrift enum serialization compat."""
        base = {"priority": 1}
        apply_overlay(base, {"priority": "HIGH"})
        self.assertEqual(base, {"priority": "HIGH"})

    def test_structural_type_mismatch_raises(self) -> None:
        with self.assertRaises(TypeError):
            apply_overlay({"a": [1]}, {"a": "b"})
        with self.assertRaises(TypeError):
            apply_overlay({"a": {"b": 1}}, {"a": "not_a_dict"})

    def test_new_key_deep_copied(self) -> None:
        inner = {"key": "value"}
        base: dict[str, object] = {}
        apply_overlay(base, {"new": inner})
        inner["key"] = "changed"
        self.assertEqual(
            base["new"],
            {"key": "value"},
            "deep copy should isolate from overlay mutation",
        )

    def test_tuple_replaces_list(self) -> None:
        """BC: tuple in overlay replaces list (legacy alias for PUT)."""
        base = {"containers": [{"name": "old1"}, {"name": "old2"}]}
        apply_overlay(base, {"containers": ({"name": "new"},)})
        self.assertEqual(base, {"containers": [{"name": "new"}]})

    def test_tuple_new_key_becomes_list(self) -> None:
        """BC: tuple converted to list when key doesn't exist in base."""
        base: dict[str, object] = {}
        apply_overlay(base, {"items": (1, 2)})
        self.assertEqual(base, {"items": [1, 2]})


# =============================================================================
# Operators: PUT, JOIN, DEL
# =============================================================================


class OperatorKeyFormatTest(unittest.TestCase):
    """Operator key prefixes are part of the serialized format in metadata.

    Changing them would break stored overlays, so they must be stable.
    """

    def test_put_key_format(self) -> None:
        self.assertEqual(PUT("x"), "__put__:x")

    def test_join_key_format(self) -> None:
        self.assertEqual(JOIN("x", on="y"), "__join__:x:y")

    def test_del_key_format(self) -> None:
        self.assertEqual(DEL("x"), "__del__:x")


class PutOperatorTest(unittest.TestCase):
    """PUT(key) replaces a value instead of merging/appending."""

    def test_put_replaces_list(self) -> None:
        base = {"containers": [{"name": "old1"}, {"name": "old2"}]}
        apply_overlay(base, {PUT("containers"): [{"name": "only"}]})
        self.assertEqual(
            base, {"containers": [{"name": "only"}]}, "PUT should replace list"
        )

    def test_put_replaces_dict(self) -> None:
        base = {"spec": {"cpu": "500m", "memory": "1Gi"}}
        apply_overlay(base, {PUT("spec"): {"gpu": "1"}})
        self.assertEqual(
            base, {"spec": {"gpu": "1"}}, "PUT should replace dict entirely"
        )

    def test_put_new_key(self) -> None:
        base: dict[str, object] = {"a": 1}
        apply_overlay(base, {PUT("b"): [1, 2]})
        self.assertEqual(base, {"a": 1, "b": [1, 2]})

    def test_put_conflict_resolution(self) -> None:
        """PUT on a field that already exists as a plain key replaces it."""
        base = {"tags": ["old"]}
        apply_overlay(base, {PUT("tags"): ["new"]})
        self.assertEqual(base, {"tags": ["new"]})


class JoinOperatorTest(unittest.TestCase):
    """JOIN(key, on=field) does strategic merge on list items by key field."""

    def test_join_merges_matched_item(self) -> None:
        """Matched items have their fields merged (top-level only)."""
        base = {"containers": [{"name": "main", "image": "v1", "cpu": "1"}]}
        apply_overlay(
            base,
            {JOIN("containers", on="name"): [{"name": "main", "memory": "1Gi"}]},
        )
        self.assertEqual(
            base["containers"],
            [{"name": "main", "image": "v1", "cpu": "1", "memory": "1Gi"}],
        )

    def test_join_appends_unmatched_item(self) -> None:
        base = {"containers": [{"name": "main", "image": "v1"}]}
        apply_overlay(
            base,
            {
                JOIN("containers", on="name"): [
                    {"name": "sidecar", "image": "proxy:v1"},
                ]
            },
        )
        self.assertEqual(len(base["containers"]), 2)
        self.assertEqual(base["containers"][1]["name"], "sidecar")

    def test_join_mixed_merge_and_append(self) -> None:
        """Matched items merge, unmatched items append — in one call."""
        base = {"containers": [{"name": "main", "image": "v1"}]}
        apply_overlay(
            base,
            {
                JOIN("containers", on="name"): [
                    {"name": "main", "memory": "1Gi"},
                    {"name": "sidecar", "image": "proxy:v1"},
                ]
            },
        )
        self.assertEqual(len(base["containers"]), 2)
        self.assertEqual(base["containers"][0]["memory"], "1Gi")
        self.assertEqual(base["containers"][0]["image"], "v1")
        self.assertEqual(base["containers"][1]["name"], "sidecar")

    def test_join_custom_key_field(self) -> None:
        base = {"volumes": [{"id": "vol-1", "size": "100Gi"}]}
        apply_overlay(
            base,
            {JOIN("volumes", on="id"): [{"id": "vol-1", "mountPath": "/data"}]},
        )
        self.assertEqual(len(base["volumes"]), 1)
        self.assertEqual(base["volumes"][0]["mountPath"], "/data")
        self.assertEqual(base["volumes"][0]["size"], "100Gi")

    def test_join_empty_overlay_is_noop(self) -> None:
        base = {"containers": [{"name": "main"}]}
        apply_overlay(base, {JOIN("containers", on="name"): []})
        self.assertEqual(len(base["containers"]), 1)

    def test_join_non_dict_items_raises(self) -> None:
        """JOIN on a list of primitives raises TypeError."""
        base = {"tags": ["prod", "gpu"]}
        with self.assertRaises(TypeError):
            apply_overlay(base, {JOIN("tags", on="name"): [{"name": "new"}]})

    def test_join_non_list_value_raises(self) -> None:
        base = {"containers": [{"name": "main"}]}
        with self.assertRaises(TypeError):
            apply_overlay(base, {JOIN("containers", on="name"): {"name": "bad"}})

    def test_join_non_dict_overlay_items_raises(self) -> None:
        """JOIN overlay items must all be dicts."""
        base = {"containers": [{"name": "main"}]}
        with self.assertRaises(TypeError):
            apply_overlay(base, {JOIN("containers", on="name"): ["not_a_dict"]})

    def test_join_sets_plain_when_no_base_field(self) -> None:
        """JOIN on a missing field creates the plain field."""
        base: dict[str, object] = {}
        apply_overlay(
            base,
            {JOIN("containers", on="name"): [{"name": "main"}]},
        )
        self.assertEqual(base["containers"], [{"name": "main"}])

    def test_join_does_not_mutate_overlay(self) -> None:
        base = {"containers": [{"name": "main", "image": "v1"}]}
        overlay = {JOIN("containers", on="name"): [{"name": "main", "env": "X"}]}
        overlay_copy = copy.deepcopy(overlay)
        apply_overlay(base, overlay)
        self.assertEqual(overlay, overlay_copy, "overlay should not be mutated")


class DelOperatorTest(unittest.TestCase):
    """DEL(key) removes a key from the base dict."""

    def test_del_removes_key(self) -> None:
        base: dict[str, object] = {"keep": 1, "remove": "old"}
        apply_overlay(base, {DEL("remove"): None})
        self.assertEqual(base, {"keep": 1})

    def test_del_only_key(self) -> None:
        """DEL on the only key in base empties the dict."""
        base: dict[str, object] = {"field": "value"}
        apply_overlay(base, {DEL("field"): None})
        self.assertEqual(base, {})

    def test_del_missing_key_is_noop(self) -> None:
        """DEL on a missing key is a no-op."""
        base: dict[str, object] = {"a": 1}
        apply_overlay(base, {DEL("b"): None})
        self.assertEqual(base, {"a": 1})


class ConflictResolutionTest(unittest.TestCase):
    """Operators for the same field replace earlier operations — last call wins."""

    def test_put_then_plain_replaces(self) -> None:
        """Plain key after PUT on same field: PUT is removed, plain takes over."""
        base: dict[str, object] = {PUT("tags"): ["old"]}
        apply_overlay(base, {"tags": ["new"]})
        self.assertNotIn(PUT("tags"), base)
        self.assertEqual(base["tags"], ["new"])

    def test_plain_then_put_replaces(self) -> None:
        base = {"tags": ["old"]}
        apply_overlay(base, {PUT("tags"): ["new"]})
        self.assertEqual(base, {"tags": ["new"]})

    def test_join_then_del_replaces(self) -> None:
        base: dict[str, object] = {JOIN("containers", on="name"): [{"name": "main"}]}
        apply_overlay(base, {DEL("containers"): None})
        self.assertEqual(base, {})

    def test_del_then_put_replaces(self) -> None:
        base: dict[str, object] = {DEL("field"): None}
        apply_overlay(base, {PUT("field"): "new"})
        self.assertEqual(base, {"field": "new"})

    def test_accumulate_via_set_overlay(self) -> None:
        """set_overlay calls accumulate; last operator wins per field."""
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(role, "k8s", "Pod", {"tags": ["prod"]})
        set_overlay(role, "k8s", "Pod", {PUT("tags"): ["gpu"]})

        overlay = get_overlay(role, "k8s", "Pod")
        self.assertNotIn("tags", overlay)
        self.assertEqual(overlay[PUT("tags")], ["gpu"])


# =============================================================================
# set_overlay / get_overlay
# =============================================================================


class SetGetOverlayTest(unittest.TestCase):
    def test_creates_namespace(self) -> None:
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(role, "sched", "JobSpec", {"priority": "high"})

        self.assertIn("sched", role.metadata)
        self.assertEqual(
            role.metadata["sched"]["JobSpec"],
            {"priority": "high"},
        )

    def test_merges_existing(self) -> None:
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(role, "sched", "JobSpec", {"priority": "high"})
        set_overlay(role, "sched", "JobSpec", {"oncall": "myoncall"})

        overlay = get_overlay(role, "sched", "JobSpec")
        self.assertEqual(overlay, {"priority": "high", "oncall": "myoncall"})

    def test_multiple_types(self) -> None:
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(role, "sched", "JobSpec", {"priority": "high"})
        set_overlay(role, "sched", "JobConfig", {"retries": 3})

        self.assertEqual(get_overlay(role, "sched", "JobSpec"), {"priority": "high"})
        self.assertEqual(get_overlay(role, "sched", "JobConfig"), {"retries": 3})

    def test_get_not_found(self) -> None:
        role = Role(name="w", image="img", entrypoint="e")
        self.assertEqual(get_overlay(role, "sched", "JobSpec"), {})

    def test_backwards_compat_flat(self) -> None:
        """BC: metadata["kubernetes"] = {overlay} without kind nesting."""
        role = Role(name="w", image="img", entrypoint="e")
        role.metadata["kubernetes"] = {
            "spec": {"nodeSelector": {"gpu": "true"}},
            "apiVersion": "v1",
        }
        overlay = get_overlay(role, "kubernetes", "pod")
        self.assertEqual(
            overlay,
            {"spec": {"nodeSelector": {"gpu": "true"}}, "apiVersion": "v1"},
        )

    def test_nested_format_no_false_flat_fallback(self) -> None:
        """Nested format returns {} for unknown kind (no false flat fallback)."""
        role = Role(name="w", image="img", entrypoint="e")
        role.metadata["sched"] = {
            "JobSpec": {"priority": "high"},
            "JobConfig": {"retries": 3},
        }
        self.assertEqual(get_overlay(role, "sched", "NonExistent"), {})

    def test_appdef(self) -> None:
        app = AppDef(
            name="app",
            roles=[Role(name="w", image="img", entrypoint="e")],
        )
        set_overlay(app, "sched", "JobDef", {"priority": "HIGH"})
        self.assertEqual(get_overlay(app, "sched", "JobDef"), {"priority": "HIGH"})

    def test_direct_metadata_readable(self) -> None:
        """BC: Overlays set via direct metadata access are readable."""
        role = Role(name="w", image="img", entrypoint="e")
        role.metadata["sched"] = {"JobSpec": {"priority": "high"}}
        self.assertEqual(get_overlay(role, "sched", "JobSpec"), {"priority": "high"})

    def test_replaces_non_dict_namespace(self) -> None:
        role = Role(name="w", image="img", entrypoint="e")
        role.metadata["sched"] = "stale"
        set_overlay(role, "sched", "JobSpec", {"priority": "high"})
        self.assertEqual(get_overlay(role, "sched", "JobSpec"), {"priority": "high"})

    def test_non_dict_namespace_returns_empty(self) -> None:
        role = Role(name="w", image="img", entrypoint="e")
        role.metadata["kubernetes"] = 123
        self.assertEqual(get_overlay(role, "kubernetes", "V1Pod"), {})


# =============================================================================
# validate_overlay
# =============================================================================


class ValidateOverlayTest(unittest.TestCase):
    def test_blocklist(self) -> None:
        with self.assertRaises(ValueError) as cm:
            validate_overlay(
                {"env": {"FOO": "bar"}, "priority": "high"},
                blocklist=["env", "command"],
                overlay_name="JobSpec",
            )
        self.assertIn("env", str(cm.exception))

    def test_blocklist_with_operator_key(self) -> None:
        """Operator-prefixed keys are resolved before blocklist check."""
        with self.assertRaises(ValueError):
            validate_overlay(
                {PUT("env"): {"FOO": "bar"}},
                blocklist=["env"],
                overlay_name="JobSpec",
            )

    def test_forbidden_keys(self) -> None:
        with self.assertRaises(ValueError) as cm:
            validate_overlay(
                {"retries": True},
                forbidden_keys={"retries"},
                overlay_name="JobSpec",
                suggestion="Use JobConfig.",
            )
        self.assertIn("retries", str(cm.exception))
        self.assertIn("JobConfig", str(cm.exception))

    def test_passes(self) -> None:
        validate_overlay(
            {"priority": "high"},
            blocklist=["env"],
            forbidden_keys={"retries"},
            overlay_name="JobSpec",
        )


# =============================================================================
# File loading
# =============================================================================


class LoadOverlayFileTest(unittest.TestCase):
    def test_json_bare_path(self) -> None:
        overlay = {"spec": {"nodeSelector": {"gpu": "true"}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(overlay, f)
            f.flush()
            path = f.name
        try:
            self.assertEqual(_load_overlay_file(path), overlay)
        finally:
            os.unlink(path)

    def test_json_file_uri(self) -> None:
        overlay = {"spec": {"gpu": "true"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(overlay, f)
            f.flush()
            path = f.name
        try:
            self.assertEqual(_load_overlay_file(f"file://{path}"), overlay)
        finally:
            os.unlink(path)

    def test_yaml_bare_path(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("spec:\n  nodeSelector:\n    gpu: 'true'\n")
            f.flush()
            path = f.name
        try:
            self.assertEqual(
                _load_overlay_file(path),
                {"spec": {"nodeSelector": {"gpu": "true"}}},
            )
        finally:
            os.unlink(path)

    def test_non_dict_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            path = f.name
        try:
            with self.assertRaises(ValueError):
                _load_overlay_file(path)
        finally:
            os.unlink(path)

    def test_yaml_tuple_tag(self) -> None:
        """BC: !!python/tuple tag is supported for list-replace semantics."""
        yaml_content = "spec:\n  tolerations: !!python/tuple\n    - key: gpu\n      operator: Exists\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name
        try:
            result = _load_overlay_file(path)
            self.assertIsInstance(
                result["spec"]["tolerations"],
                tuple,
                "!!python/tuple should be preserved",
            )
        finally:
            os.unlink(path)


class GetOverlayFileURITest(unittest.TestCase):
    def test_from_json_file(self) -> None:
        """get_overlay loads overlay from file URI in metadata."""
        overlay = {"pod": {"spec": {"nodeSelector": {"gpu": "true"}}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(overlay, f)
            f.flush()
            path = f.name
        try:
            role = Role(name="w", image="img", entrypoint="e")
            role.metadata["kubernetes"] = path
            self.assertEqual(
                get_overlay(role, "kubernetes", "pod"),
                {"spec": {"nodeSelector": {"gpu": "true"}}},
            )
        finally:
            os.unlink(path)

    def test_from_file_flat_format(self) -> None:
        """BC: file with flat overlay (no kind nesting) still works."""
        overlay = {"spec": {"nodeSelector": {"gpu": "true"}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(overlay, f)
            f.flush()
            path = f.name
        try:
            role = Role(name="w", image="img", entrypoint="e")
            role.metadata["kubernetes"] = path
            result = get_overlay(role, "kubernetes", "pod")
            self.assertEqual(result, {"spec": {"nodeSelector": {"gpu": "true"}}})
        finally:
            os.unlink(path)


# =============================================================================
# End-to-end usage examples
# =============================================================================


class OverlayUsageExamplesTest(unittest.TestCase):
    """Usage examples that double as integration tests.

    These show the patterns users should follow.
    """

    def test_kubernetes_node_selector(self) -> None:
        """Add a node selector to a Kubernetes pod overlay."""
        role = Role(name="trainer", image="my-image", entrypoint="train.py")
        set_overlay(
            role,
            "kubernetes",
            "V1Pod",
            {"spec": {"nodeSelector": {"accelerator": "a100"}}},
        )

        overlay = get_overlay(role, "kubernetes", "V1Pod")
        self.assertEqual(overlay["spec"]["nodeSelector"]["accelerator"], "a100")

    def test_kubernetes_merge_container_probes(self) -> None:
        """Use JOIN to add probes to an existing container by name."""
        role = Role(name="trainer", image="my-image", entrypoint="train.py")
        set_overlay(
            role,
            "kubernetes",
            "V1Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {
                            "name": "trainer-0",
                            "livenessProbe": {
                                "httpGet": {"path": "/health", "port": 8080},
                            },
                        },
                    ],
                }
            },
        )

        overlay = get_overlay(role, "kubernetes", "V1Pod")
        containers_key = JOIN("containers", on="name")
        self.assertIn(containers_key, overlay["spec"])

    def test_kubernetes_replace_containers(self) -> None:
        """Use PUT to replace the entire containers list."""
        role = Role(name="trainer", image="my-image", entrypoint="train.py")
        set_overlay(
            role,
            "kubernetes",
            "V1Pod",
            {"spec": {PUT("containers"): [{"name": "only-one"}]}},
        )

        overlay = get_overlay(role, "kubernetes", "V1Pod")
        self.assertEqual(overlay["spec"][PUT("containers")], [{"name": "only-one"}])

    def test_mast_job_definition(self) -> None:
        """Set MAST job-level overlay on AppDef."""
        app = AppDef(
            name="my_job",
            roles=[Role(name="w", image="img", entrypoint="e")],
        )
        set_overlay(
            app,
            "mast",
            "HpcJobDefinition",
            {"jobType": "OFFLINE_TRAINING", "priorityBand": "EXPEDITED"},
        )

        overlay = get_overlay(app, "mast", "HpcJobDefinition")
        self.assertEqual(overlay["jobType"], "OFFLINE_TRAINING")

    def test_overlay_accumulation(self) -> None:
        """Multiple set_overlay calls accumulate (dicts merge, lists append)."""
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(role, "k8s", "Pod", {"spec": {"tolerations": [{"key": "gpu"}]}})
        set_overlay(role, "k8s", "Pod", {"spec": {"tolerations": [{"key": "spot"}]}})
        set_overlay(role, "k8s", "Pod", {"spec": {"nodeSelector": {"gpu": "true"}}})

        overlay = get_overlay(role, "k8s", "Pod")
        self.assertEqual(
            overlay["spec"]["tolerations"],
            [{"key": "gpu"}, {"key": "spot"}],
            "lists should append across set_overlay calls",
        )
        self.assertEqual(
            overlay["spec"]["nodeSelector"], {"gpu": "true"}, "dicts should merge"
        )

    def test_del_removes_field_from_thrift_request(self) -> None:
        """DEL removes a field so the server uses its default."""
        base = {"priority": "HIGH", "timeout": 3600, "oncall": "team"}
        overlay = {DEL("timeout"): None}

        apply_overlay(base, overlay)

        self.assertNotIn("timeout", base)
        self.assertEqual(base, {"priority": "HIGH", "oncall": "team"})

    def test_full_roundtrip(self) -> None:
        """set_overlay → get_overlay → apply_overlay on a scheduler base dict."""
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(
            role,
            "mast",
            "HpcTaskGroupSpec",
            {"opecTag": "DEDICATED_ONLY", "oncallShortname": "myoncall"},
        )

        overlay = get_overlay(role, "mast", "HpcTaskGroupSpec")
        base = {"command": "/bin/echo", "arguments": [], "oncallShortname": "default"}
        apply_overlay(base, overlay)

        self.assertEqual(base["opecTag"], "DEDICATED_ONLY")
        self.assertEqual(base["oncallShortname"], "myoncall")
        self.assertEqual(base["command"], "/bin/echo")

    def test_roundtrip_with_operators(self) -> None:
        """Operators survive set_overlay and resolve during apply_overlay."""
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(
            role,
            "k8s",
            "V1Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {"name": "main", "livenessProbe": {"path": "/health"}},
                    ],
                    PUT("volumes"): [{"name": "data", "emptyDir": {}}],
                    DEL("hostNetwork"): None,
                },
            },
        )

        overlay = get_overlay(role, "k8s", "V1Pod")
        base: dict[str, Any] = {
            "spec": {
                "containers": [{"name": "main", "image": "v1"}],
                "volumes": [{"name": "old"}],
                "hostNetwork": True,
            },
        }
        apply_overlay(base, overlay)

        # JOIN merged into existing container
        self.assertEqual(
            base["spec"]["containers"][0]["livenessProbe"], {"path": "/health"}
        )
        self.assertEqual(base["spec"]["containers"][0]["image"], "v1")
        # PUT replaced volumes
        self.assertEqual(base["spec"]["volumes"], [{"name": "data", "emptyDir": {}}])
        # DEL removed hostNetwork
        self.assertNotIn("hostNetwork", base["spec"])

    def test_join_accumulates_via_set_overlay(self) -> None:
        """Multiple set_overlay calls with JOIN accumulate items."""
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(
            role,
            "k8s",
            "Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {"name": "main", "livenessProbe": {"path": "/health"}},
                    ]
                },
            },
        )
        set_overlay(
            role,
            "k8s",
            "Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {"name": "sidecar", "image": "proxy:v1"},
                    ]
                },
            },
        )

        overlay = get_overlay(role, "k8s", "Pod")
        join_key = JOIN("containers", on="name")
        self.assertEqual(
            len(overlay["spec"][join_key]), 2, "JOIN items should accumulate"
        )
        self.assertEqual(overlay["spec"][join_key][0]["name"], "main")
        self.assertEqual(overlay["spec"][join_key][1]["name"], "sidecar")

    def test_join_duplicate_merge_key_still_merges(self) -> None:
        """Accumulated JOIN items with same merge key all merge into one base item."""
        role = Role(name="w", image="img", entrypoint="e")
        set_overlay(
            role,
            "k8s",
            "Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {"name": "main", "cpu": "1"},
                    ]
                },
            },
        )
        set_overlay(
            role,
            "k8s",
            "Pod",
            {
                "spec": {
                    JOIN("containers", on="name"): [
                        {"name": "main", "memory": "1Gi"},
                    ]
                },
            },
        )

        # Accumulated: [{"name": "main", "cpu": "1"}, {"name": "main", "memory": "1Gi"}]
        overlay = get_overlay(role, "k8s", "Pod")
        base: dict[str, Any] = {
            "spec": {"containers": [{"name": "main", "image": "v1"}]},
        }
        apply_overlay(base, overlay)

        # Both accumulated items match "main" and merge into the same base item
        self.assertEqual(
            len(base["spec"]["containers"]),
            1,
            "duplicate merge keys should merge, not create duplicates",
        )
        self.assertEqual(base["spec"]["containers"][0]["cpu"], "1")
        self.assertEqual(base["spec"]["containers"][0]["memory"], "1Gi")
        self.assertEqual(base["spec"]["containers"][0]["image"], "v1")
