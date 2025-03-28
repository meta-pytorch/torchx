# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torchx.components.component_test_base import ComponentTestCase
from torchx.components.dist import _TORCH_DEBUG_FLAGS, ddp, parse_nnodes, spmd


class DDPTest(ComponentTestCase):
    def test_ddp(self) -> None:
        import torchx.components.dist as dist

        self.validate(dist, "ddp")

    def test_ddp_mounts(self) -> None:
        app = ddp(
            script="foo.py", mounts=["type=bind", "src=/dst", "dst=/dst", "readonly"]
        )
        self.assertEqual(len(app.roles[0].mounts), 1)

    def test_ddp_parse_j(self) -> None:
        """test samples for different forms of -j {nnodes}x{nproc_per_node}"""
        self.assertEqual(parse_nnodes("2"), (1, 1, 2, "1"))  # nproc_per_node is 2
        self.assertEqual(parse_nnodes("1x2"), (1, 1, 2, "1"))
        self.assertEqual(parse_nnodes("1:2x3"), (1, 2, 3, "1:2"))

    def test_ddp_parse_j_exception(self) -> None:
        j_exception = ["1x", "x2", ":3", ":2x1", "1x2:3"]
        for j in j_exception:
            with self.assertRaises(ValueError):
                parse_nnodes(j)

    def test_ddp_debug(self) -> None:
        app = ddp(script="foo.py", debug=True)
        env = app.roles[0].env
        for k, v in _TORCH_DEBUG_FLAGS.items():
            self.assertEqual(env[k], v)

    def test_ddp_metadata(self) -> None:
        metadata = {"key": "value"}
        app = ddp(script="foo.py", metadata=metadata)
        for k, v in metadata.items():
            self.assertEqual(app.metadata[k], v)
        self.assertEqual(len(metadata), len(app.metadata))

    def test_ddp_rdzv_backend_static(self) -> None:
        rdzv_conf = "join_timeout=600,close_timeout=600,timeout=600"
        app = ddp(script="foo.py", rdzv_backend="static", rdzv_conf=rdzv_conf)
        cmd = app.roles[0].args[1]
        self.assertTrue(f"--rdzv_conf {rdzv_conf}" in cmd)
        self.assertTrue("--rdzv_backend static" in cmd)
        self.assertTrue("--node_rank" in cmd)


class SpmdTest(ComponentTestCase):
    def test_validate_spmd(self) -> None:
        import torchx.components.dist as dist

        self.validate(dist, "ddp")

    def test_spmd_metadata(self) -> None:
        metadata = {"key": "value"}
        app = spmd(script="foo.py", metadata=metadata)
        for k, v in metadata.items():
            self.assertEqual(app.metadata[k], v)
        self.assertEqual(len(metadata), len(app.metadata))

    def test_spmd_call_by_module_or_script_no_name(self) -> None:
        appdef = spmd(script="foo/bar.py")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd("-a", "b", script="foo/bar.py")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd(m="foo.bar")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd("-a", "b", m="foo.bar")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        with self.assertRaises(ValueError):
            spmd()

        with self.assertRaises(ValueError):
            spmd(m="foo.bar", script="foo/bar.py")

    def test_spmd_call_by_module_or_script_with_name(self) -> None:
        appdef = spmd(script="foo/bar.py", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", script="foo/bar.py", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])

        appdef = spmd(m="foo.bar", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", m="foo.bar", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

    def test_spmd_call_by_module_or_script_with_experiment_name(self) -> None:
        appdef = spmd(script="foo/bar.py", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", script="foo/bar.py", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd(m="foo.bar", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", m="foo.bar", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

    def test_spmd_call_by_module_or_script_with_run_name(self) -> None:
        appdef = spmd(script="foo/bar.py", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )

        appdef = spmd("-a", "b", script="foo/bar.py", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )

        appdef = spmd(m="foo.bar", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )

        appdef = spmd("-a", "b", m="foo.bar", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )
