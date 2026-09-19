# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import warnings

from torchx.components import dist
from torchx.components.component_test_base import ComponentTestCase
from torchx.specs import builders, finder


class TorchrunTest(ComponentTestCase):
    def test_torchrun(self) -> None:
        self.validate(dist, "torchrun")

    def test_torchrun_mounts(self) -> None:
        app = dist.torchrun(
            script="foo.py", mounts=["type=bind", "src=/dst", "dst=/dst", "readonly"]
        )
        self.assertEqual(len(app.roles[0].mounts), 1)

    def test_torchrun_parse_j(self) -> None:
        """test samples for different forms of -j {nnodes}x{nproc_per_node}"""
        self.assertEqual(dist.parse_nnodes("2"), (1, 1, 2, "1"))  # nproc_per_node is 2
        self.assertEqual(dist.parse_nnodes("1x2"), (1, 1, 2, "1"))
        self.assertEqual(dist.parse_nnodes("1:2x3"), (1, 2, 3, "1:2"))

    def test_torchrun_parse_j_exception(self) -> None:
        j_exception = ["1x", "x2", ":3", ":2x1", "1x2:3"]
        for j in j_exception:
            with self.assertRaises(ValueError):
                dist.parse_nnodes(j)

    def test_torchrun_debug(self) -> None:
        app = dist.torchrun(script="foo.py", debug=True)
        env = app.roles[0].env
        for k, v in dist._TORCH_DEBUG_FLAGS.items():
            self.assertEqual(env[k], v)

    def test_torchrun_metadata(self) -> None:
        metadata = {"key": "value"}
        app = dist.torchrun(script="foo.py", metadata=metadata)
        for k, v in metadata.items():
            self.assertEqual(app.metadata[k], v)
        self.assertEqual(len(metadata), len(app.metadata))

    def test_torchrun_does_not_mutate_caller_env_and_metadata(self) -> None:
        env = {"FOO": "bar"}
        metadata = {"key": "value"}
        app = dist.torchrun(script="foo.py", env=env, metadata=metadata)
        self.assertEqual({"FOO": "bar"}, env)
        self.assertEqual({"key": "value"}, metadata)
        # the component's additions go to the AppDef, not the caller's dict
        self.assertIn("TORCHX_TRACKING_EXPERIMENT_NAME", app.roles[0].env)

    def test_torchrun_rdzv_backend_static(self) -> None:
        rdzv_conf = "join_timeout=600,close_timeout=600,timeout=600"
        app = dist.torchrun(script="foo.py", rdzv_backend="static", rdzv_conf=rdzv_conf)
        cmd = app.roles[0].args[1]
        self.assertTrue(f"--rdzv_conf {rdzv_conf}" in cmd)
        self.assertTrue("--rdzv_backend static" in cmd)
        self.assertTrue("--node_rank" in cmd)

    def test_named_host_gpu_inference(self) -> None:
        app = dist.torchrun(script="train.py", h="gpu.small", j="2")
        self.assertEqual(app.roles[0].num_replicas, 2)
        self.assertIn("--nproc_per_node 1", app.roles[0].args[1])
        with self.assertRaisesRegex(ValueError, "nproc_per_node cannot be inferred"):
            dist.torchrun(script="train.py", h="cpu.small", j="2")

    def test_named_host_reports_a_malformed_j(self) -> None:
        for j in ["1:2", "1x", ":3", "1x2:3"]:
            with self.subTest(j=j):
                with self.assertRaisesRegex(ValueError, "Invalid format for -j"):
                    dist.torchrun(script="train.py", h="gpu.small", j=j)

    def test_elastic_topology(self) -> None:
        app = dist.torchrun(script="train.py", j="2:4x3", rdzv_port=29501)
        self.assertEqual(app.roles[0].min_replicas, 2)
        self.assertEqual(app.roles[0].num_replicas, 4)
        self.assertIn("--nnodes 2:4", app.roles[0].args[1])
        self.assertIn("--nproc_per_node 3", app.roles[0].args[1])
        self.assertIn(":29501", app.roles[0].args[1])

    def test_torchrun_call_by_module_or_script_no_name(self) -> None:
        appdef = dist.torchrun(script="foo/bar.py")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = dist.torchrun("-a", "b", script="foo/bar.py")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = dist.torchrun(m="foo.bar")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = dist.torchrun("-a", "b", m="foo.bar")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        with self.assertRaises(ValueError):
            dist.torchrun()

        with self.assertRaises(ValueError):
            dist.torchrun(m="foo.bar", script="foo/bar.py")

    def test_torchrun_call_by_module_or_script_with_name(self) -> None:
        appdef = dist.torchrun(script="foo/bar.py", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = dist.torchrun("-a", "b", script="foo/bar.py", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])

        appdef = dist.torchrun(m="foo.bar", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = dist.torchrun("-a", "b", m="foo.bar", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("trial_1", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

    def test_torchrun_call_by_module_or_script_with_experiment_name(self) -> None:
        appdef = dist.torchrun(script="foo/bar.py", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = dist.torchrun("-a", "b", script="foo/bar.py", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = dist.torchrun(m="foo.bar", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = dist.torchrun("-a", "b", m="foo.bar", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("bar", appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"])
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

    def test_torchrun_call_by_module_or_script_with_run_name(self) -> None:
        appdef = dist.torchrun(script="foo/bar.py", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )

        appdef = dist.torchrun("-a", "b", script="foo/bar.py", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )

        appdef = dist.torchrun(m="foo.bar", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )

        appdef = dist.torchrun("-a", "b", m="foo.bar", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            "default-experiment",
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )
        self.assertEqual(
            "trial_1",
            appdef.roles[0].env["TORCHX_TRACKING_RUN_NAME"],
        )


class DeprecatedAliasesTest(ComponentTestCase):
    def test_resolve_and_forward(self) -> None:
        for name in ("ddp", "spmd"):
            with self.subTest(component=name):
                self.validate(dist, name)
                component = finder.get_component(f"dist.{name}")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    app = component.fn("--epochs", "2", script="train.py")
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertIn(f"dist.{name} is deprecated", str(caught[0].message))
                self.assertIn("dist.torchrun", str(caught[0].message))
                self.assertEqual(caught[0].filename, __file__)
                if name == "spmd":
                    expected = dist.torchrun(
                        "--epochs", "2", script="train.py", h="gpu.small", j="1x1"
                    )
                else:
                    expected = dist.torchrun("--epochs", "2", script="train.py")
                self.assertEqual(app, expected)

    def test_ddp_forwards_options_and_preserves_bare_j(self) -> None:
        with self.assertWarnsRegex(UserWarning, "dist.ddp.*dist.torchrun"):
            app = dist.ddp(
                "--epochs",
                "2",
                m="example.train",
                image="trainer:latest",
                name="experiment/trial",
                h="gpu.small",
                cpu=4,
                gpu=2,
                memMB=2048,
                j="2",
                env={"MODE": "test"},
                metadata={"owner": "example"},
                max_retries=2,
                rdzv_port=29501,
                rdzv_backend="static",
                rdzv_conf="timeout=600",
                mounts=["type=bind", "src=/data", "dst=/data"],
                debug=True,
                tee=1,
            )
        expected = dist.torchrun(
            "--epochs",
            "2",
            m="example.train",
            image="trainer:latest",
            name="experiment/trial",
            h="gpu.small",
            cpu=4,
            gpu=2,
            memMB=2048,
            j="1x2",
            env={"MODE": "test"},
            metadata={"owner": "example"},
            max_retries=2,
            rdzv_port=29501,
            rdzv_backend="static",
            rdzv_conf="timeout=600",
            mounts=["type=bind", "src=/data", "dst=/data"],
            debug=True,
            tee=1,
        )
        self.assertEqual(app, expected)
        self.assertEqual(app.roles[0].num_replicas, 1)
        self.assertIn("--nproc_per_node 2", app.roles[0].args[1])

    def test_spmd_forwards_options_and_infers_gpus(self) -> None:
        with self.assertWarnsRegex(UserWarning, "dist.spmd.*dist.torchrun"):
            app = dist.spmd(
                "--epochs",
                "2",
                m="example.train",
                image="trainer:latest",
                name="experiment/trial",
                h="gpu.small",
                j="2",
                env={"MODE": "test"},
                metadata={"owner": "example"},
                max_retries=2,
                mounts=["type=bind", "src=/data", "dst=/data"],
                debug=True,
            )
        expected = dist.torchrun(
            "--epochs",
            "2",
            m="example.train",
            image="trainer:latest",
            name="experiment/trial",
            h="gpu.small",
            j="2",
            env={"MODE": "test"},
            metadata={"owner": "example"},
            max_retries=2,
            mounts=["type=bind", "src=/data", "dst=/data"],
            debug=True,
        )
        self.assertEqual(app, expected)
        self.assertEqual(app.roles[0].num_replicas, 2)
        self.assertIn("--nproc_per_node 1", app.roles[0].args[1])

    def test_cli_arguments(self) -> None:
        for name in ("ddp", "spmd", "torchrun"):
            with self.subTest(component=name):
                component = finder.get_component(f"dist.{name}")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    app = builders.materialize_appdef(
                        component.fn,
                        ["--script", "train.py", "-j", "2x3", "--", "--epochs", "2"],
                    )
                self.assertEqual(app.roles[0].num_replicas, 2)
                self.assertIn("--nproc_per_node 3", app.roles[0].args[1])
                self.assertIn("--epochs 2", app.roles[0].args[1])
                deprecations = [w for w in caught if "[Deprecated]" in str(w.message)]
                self.assertEqual(len(deprecations), 0 if name == "torchrun" else 1)
