#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from datetime import datetime
from io import StringIO
from typing import Dict, Iterable, List, Mapping, Optional
from unittest.mock import patch

from torchx.runner.config import (
    apply,
    dump,
    ENV_TORCHXCONFIG,
    find_configs,
    get_config,
    get_configs,
    load,
    load_sections,
)
from torchx.schedulers import get_scheduler_factories, Scheduler
from torchx.schedulers.api import DescribeAppResponse, ListAppResponse, Stream
from torchx.specs import AppDef, AppDryRunInfo, CfgVal, runopts
from torchx.test.fixtures import TestWithTmpDir
from torchx.workspace import Workspace


class TestScheduler(Scheduler):
    def __init__(self, session_name: str) -> None:
        super().__init__("test", session_name)

    def schedule(self, dryrun_info: AppDryRunInfo) -> str:
        raise NotImplementedError()

    def _submit_dryrun(self, app: AppDef, cfg: Mapping[str, CfgVal]) -> AppDryRunInfo:
        raise NotImplementedError()

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        raise NotImplementedError()

    def _cancel_existing(self, app_id: str) -> None:
        raise NotImplementedError()

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        raise NotImplementedError()

    def list(self) -> List[ListAppResponse]:
        raise NotImplementedError()

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "i",
            type_=int,
            default=1,
            help="an int option",
        )
        opts.add(
            "f",
            type_=float,
            default=1.2,
            help="a float option",
        )
        opts.add(
            "s",
            type_=str,
            default="foobar",
            help="an str option",
        )
        opts.add(
            "bTrue",
            type_=bool,
            default=True,
            help="an bool_true option",
        )
        opts.add(
            "bFalse",
            type_=bool,
            default=False,
            help="an bool_false option",
        )
        opts.add(
            "l",
            type_=list[str],
            default=["a", "b", "c"],
            help="a list option",
        )
        opts.add(
            "l_typing",
            type_=List[str],
            default=["a", "b", "c"],
            help="a typing.List option",
        )
        opts.add(
            "l_none",
            type_=list[str],
            default=None,
            help="a None list option",
        )
        opts.add(
            "d",
            type_=dict[str, str],
            default={"foo": "bar"},
            help="a dict option",
        )
        opts.add(
            "d_typing",
            type_=Dict[str, str],
            default={"foo": "bar"},
            help="a typing.Dict option",
        )
        opts.add(
            "d_none",
            type_=Dict[str, str],
            default=None,
            help="a None dict option",
        )
        opts.add(
            "empty",
            type_=str,
            default=None,
            help="an empty option",
        )
        return opts


_CONFIG = """#
[local_cwd]
log_dir = /home/bob/logs
prepend_cwd = True
"""

_CONFIG_INVALID = """#
[test]
a_run_opt_that = does_not_exist
s = option_that_exists
"""

_TEAM_CONFIG = """#
[test]
s = team_default
i = 50
f = 1.2
d = a:b,c:d
d_none= x:y
"""

_MY_CONFIG = """#
[test]
s = my_default
i = 100
l = abc;def
l_typing = ghi;jkl
d = a:b,c:d
d_typing = e:f,g:h
"""

_MY_CONFIG2 = """#
[test]
s = my_default
i = 200
"""

_COMPONENT_CONFIG_0 = """#
[component:dist.ddp]
image = barbaz
script_args = a b c --d=e --f g

[component:utils.echo]
msg = abracadabra

# bad section
[component:]
foo = bar

[:bad]
leads = with delimiter

[local]
1 = 2
"""

_COMPONENT_CONFIG_1 = """#
[component:dist.ddp]
j = 1x2
image = foobar
env = A=B,C=D

[component:utils.touch]
file = tmp.txt
"""

TORCHX_DEFAULT_CONFIG_DIRS = "torchx.runner.config.DEFAULT_CONFIG_DIRS"
TORCHX_GET_SCHEDULER_FACTORIES = "torchx.runner.config.get_scheduler_factories"


class ConfigTest(TestWithTmpDir):
    def setUp(self) -> None:
        super().setUp()
        self.write(".torchxconfig", _TEAM_CONFIG)
        self.write(os.path.join("home", "bob", ".torchxconfig"), _MY_CONFIG)
        self.write("another_torchx_config", _MY_CONFIG2)

    def test_load_component_defaults(self) -> None:
        configdir0 = self.tmpdir / "test_load_component_defaults" / "0"
        configdir1 = self.tmpdir / "test_load_component_defaults" / "1"
        configdir0.mkdir(parents=True, exist_ok=True)
        configdir1.mkdir(parents=True, exist_ok=True)
        self.write(str(configdir0 / ".torchxconfig"), _COMPONENT_CONFIG_0)
        self.write(str(configdir1 / ".torchxconfig"), _COMPONENT_CONFIG_1)

        defaults = load_sections(
            prefix="component", dirs=[str(configdir0), str(configdir1)]
        )

        self.assertDictEqual(
            {
                "dist.ddp": {
                    "j": "1x2",
                    "image": "barbaz",
                    "env": "A=B,C=D",
                    "script_args": "a b c --d=e --f g",
                },
                "utils.echo": {
                    "msg": "abracadabra",
                },
                "utils.touch": {"file": "tmp.txt"},
            },
            defaults,
        )

    def test_get_configs(self) -> None:
        configdir0 = self.tmpdir / "test_load_component_defaults" / "0"
        configdir1 = self.tmpdir / "test_load_component_defaults" / "1"
        configdir0.mkdir(parents=True, exist_ok=True)
        configdir1.mkdir(parents=True, exist_ok=True)
        self.write(str(configdir0 / ".torchxconfig"), _COMPONENT_CONFIG_0)
        self.write(str(configdir1 / ".torchxconfig"), _COMPONENT_CONFIG_1)
        dirs = [str(configdir0), str(configdir1)]

        self.assertDictEqual(
            {},
            get_configs(
                prefix="component",
                name="non-existent",
                dirs=dirs,
            ),
        )
        self.assertDictEqual(
            {
                "j": "1x2",
                "image": "barbaz",
                "env": "A=B,C=D",
                "script_args": "a b c --d=e --f g",
            },
            get_configs(
                prefix="component",
                name="dist.ddp",
                dirs=dirs,
            ),
        )

    def test_no_config(self) -> None:
        config_dir = self.tmpdir
        with patch.dict(os.environ, {ENV_TORCHXCONFIG: str("")}):
            configs = find_configs(dirs=[str(config_dir)])
            self.assertEqual([], configs)

    def test_find_configs(self) -> None:
        config_dir = self.tmpdir
        cwd_dir = config_dir / "cwd"
        home_dir = config_dir / "home"
        empty_home_dir = config_dir / "home_empty"
        custom_dir = config_dir / "custom"

        cwd_dir.mkdir()
        custom_dir.mkdir()

        base_config = self.touch(str(config_dir / ".torchxconfig"))
        cwd_config = self.touch(str(cwd_dir / ".torchxconfig"))
        home_config = self.touch(str(home_dir / ".torchxconfig"))
        override_config = self.touch(str(custom_dir / ".torchxconfig"))

        # should find configs in the specified dirs
        configs = find_configs(dirs=[str(config_dir)])
        self.assertEqual([str(base_config)], configs)

        # should find config in HOME and cwd if no dirs is specified
        with patch(TORCHX_DEFAULT_CONFIG_DIRS, [str(home_dir), str(cwd_dir)]):
            configs = find_configs()
            self.assertEqual([str(home_config), str(cwd_config)], configs)

        # should find config in cwd only if no $HOME/ .torchxconfig
        with patch(TORCHX_DEFAULT_CONFIG_DIRS, [str(empty_home_dir), str(cwd_dir)]):
            configs = find_configs()
            self.assertEqual([str(cwd_config)], configs)

        # if TORCHXCONFIG env var exists then should just return the config specified
        with patch.dict(os.environ, {ENV_TORCHXCONFIG: str(override_config)}):
            configs = find_configs(dirs=[str(config_dir)])
            self.assertEqual([str(override_config)], configs)

        # if TORCHXCONFIG points to a non-existing file, then assert exception
        with patch.dict(
            os.environ,
            {ENV_TORCHXCONFIG: str(config_dir / ".torchxconfig_nonexistent")},
        ):
            with self.assertRaises(FileNotFoundError):
                find_configs(dirs=[str(config_dir)])

    def test_get_config(self) -> None:
        configdir0 = self.tmpdir / "test_load_component_defaults" / "0"
        configdir1 = self.tmpdir / "test_load_component_defaults" / "1"
        configdir0.mkdir(parents=True, exist_ok=True)
        configdir1.mkdir(parents=True, exist_ok=True)
        self.write(str(configdir0 / ".torchxconfig"), _COMPONENT_CONFIG_0)
        self.write(str(configdir1 / ".torchxconfig"), _COMPONENT_CONFIG_1)
        dirs = [str(configdir0), str(configdir1)]

        self.assertEqual(
            "1x2",
            get_config(prefix="component", name="dist.ddp", key="j", dirs=dirs),
        )
        self.assertEqual(
            "barbaz",
            get_config(prefix="component", name="dist.ddp", key="image", dirs=dirs),
        )
        self.assertIsNone(
            get_config(prefix="component", name="dist.ddp", key="badkey", dirs=dirs),
        )
        self.assertIsNone(
            get_config(prefix="component", name="badname", key="j", dirs=dirs),
        )
        self.assertIsNone(
            get_config(prefix="badprefix", name="dist.ddp", key="j", dirs=dirs),
        )

        # check that if TORCHXCONFIG is set then only that config is loaded
        override_config = self.tmpdir / ".torchxconfig_custom"
        override_config_contents = """
[component:dist.ddp]
image = foobar_custom
        """
        self.write(str(override_config), override_config_contents)

        with patch.dict(os.environ, {ENV_TORCHXCONFIG: str(override_config)}):
            self.assertDictEqual(
                {"image": "foobar_custom"},
                get_configs(prefix="component", name="dist.ddp", dirs=dirs),
            )

    def test_load(self) -> None:
        cfg = {}
        load(scheduler="local_cwd", f=StringIO(_CONFIG), cfg=cfg)
        self.assertEqual("/home/bob/logs", cfg.get("log_dir"))
        self.assertEqual(True, cfg.get("prepend_cwd"))

    def test_no_override_load(self) -> None:
        cfg: Dict[str, CfgVal] = {"log_dir": "/foo/bar", "debug": 1}

        load(scheduler="local_cwd", f=StringIO(_CONFIG), cfg=cfg)
        self.assertEqual("/foo/bar", cfg.get("log_dir"))
        self.assertEqual(1, cfg.get("debug"))
        self.assertEqual(True, cfg.get("prepend_cwd"))

    @patch(
        TORCHX_GET_SCHEDULER_FACTORIES,
        return_value={"test": TestScheduler},
    )
    def test_apply_default(self, _) -> None:
        with patch(
            TORCHX_DEFAULT_CONFIG_DIRS,
            [str(self.tmpdir / "home"), str(self.tmpdir)],
        ):
            cfg: Dict[str, CfgVal] = {"s": "runtime_value"}
            apply(scheduler="test", cfg=cfg)

            self.assertEqual("runtime_value", cfg.get("s"))
            self.assertEqual(50, cfg.get("i"))
            self.assertEqual(1.2, cfg.get("f"))
            self.assertEqual({"a": "b", "c": "d"}, cfg.get("d"))
            self.assertEqual({"x": "y"}, cfg.get("d_none"))

    @patch(
        TORCHX_GET_SCHEDULER_FACTORIES,
        return_value={"test": TestScheduler},
    )
    def test_apply_dirs(self, _) -> None:
        cfg: Dict[str, CfgVal] = {"s": "runtime_value"}
        apply(
            scheduler="test",
            cfg=cfg,
            dirs=[str(self.tmpdir / "home" / "bob"), str(self.tmpdir)],
        )
        self.assertEqual("runtime_value", cfg.get("s"))
        self.assertEqual(100, cfg.get("i"))
        self.assertEqual(1.2, cfg.get("f"))
        self.assertEqual({"a": "b", "c": "d"}, cfg.get("d"))
        self.assertEqual({"e": "f", "g": "h"}, cfg.get("d_typing"))
        self.assertEqual(["abc", "def"], cfg.get("l"))
        self.assertEqual(["ghi", "jkl"], cfg.get("l_typing"))

    def test_dump_invalid_scheduler(self) -> None:
        with self.assertRaises(ValueError):
            dump(f=StringIO(), schedulers=["does-not-exist"])

    @patch(
        TORCHX_GET_SCHEDULER_FACTORIES,
        return_value={"test": TestScheduler},
    )
    def test_dump_only_required(self, _) -> None:
        sfile = StringIO()

        # test scheduler has no required options hence expect empty string
        dump(f=sfile, required_only=True)

        cfg = {}
        sfile.seek(0)
        load(scheduler="test", f=sfile, cfg=cfg)

        # empty
        self.assertEqual({}, cfg)

    @patch(
        TORCHX_GET_SCHEDULER_FACTORIES,
        return_value={"test": TestScheduler},
    )
    def test_load_invalid_runopt(self, _) -> None:
        cfg = {}
        load(
            scheduler="test",
            f=StringIO(_CONFIG_INVALID),
            cfg=cfg,
        )
        # options in the config file but not in runopts
        # should be ignored (we shouldn't throw an error since
        # this makes things super hard to guarantee BC - stale config file will fail
        # to run, we don't want that)

        self.assertEqual("option_that_exists", cfg.get("s"))

    def test_load_no_section(self) -> None:
        cfg = {}
        load(
            scheduler="local_cwd",
            f=StringIO(),
            cfg=cfg,
        )
        # is empty
        self.assertEqual({}, cfg)

        load(
            scheduler="local_cwd",
            f=StringIO("[scheduler_args.local_cwd]\n"),
            cfg=cfg,
        )
        # still empty
        self.assertEqual({}, cfg)

    @patch(
        TORCHX_GET_SCHEDULER_FACTORIES,
        return_value={"test": TestScheduler},
    )
    def test_dump_and_load_all_runopt_types(self, _) -> None:
        sfile = StringIO()
        dump(sfile)

        sfile.seek(0)

        cfg = {}
        load(scheduler="test", f=sfile, cfg=cfg)

        # all runopts in the TestScheduler have defaults, just check against those
        for opt_name, opt in TestScheduler("test").run_opts():
            self.assertEqual(opt.default, cfg.get(opt_name))

    def test_dump_and_load_all_registered_schedulers(self) -> None:
        # dump all the runopts for all registered schedulers
        # load them back as run cfg
        # checks that all scheduler run options are accounted for

        sfile = StringIO()
        dump(sfile)

        scheduler_factories = get_scheduler_factories()

        for sched_name, sched in scheduler_factories.items():
            sfile.seek(0)  # reset the file pos
            cfg = {}
            try:
                load(scheduler=sched_name, f=sfile, cfg=cfg)
            except ModuleNotFoundError:
                # just test the ones that have been installed
                continue

            for opt_name, _ in sched("test").run_opts():
                self.assertTrue(
                    opt_name in cfg,
                    f"missing {opt_name} in {sched} run opts with cfg {cfg}",
                )

    def test_get_workspace_config(self) -> None:
        configdir = self.tmpdir
        self.write(
            str(configdir / ".torchxconfig"),
            """#
[cli:run]
workspace =
    /home/foo/third-party/verl: verl
    /home/foo/bar/scripts/.torchxconfig: verl/.torchxconfig
    /home/foo/baz:
""",
        )

        workspace_config = get_config(
            prefix="cli", name="run", key="workspace", dirs=[str(configdir)]
        )
        self.assertIsNotNone(workspace_config)

        workspace = Workspace.from_str(workspace_config)
        self.assertDictEqual(
            {
                "/home/foo/third-party/verl": "verl",
                "/home/foo/bar/scripts/.torchxconfig": "verl/.torchxconfig",
                "/home/foo/baz": "",
            },
            workspace.projects,
        )
