#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Status: Beta

You can store the scheduler run cfg (run configs) for your project
by storing them in the ``.torchxconfig`` file. Currently this file is only read
and honored when running the component from the CLI.

CLI Usage
~~~~~~~~~~~


**Scheduler Config**

#. ``cd`` into the directory where you want the ``.torchxconfig`` file to be dropped.
   The CLI only picks up ``.torchxconfig`` files from the current-working-directory (CWD)
   so chose a directory where you typically run ``torchx`` from. Typically this
   is the root of your project directory.

#. Generate the config file by running

   .. code-block:: shell-session

    $ torchx configure -s <comma,delimited,scheduler,names>

    # -- or for all registered schedulers --
    $ torchx configure

#. If you specified ``-s local_cwd,kubernetes``, you should see a ``.torchxconfig``
   file as shown below:

   .. code-block:: shell-session

    $ cat .torchxconfig
    [local_cwd]

    [kubernetes]
    queue = #FIXME:(str) Volcano queue to schedule job in

#. ``.torchxconfig`` in in INI format and the section names map to the scheduler names.
   Each section contains the run configs for the scheduler as ``$key = $value`` pairs.
   You may find that certain schedulers have empty sections, this means that
   the scheduler defines sensible defaults for all its run configs hence no run configs
   are required at runtime. If you'd like to override the default you can add them.
   **TIP:** To see all the run options for a scheduler use ``torchx runopts <scheduler_name>``.

#. The sections with ``FIXME`` placeholders are run configs that are required
   by the scheduler. Replace these with the values that apply to you.

#. **IMPORTANT:** If you are happy with the scheduler provided defaults for a particular
   run config, you **should not** redundantly specify them in ``.torchxconfig`` with the
   same default value. This is because the scheduler may decide to change the default
   value at a later date which would leave you with a stale default.

#. Now you can run your component without having to specify the scheduler run configs
   each time. Just make sure the directory you are running ``torchx`` cli from actually
   has ``.torchxconfig``!

   .. code-block:: shell-session

    $ ls .torchxconfig
    .torchxconfig

    $ torchx run -s local_cwd ./my_component.py:train

#. In addition, it is possible to specify a different config other than .torchxconfig to
   load at runtime. Requirements are that the config path is specified by enviornment
   variable TORCHXCONFIG. It also disables hierarchy loading configs from multiple
   directories as the cases otherwise.

#. User level .torchxconfig
   In addition to the project-level .torchxconfig at the root of the project directory,
   you can create one in ``$HOME/.torchxconfig`` to override or specify additional default configs.
   This config file will get overlaid on top of the one defined at the project root.

#. Config options take the following precedence (high to low):
    1. Options specified directly from the CLI
    2. If TORCHXCONFIG env variable is set, the options specified in that file
    3. If TORCHXCONFIG env variable is not set,
        a. Options specified in user level .torchxconfig (``$HOME/.torchxconfig``)
        b. Options specified in .torchxconfig
    4. Any default values in the code

   Note that malformed or unrecognized options are simply skipped and not applied

**Component Config**

You can specify component defaults by adding a section prefixed with
``component:``.

.. code-block:: ini

    [component:dist.ddp]
    j=2x8
    cpu=4

Now when you run the ``dist.ddp`` component those configs are automatically
picked up.

.. code-block:: shell-session

    $ torchx run -s local_cwd dist.ddp
    ... runs with -j 2x8 --cpu 4


**CLI Subcommand Config**

The default arguments for the ``torchx`` subcommands can be overwritten. Any
``--foo FOO`` argument can be set via the correspond ``[cli:<cmd>]`` settings
block.

For the ``run`` command you can additionally set ``component`` to set the
default component to run.

.. code-block:: ini

    [cli:run]
    component=dist.ddp
    scheduler=local_docker
    workspace=file://some_workspace


Programmatic Usage
~~~~~~~~~~~~~~~~~~~

Unlike the cli, ``.torchxconfig`` file **is not** picked up automatically
from ``CWD`` if you are programmatically running your component with :py:class:`torchx.runner.Runner`.
You'll have to manually specify the directory containing ``.torchxconfig``.

Below is an example

.. doctest:: [runner_config_example]

 from torchx.runner import get_runner
 from torchx.runner.config import apply
 import torchx.specs as specs

 def my_component(a: int) -> specs.AppDef:
    # <... component body omitted for brevity ...>
    pass

 scheduler = "local_cwd"
 cfg = {"log_dir": "/these/take/outmost/precedence"}

 apply(scheduler, cfg, dirs=["/home/bob"])  # looks for /home/bob/.torchxconfig
 get_runner().run(my_component(1), scheduler, cfg)

You may also specify multiple directories (in preceding order) which is useful when
you want to keep personal config overrides on top of a project defined default.

"""
import configparser as configparser
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO

from torchx.schedulers import get_scheduler_factories, Scheduler
from torchx.specs import CfgVal, get_type_name
from torchx.specs.api import runopt


CONFIG_FILE = ".torchxconfig"
CONFIG_PREFIX_DELIM = ":"
ENV_TORCHXCONFIG = "TORCHXCONFIG"
DEFAULT_CONFIG_DIRS = [str(Path.home()), str(Path.cwd())]

_NONE = "None"

log: logging.Logger = logging.getLogger(__name__)


def _configparser() -> configparser.ConfigParser:
    """
    Sets up the configparser and returns it. The same config parser
    should be used between dumps() and loads() methods for ser/de compatibility
    """

    config = configparser.ConfigParser()
    # if optionxform is not overridden, configparser will by default lowercase
    # the option keys because it is compatible with Windows INI files
    # which are expected to be parsed case insensitive.
    # override since torchx's runopts are case-sensitive
    # see: https://stackoverflow.com/questions/19359556/configparser-reads-capital-keys-and-make-them-lower-case
    # pyre-ignore[8]
    config.optionxform = lambda option: option

    return config


def _get_scheduler(name: str) -> Scheduler:
    schedulers = {
        **get_scheduler_factories(),
        **(
            get_scheduler_factories(
                group="torchx.schedulers.orchestrator", skip_defaults=True
            )
            or {}
        ),
    }
    if name not in schedulers:
        raise ValueError(
            f"`{name}` is not a registered scheduler. Valid scheduler names: {schedulers.keys()}"
        )
    sched = schedulers[name](session_name="_")
    return sched


def _fixme_placeholder(runopt: runopt, max_len: int = 60) -> str:
    ph = f"#FIXME:({get_type_name(runopt.opt_type)}) {runopt.help}"
    return ph if len(ph) <= max_len else f"{ph[:max_len]}..."


def dump(
    f: TextIO, schedulers: Optional[List[str]] = None, required_only: bool = False
) -> None:
    """
    Dumps a default INI-style config template containing the :py:class:torchx.specs.runopts for the
    given scheduler names into the file-like object specified by ``f``.
    If no ``schedulers`` are specified dumps all known registered schedulers.

    Optional runopts are pre-filled  with their default values.
    Required runopts are set with a ``FIXME: ...`` placeholder.
    To only dump required runopts pass ``required_only=True``.

    Each scheduler's runopts are written in the section called
    ``[{scheduler_name}]``.

    For example:

    ::

     [kubernetes]
     namespace = default
     queue = #FIXME (str)Volcano queue to schedule job in

    Raises:
        ValueError: if given a scheduler name that is not known
    """

    if schedulers:
        scheds = schedulers
    else:
        scheduler_factories = {
            **get_scheduler_factories(),
            **(
                get_scheduler_factories(
                    group="torchx.schedulers.orchestrator", skip_defaults=True
                )
                or {}
            ),
        }
        scheds = scheduler_factories.keys()

    config = _configparser()
    for sched_name in scheds:
        try:
            sched = _get_scheduler(sched_name)
        except ModuleNotFoundError:
            continue

        section = f"{sched_name}"
        config.add_section(section)

        for opt_name, opt in sched.run_opts():
            if opt.is_required:
                val = _fixme_placeholder(opt)
            else:  # not required runopts MUST have a default
                if required_only:
                    continue

                # serialize list elements with `;` delimiter (consistent with torchx cli)
                if opt.is_type_list_of_str:
                    # deal with empty or None default lists
                    if opt.default:
                        # pyre-ignore[6] opt.default type checked already as List[str]
                        val = ";".join(opt.default)
                    else:
                        val = _NONE
                elif opt.is_type_dict_of_str:
                    # deal with empty or None default lists
                    if opt.default:
                        # pyre-ignore[16] opt.default type checked already as Dict[str, str]
                        val = ";".join([f"{k}:{v}" for k, v in opt.default.items()])
                    else:
                        val = _NONE
                else:
                    val = f"{opt.default}"

            config.set(section, opt_name, val)
    config.write(f, space_around_delimiters=True)


def apply(
    scheduler: str,
    cfg: Dict[str, CfgVal],
    dirs: Optional[List[str]] = None,
) -> None:
    """
    Loads a ``.torchxconfig`` INI file from the specified directories in
    preceding order and applies the run configs for the scheduler onto
    the given ``cfg``.

    If no ``dirs`` is specified, then it looks for ``.torchxconfig`` in the
    current working directory. If a specified directory does not have ``.torchxconfig``
    then it is ignored.

    Note that the configs already present in the given ``cfg`` take precedence
    over the ones in the config file and only new configs are added. The same holds
    true for the configs loaded in list order.

    For instance if ``cfg={"foo":"bar"}`` and the config file is:

    ::

     # dir_1/.torchxconfig
     [local_cwd]
     foo = baz
     hello = world

     # dir_2/.torchxconfig
     [local_cwd]
     hello = bob


    Then after the method call, ``cfg={"foo":"bar","hello":"world"}``.
    """

    for configfile in find_configs(dirs):
        with open(configfile, "r") as f:
            load(scheduler, f, cfg)
            log.info(f"loaded configs from {configfile}")


def load_sections(
    prefix: str,
    dirs: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Loads the content of the sections in the given ``.torchxconfig`` file
    that start with the specified prefix. Returns a map of maps of the
    section name WITHOUT the prefix with the contents of the section loaded
    into a map. ``":"`` is used as the prefix delimiter.

    Example config format for specifying defaults for the builtin
    component ``dist.ddp`` is shown below:

    ::

     [component:dist.ddp]
     j = 1x2
     image = ghcr.io/foo:1

     # calling `load_sections(prefix="component")` returns
     #  {
     #    "dist.ddp": {
     #       "j":"1x2",
     #       "image":"ghcr.io/foo:1",
     #     },
     #  }


    The keys in the section must match the parameter name of the component function.
    The example below shows how to represent the various types that are allowable
    as component parameter types.

    ::

     [component:foo.bar]
     int = 1
     float = 1.2
     bool = True # or False
     str = foobar
     list = a,b,c
     map = A=B,C=D
     vararg = -a b --c=d e

     # to call the component as:
     foo.bar(
        "-a", "b", "--c=d", "e",
        int=1,
        float=1.2,
        bool=True,
        str="foobar",
        list=["a", "b", "c"],
        map={"A":"B", "C": "D"})

    """

    def strip_prefix(section_name: str) -> Optional[str]:
        # returns the section_name with the prefix removed
        # or None if the section name does not match the prefix
        idx = section_name.find(CONFIG_PREFIX_DELIM)

        # index guard (e.g ":foo" and "foo:" are bad, just return None)
        if 0 < idx < len(section_name) - 1:
            if section_name[0:idx] == prefix:
                return section_name[idx + 1 :]
        return None

    sections: Dict[str, Dict[str, str]] = {}
    for configfile in find_configs(dirs):
        with open(configfile, "r") as f:
            config = _configparser()
            config.read_file(f)
        for section_name in config.sections():
            name = strip_prefix(section_name)
            if name:
                section = sections.setdefault(name, {})
                for key, value in config.items(section_name):
                    if key not in section:
                        log.debug(
                            f"Loaded config: {prefix}.{key}={value} from {configfile}"
                        )
                        section[key] = value

    return sections


def get_configs(
    prefix: str,
    name: str,
    dirs: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Gets all the config values in the section ``["{prefix}:{name}"]``.
    Or an empty map if the section does not exist.

    Example:

    ::

     # for config file:
     # [foo:bar]
     # baz = 1

     get_configs(prefix="foo", name="bar") # returns {"baz": "1"}
     get_config(prefix="foo", name="barr") # returns {}

    """
    sections = load_sections(prefix, dirs)
    return sections.get(name, {})


def get_config(
    prefix: str,
    name: str,
    key: str,
    dirs: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Gets the config value for the ``key`` in the section ``["{prefix}:{name}"]``.
    Or ``None`` if no section or key exists

    Example:

    ::

     # for config file:
     # [foo:bar]
     # baz = 1

     get_config(prefix="foo", name="bar", key="baz") == 1
     get_config(prefix="foo", name="bar", key="bazz") == None
     get_config(prefix="foo", name="barr", key="baz") == None
     get_config(prefix="fooo", name="bar", key="baz") == None

    """
    return get_configs(prefix, name, dirs).get(key, None)


def find_configs(dirs: Optional[Iterable[str]] = None) -> List[str]:
    """
    Finds and returns the filepath to ``.torchxconfig`` files based
    on the following logic:

    1. If the environment variable ``TORCHXCONFIG`` exists, then its value
       is returned in a single-element list and the directories specified through
       the ``dirs`` parameter is NOT searched.
    2. Otherwise, a ``.torchxconfig`` file is looked for in ``dirs`` and
       the filepaths to existing config files are returned. If ``dirs`` is
       not specified or is empty then ``dirs`` defaults to ``[$HOME, $CWD]``
       where CWD is current working dir.

    """

    config = os.getenv(ENV_TORCHXCONFIG)
    if config is not None:
        if not config:
            return []
        configfile = Path(config)
        if not configfile.is_file():
            raise FileNotFoundError(
                f"`{ENV_TORCHXCONFIG}={config}` does not exist or is not a file."
            )
        return [str(configfile)]
    else:
        config_files = []
        if not dirs:
            dirs = DEFAULT_CONFIG_DIRS
        for d in dirs:
            configfile = Path(d) / CONFIG_FILE
            if os.access(configfile, os.R_OK):
                config_files.append(str(configfile))
    return config_files


def load(scheduler: str, f: TextIO, cfg: Dict[str, CfgVal]) -> None:
    """
    loads the section ``[{scheduler}]`` from the given
    configfile ``f`` (in .INI format) into the provided ``runcfg``, only adding
    configs that are NOT currently in the given ``runcfg`` (e.g. does not
    override existing values in ``runcfg``). If no section is found, does nothing.
    """

    config = _configparser()
    config.read_file(f)

    runopts = _get_scheduler(scheduler).run_opts()

    section = f"{scheduler}"
    if config.has_section(section):
        for name, value in config.items(section):
            if name in cfg.keys():
                # DO NOT OVERRIDE existing configs
                continue

            if value == _NONE:
                # should map to None (not str 'None')
                # this also handles empty or None lists
                cfg[name] = None
            else:
                opt = runopts.get(name)

                if opt is None:
                    log.warning(
                        f"`{name} = {value}` was declared in the [{section}] section "
                        f" of the config file but is not a runopt of `{scheduler}` scheduler."
                        f" Remove the entry from the config file to no longer see this warning"
                    )
                else:
                    if opt.opt_type is bool:
                        # need to handle bool specially since str -> bool is based on
                        # str emptiness not value (e.g. bool("False") == True)
                        cfg[name] = config.getboolean(section, name)
                    elif opt.is_type_list_of_str:
                        cfg[name] = value.split(";")
                    elif opt.is_type_dict_of_str:
                        cfg[name] = {
                            s.split(":", 1)[0]: s.split(":", 1)[1]
                            for s in value.replace(",", ";").split(";")
                        }
                    else:
                        # pyre-ignore[29]
                        cfg[name] = opt.opt_type(value)
