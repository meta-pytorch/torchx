# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Defines methods for structured (higher order) component argument parsing.
Use the functionalities defined in this module to author components
in such a way that the structured component arguments are consistent across the board.

A structured component argument is a function argument to a component (a function that returns an ``AppDef``)
that is human-friendly (less typing in the CLI or better readability) but technically embeds multiple
primitive arguments.

Examples:

#. ``-j {NNODES}x{NPROC_PER_NODE}`` (e.g. ``-j 1x2``): Compactly represents the number of
    nodes and number of workers per node for a distributed application. Otherwise would've had to be taken
    as two separate arguments: ``--nnodes 1 --nproc_per_node 8``
#. ``--name {EXPERIMENT_NAME}/{RUN_NAME}`` (e.g. ``--name t5_modeling/bfloat16_trial``): Uses a single
    ``--name`` parameter to parse experiment and run names for logging experiments and trials (runs)
    with an experiment tracker. The ``/`` delimiter is a natural way to group runs within experiments.

"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torchx import specs


@dataclass
class StructuredNameArgument:
    experiment_name: str
    run_name: str

    def __str__(self) -> str:
        return f"{self.experiment_name or ''}/{self.run_name}"

    @staticmethod
    def parse_from(
        name: str,
        m: Optional[str] = None,
        script: Optional[str] = None,
        default_experiment_name: str = "default-experiment",
    ) -> "StructuredNameArgument":
        """
        Creates an :py:class:`ArgName` from the component arguments:
        ``name``, ``m`` (main module), ``script`` (main script).

        The ``name`` MUST be of the form ``{EXPERIMENT_NAME}/{RUN_NAME}`` where either or both
        ``{EXPERIMENT_NAME}`` and ``{RUN_NAME}`` may be left empy.
        However, the ``name`` must include the ``/`` delimiter.

        For instance:

        #. ``foo/``: specifies an experiment name but no run name
        #. ``/bar``: specifies a run name but no experiment name
        #. ``foo``: specifieds a run name but no experiment name
        #. ``foo/bar``: specifies both experiment and run names
        #. ``/``: does not specify experiment nor run name

        If the run name is left empty then one is derived from either the ``m``
        or ``script``, whichever is not null.

        .. important::
            Exactly one of ``m`` (main module)` or ``script`` (path to script) must be provided.
            If both or neither are provided, then this function throws.


        Examples:
            .. doctest::

                >>> import os
                >>> from torchx.components.structured_arg import StructuredNameArgument
                >>> StructuredNameArgument.parse_from(name="foo/bar", script="bar/baz.py")
                StructuredNameArgument(experiment_name='foo', run_name='bar')

                >>> StructuredNameArgument.parse_from(name="foo/", script="bar/baz.py")
                StructuredNameArgument(experiment_name='foo', run_name='baz')

                >>> StructuredNameArgument.parse_from(name="/bar", script="bar/baz.py")
                StructuredNameArgument(experiment_name='default-experiment', run_name='bar')

                >>> StructuredNameArgument.parse_from(name="foobar", m="foo.bar")
                StructuredNameArgument(experiment_name='default-experiment', run_name='foobar')

                >>> StructuredNameArgument.parse_from(name="foo/bar", m="foo.bar.baz")
                StructuredNameArgument(experiment_name='foo', run_name='bar')

                >>> StructuredNameArgument.parse_from(name="foo/", m="foo.bar.baz")
                StructuredNameArgument(experiment_name='foo', run_name='baz')

                >>> StructuredNameArgument.parse_from(name="/bar", m="foo.bar.baz")
                StructuredNameArgument(experiment_name='default-experiment', run_name='bar')

                >>> StructuredNameArgument.parse_from(name="foo/bar")
                Traceback (most recent call last):
                    ...
                ValueError: No main module or script specified.


        Arguments:
            name: ``{EXPERIMENT_NAME}/{RUN_NAME}``, ``/{RUN_NAME}``, or ``{EXPERIMENT_NAME}/``, or ``{RUN_NAME}``
            m: the main module (e.g. ``foo.bar.baz`` for ``foo/bar/baz.py``)
            script: path to the main script


        Raises:
            ValueError: if both ``m`` and ``script`` are empty or both are non-empty.
            ValueError: if the ``name`` does not contain the experiment/run name delimiter: ``/``.
        """
        if not m and not script:
            raise ValueError(
                "No main module or script specified. Specify either a main module or a script path"
            )
        if m and script:
            raise ValueError(
                "Both main module and script set. Specify exactly one of: main module or script, but not both"
            )

        run_name = ""
        experiment_name = ""

        delim_idx = name.find("/")
        # just assume that name is the run_name (experiment name should default)
        if delim_idx < 0:
            run_name = name
        elif delim_idx >= 0 and delim_idx < len(name) - 1:
            # deal with:
            # 1. /FOO (only run name)
            # 2. FOO/BAR (both exp and run name)
            #
            # FOO/ (only exp name) will not enter this branch
            # and end up getting an empty run_name (as declared above)
            run_name = name[delim_idx + 1 :]

        if delim_idx > 0:
            experiment_name = name[:delim_idx]

        if not run_name:
            if m:  # use the last module name
                run_name = m.rpartition(".")[2]
            else:  # use script name w/ no extension
                assert script, "`script` can't be `None` here due checks above"
                run_name = Path(script).stem
        return StructuredNameArgument(
            experiment_name or default_experiment_name, run_name
        )


@dataclass
class StructuredJArgument:
    nnodes: int
    nproc_per_node: int

    def __str__(self) -> str:
        return f"{self.nnodes}x{self.nproc_per_node}"

    @staticmethod
    def parse_from(h: str, j: str) -> "StructuredJArgument":
        """
        Creates an :py:class:`ArgJ` instance given the ``h`` (host) and ``j`` (nnodes x nproc_per_node)
        component arguments.

        If the host has GPUs and ``j`` only specified nnodes (e.g. ``-j 2`` versus ``-j 2x8``), then
        nproc_per_node is set equal to the number of GPUs on the host. If nproc_per_node was explicitly
        specified, then it is honored even if it does not match the number of GPUs on the host.
        However, a warning message is displayed reminding the user that there is a mismatch between
        the GPU count on the host and the configured nproc_per_node.

        Example (GPU):

            .. doctest::

                >>> from torchx.components.structured_arg import StructuredJArgument
                >>> str(StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2"))
                '2x8'

                >>> str(StructuredJArgument.parse_from(h="aws_p4d.24xlarge", j="2x4"))
                '2x4'


        For hosts with no GPU devices, one MUST specify nproc_per_node. Otherwise this function will
        raise an Error.

        Example (CPU or Trainium):

            .. doctest::

                >>> str(StructuredJArgument.parse_from(h="aws_trn1.32xlarge", j="2"))
                Traceback (most recent call last):
                    ...
                ValueError: nproc_per_node cannot be inferred from GPU count. `trn1.32xlarge` is not a GPU instance. ...

                >>> str(StructuredJArgument.parse_from(h="aws_trn1.32xlarge", j="2x16"))
                '2x16'

        """
        nums = j.split("x")
        num_gpus = specs.named_resources[h].gpu
        if len(nums) == 1:  # -j 1
            nnodes = int(nums[0])
            # infer nproc_per_node from # of gpus in host

            if num_gpus > 0:
                nproc_per_node = num_gpus
            else:
                raise ValueError(
                    f"nproc_per_node cannot be inferred from GPU count."
                    f" `{h}` is not a GPU instance."
                    f" You must specify `-j $NNODESx$NPROCS_PER_NODE` (e.g. `-j {nnodes}x8`)"
                )

        elif len(nums) == 2:  # -j 1x2
            nnodes = int(nums[0])
            nproc_per_node = int(nums[1])

            if nproc_per_node != num_gpus:
                warnings.warn(
                    f"In `-j {j}` you specified nproc_per_node={nproc_per_node}"
                    f" which does not equal the number of GPUs on a {h}: {num_gpus}."
                    f" This may lead to under-utilization or an error. "
                    f" If this was intentional, ignore this warning."
                    f" Otherwise set `-j {nnodes}` to auto-set nproc_per_node"
                    f" to the number of GPUs on the host."
                )
        else:
            raise ValueError(
                f"Invalid format for `-j $NNODESx$NPROCS_PER_NODE` (e.g. `-j 1x8`). Given: {j}"
            )

        return StructuredJArgument(nnodes=nnodes, nproc_per_node=nproc_per_node)
