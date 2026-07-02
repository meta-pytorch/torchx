# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

r"""
`torchx.specs.named_resources_aws` contains resource definitions that represent corresponding AWS instance types
taken from https://aws.amazon.com/ec2/instance-types/. The resources are exposed
via entrypoints after installing torchx lib. The mapping is stored in the `setup.py` file.

The named resources currently do not specify AWS instance type capabilities but merely represent
the equvalent resource in mem, cpu and gpu numbers.

.. note::
    These resource definitions may change in future. It is expected for each user to
    manage their own resources. See :ref:`advanced:Registering Named Resources`
    to set up named resources.

Usage:

    ::

     from torchx.specs import named_resources
     print(named_resources["aws_t3.medium"])
     print(named_resources["aws_m5.2xlarge"])
     print(named_resources["aws_p3.2xlarge"])
     print(named_resources["aws_p3.8xlarge"])
     print(named_resources["aws_p6-b200.48xlarge"])

"""

import warnings
from typing import Callable, Mapping

from torchx.specs.api import Resource

EFA_DEVICE = "vpc.amazonaws.com/efa"
NEURON_DEVICE = "aws.amazon.com/neurondevice"

# ecs and ec2 have memtax and currently AWS Batch uses hard memory limits
# so we have to account for mem tax when registering these resources for AWS
# otherwise the job will be stuck in the jobqueue forever
# 96% is based on empirical observation that works well for most instance types
# see: https://docs.aws.amazon.com/batch/latest/userguide/memory-management.html
MEM_TAX = 0.96

# determines instance type for non-honogeneous CEs
# see https://github.com/meta-pytorch/torchx/issues/780
K8S_ITYPE = "node.kubernetes.io/instance-type"
GiB: int = int(1024 * MEM_TAX)


def instance_type_from_resource(resource: Resource) -> str:
    instance_type = resource.capabilities.get(K8S_ITYPE)
    if instance_type is None:
        warnings.warn(
            "Cannot determine resource instance type which can cause issues for non-homogeneous CEs and multinode jobs. Consider providing torchx.specs.named_resources_aws:K8S_TYPE resource capability."
        )
    # pyrefly: ignore [bad-return]
    return instance_type


def aws_p3_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=1, memMB=61 * GiB, capabilities={K8S_ITYPE: "p3.2xlarge"}
    )


def aws_p3_8xlarge() -> Resource:
    return Resource(
        cpu=32, gpu=4, memMB=244 * GiB, capabilities={K8S_ITYPE: "p3.8xlarge"}
    )


def aws_p3_16xlarge() -> Resource:
    return Resource(
        cpu=64, gpu=8, memMB=488 * GiB, capabilities={K8S_ITYPE: "p3.16xlarge"}
    )


def aws_p3dn_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=8,
        memMB=768 * GiB,
        capabilities={K8S_ITYPE: "p3dn.24xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_p4d_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=8,
        memMB=1152 * GiB,
        capabilities={K8S_ITYPE: "p4d.24xlarge"},
        devices={EFA_DEVICE: 4},
    )


def aws_p4de_24xlarge() -> Resource:
    # p4de has same cpu, gpu, memMB as p4d but gpu memory is 2x (32GB vs 64GB per GPU)
    return Resource(
        cpu=96,
        gpu=8,
        memMB=1152 * GiB,
        capabilities={K8S_ITYPE: "p4de.24xlarge"},
        devices={EFA_DEVICE: 4},
    )


def aws_p5_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=8,
        memMB=2048 * GiB,
        capabilities={K8S_ITYPE: "p5.48xlarge"},
        devices={EFA_DEVICE: 32},
    )


def aws_p5e_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=8,
        memMB=2048 * GiB,
        capabilities={K8S_ITYPE: "p5e.48xlarge"},
        devices={EFA_DEVICE: 32},
    )


def aws_p5en_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=8,
        memMB=2048 * GiB,
        capabilities={K8S_ITYPE: "p5en.48xlarge"},
        devices={EFA_DEVICE: 16},
    )


def aws_p6_b200_48xlarge() -> Resource:
    # 8x NVIDIA Blackwell B200 GPUs (1,432 GB HBM3e total),
    # 192 vCPUs (5th Gen Intel Xeon Scalable - Emerald Rapids), 2,048 GiB system memory,
    # 3.2 Tbps EFAv4 networking via 8 network cards (400 Gbps each).
    # See: https://aws.amazon.com/ec2/instance-types/p6/
    return Resource(
        cpu=192,
        gpu=8,
        memMB=2048 * GiB,
        capabilities={K8S_ITYPE: "p6-b200.48xlarge"},
        devices={EFA_DEVICE: 8},
    )


def aws_p6_b300_48xlarge() -> Resource:
    # 8x NVIDIA Blackwell Ultra B300 GPUs (2,144 GB HBM3e total),
    # 192 vCPUs, 4,096 GiB system memory,
    # 6.4 Tbps EFAv4 networking via 17 network cards (NCI 0 is ENA-only,
    # NCIs 1-16 are EFA-capable at 400 Gbps each).
    # See: https://aws.amazon.com/ec2/instance-types/p6/
    return Resource(
        cpu=192,
        gpu=8,
        memMB=4096 * GiB,
        capabilities={K8S_ITYPE: "p6-b300.48xlarge"},
        devices={EFA_DEVICE: 16},
    )


def aws_p6e_gb200_36xlarge() -> Resource:
    # 4x NVIDIA Blackwell GB200 GPUs (740 GB HBM3e total),
    # 144 vCPUs (NVIDIA Grace CPU), 960 GiB system memory.
    # Up to 17 network cards (NCI 0 is ENA-only, NCIs 1-16 are EFA-capable).
    # Available only as part of P6e-GB200 UltraServers.
    # See: https://aws.amazon.com/ec2/instance-types/p6/
    return Resource(
        cpu=144,
        gpu=4,
        memMB=960 * GiB,
        capabilities={K8S_ITYPE: "p6e-gb200.36xlarge"},
        devices={EFA_DEVICE: 16},
    )


def aws_t3_medium() -> Resource:
    return Resource(cpu=2, gpu=0, memMB=4 * GiB, capabilities={K8S_ITYPE: "t3.medium"})


def aws_m5_large() -> Resource:
    return Resource(cpu=2, gpu=0, memMB=8 * GiB, capabilities={K8S_ITYPE: "m5.large"})


def aws_m5_xlarge() -> Resource:
    return Resource(cpu=4, gpu=0, memMB=16 * GiB, capabilities={K8S_ITYPE: "m5.xlarge"})


def aws_m5_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=0, memMB=32 * GiB, capabilities={K8S_ITYPE: "m5.2xlarge"}
    )


def aws_m5_4xlarge() -> Resource:
    return Resource(
        cpu=16, gpu=0, memMB=64 * GiB, capabilities={K8S_ITYPE: "m5.4xlarge"}
    )


def aws_m5_8xlarge() -> Resource:
    return Resource(
        cpu=32, gpu=0, memMB=128 * GiB, capabilities={K8S_ITYPE: "m5.8xlarge"}
    )


def aws_m5_12xlarge() -> Resource:
    return Resource(
        cpu=48, gpu=0, memMB=192 * GiB, capabilities={K8S_ITYPE: "m5.12xlarge"}
    )


def aws_m5_16xlarge() -> Resource:
    return Resource(
        cpu=64, gpu=0, memMB=256 * GiB, capabilities={K8S_ITYPE: "m5.16xlarge"}
    )


def aws_m5_24xlarge() -> Resource:
    return Resource(
        cpu=96, gpu=0, memMB=384 * GiB, capabilities={K8S_ITYPE: "m5.24xlarge"}
    )


def aws_m5_metal() -> Resource:
    # m5.metal: 96 logical processors on 48 physical cores, 384 GiB memory
    return Resource(
        cpu=96, gpu=0, memMB=384 * GiB, capabilities={K8S_ITYPE: "m5.metal"}
    )


# m5d.* matches m5.* in vCPU/memory but adds local NVMe SSD storage.
# See: https://aws.amazon.com/ec2/instance-types/m5/


def aws_m5d_large() -> Resource:
    return Resource(cpu=2, gpu=0, memMB=8 * GiB, capabilities={K8S_ITYPE: "m5d.large"})


def aws_m5d_xlarge() -> Resource:
    return Resource(
        cpu=4, gpu=0, memMB=16 * GiB, capabilities={K8S_ITYPE: "m5d.xlarge"}
    )


def aws_m5d_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=0, memMB=32 * GiB, capabilities={K8S_ITYPE: "m5d.2xlarge"}
    )


def aws_m5d_4xlarge() -> Resource:
    return Resource(
        cpu=16, gpu=0, memMB=64 * GiB, capabilities={K8S_ITYPE: "m5d.4xlarge"}
    )


def aws_m5d_8xlarge() -> Resource:
    return Resource(
        cpu=32, gpu=0, memMB=128 * GiB, capabilities={K8S_ITYPE: "m5d.8xlarge"}
    )


def aws_m5d_12xlarge() -> Resource:
    return Resource(
        cpu=48, gpu=0, memMB=192 * GiB, capabilities={K8S_ITYPE: "m5d.12xlarge"}
    )


def aws_m5d_16xlarge() -> Resource:
    return Resource(
        cpu=64, gpu=0, memMB=256 * GiB, capabilities={K8S_ITYPE: "m5d.16xlarge"}
    )


def aws_m5d_24xlarge() -> Resource:
    return Resource(
        cpu=96, gpu=0, memMB=384 * GiB, capabilities={K8S_ITYPE: "m5d.24xlarge"}
    )


def aws_m5d_metal() -> Resource:
    # m5d.metal: 96 logical processors on 48 physical cores, 384 GiB memory,
    # plus 4 x 900 NVMe SSD local storage.
    return Resource(
        cpu=96, gpu=0, memMB=384 * GiB, capabilities={K8S_ITYPE: "m5d.metal"}
    )


def aws_c5_18xlarge() -> Resource:
    return Resource(
        # using lower memory size than the spec since MEM_TAX is not enough for adjustment
        cpu=72,
        gpu=0,
        memMB=142 * GiB,
        capabilities={K8S_ITYPE: "c5.18xlarge"},
    )


def aws_g4dn_xlarge() -> Resource:
    return Resource(
        cpu=4, gpu=1, memMB=16 * GiB, capabilities={K8S_ITYPE: "g4dn.xlarge"}
    )


def aws_g4dn_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=1, memMB=32 * GiB, capabilities={K8S_ITYPE: "g4dn.2xlarge"}
    )


def aws_g4dn_4xlarge() -> Resource:
    return Resource(
        cpu=16, gpu=1, memMB=64 * GiB, capabilities={K8S_ITYPE: "g4dn.4xlarge"}
    )


def aws_g4dn_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=1,
        memMB=128 * GiB,
        capabilities={K8S_ITYPE: "g4dn.8xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g4dn_12xlarge() -> Resource:
    return Resource(
        cpu=48,
        gpu=4,
        memMB=192 * GiB,
        capabilities={K8S_ITYPE: "g4dn.12xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g4dn_16xlarge() -> Resource:
    return Resource(
        cpu=64,
        gpu=1,
        memMB=256 * GiB,
        capabilities={K8S_ITYPE: "g4dn.16xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g4dn_metal() -> Resource:
    return Resource(
        cpu=96,
        gpu=8,
        memMB=384 * GiB,
        capabilities={K8S_ITYPE: "g4dn.metal"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_xlarge() -> Resource:
    return Resource(cpu=4, gpu=1, memMB=16 * GiB, capabilities={K8S_ITYPE: "g5.xlarge"})


def aws_g5_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=1, memMB=32 * GiB, capabilities={K8S_ITYPE: "g5.2xlarge"}
    )


def aws_g5_4xlarge() -> Resource:
    return Resource(
        cpu=16, gpu=1, memMB=64 * GiB, capabilities={K8S_ITYPE: "g5.4xlarge"}
    )


def aws_g5_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=1,
        memMB=128 * GiB,
        capabilities={K8S_ITYPE: "g5.8xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_12xlarge() -> Resource:
    return Resource(
        cpu=48,
        gpu=4,
        memMB=192 * GiB,
        capabilities={K8S_ITYPE: "g5.12xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_16xlarge() -> Resource:
    return Resource(
        cpu=64,
        gpu=1,
        memMB=256 * GiB,
        capabilities={K8S_ITYPE: "g5.16xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=4,
        memMB=384 * GiB,
        capabilities={K8S_ITYPE: "g5.24xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=8,
        memMB=768 * GiB,
        capabilities={K8S_ITYPE: "g5.48xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g6e_xlarge() -> Resource:
    return Resource(
        cpu=4,
        gpu=1,
        memMB=32 * GiB,
        capabilities={K8S_ITYPE: "g6e.xlarge"},
    )


def aws_g6e_2xlarge() -> Resource:
    return Resource(
        cpu=8,
        gpu=1,
        memMB=64 * GiB,
        capabilities={K8S_ITYPE: "g6e.2xlarge"},
    )


def aws_g6e_4xlarge() -> Resource:
    return Resource(
        cpu=16,
        gpu=1,
        memMB=128 * GiB,
        capabilities={K8S_ITYPE: "g6e.4xlarge"},
    )


def aws_g6e_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=1,
        memMB=256 * GiB,
        capabilities={K8S_ITYPE: "g6e.8xlarge"},
    )


def aws_g6e_16xlarge() -> Resource:
    return Resource(
        cpu=64,
        gpu=1,
        memMB=512 * GiB,
        capabilities={K8S_ITYPE: "g6e.16xlarge"},
    )


def aws_g6e_12xlarge() -> Resource:
    return Resource(
        cpu=48,
        gpu=4,
        memMB=384 * GiB,
        capabilities={K8S_ITYPE: "g6e.12xlarge"},
    )


def aws_g6e_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=4,
        memMB=768 * GiB,
        capabilities={K8S_ITYPE: "g6e.24xlarge"},
        devices={EFA_DEVICE: 2},
    )


def aws_g6e_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=8,
        memMB=1536 * GiB,
        capabilities={K8S_ITYPE: "g6e.48xlarge"},
        devices={EFA_DEVICE: 4},
    )


def aws_trn1_2xlarge() -> Resource:
    return Resource(
        cpu=8,
        gpu=0,
        memMB=32 * GiB,
        capabilities={K8S_ITYPE: "trn1.2xlarge"},
        devices={NEURON_DEVICE: 1},
    )


def aws_trn1_32xlarge() -> Resource:
    return Resource(
        cpu=128,
        gpu=0,
        memMB=512 * GiB,
        capabilities={K8S_ITYPE: "trn1.32xlarge"},
        devices={EFA_DEVICE: 8, NEURON_DEVICE: 16},
    )


def aws_inf2_xlarge() -> Resource:
    return Resource(
        cpu=4,
        gpu=0,
        memMB=16 * GiB,
        capabilities={K8S_ITYPE: "inf2.xlarge"},
        devices={NEURON_DEVICE: 1},
    )


def aws_inf2_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=0,
        memMB=128 * GiB,
        capabilities={K8S_ITYPE: "inf2.8xlarge"},
        devices={NEURON_DEVICE: 1},
    )


def aws_inf2_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=0,
        memMB=384 * GiB,
        capabilities={K8S_ITYPE: "inf2.24xlarge"},
        devices={NEURON_DEVICE: 6},
    )


def aws_inf2_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=0,
        memMB=768 * GiB,
        capabilities={K8S_ITYPE: "inf2.48xlarge"},
        devices={NEURON_DEVICE: 12},
    )


NAMED_RESOURCES: Mapping[str, Callable[[], Resource]] = {
    "aws_t3.medium": aws_t3_medium,
    "aws_m5.large": aws_m5_large,
    "aws_m5.xlarge": aws_m5_xlarge,
    "aws_m5.2xlarge": aws_m5_2xlarge,
    "aws_m5.4xlarge": aws_m5_4xlarge,
    "aws_m5.8xlarge": aws_m5_8xlarge,
    "aws_m5.12xlarge": aws_m5_12xlarge,
    "aws_m5.16xlarge": aws_m5_16xlarge,
    "aws_m5.24xlarge": aws_m5_24xlarge,
    "aws_m5.metal": aws_m5_metal,
    "aws_m5d.large": aws_m5d_large,
    "aws_m5d.xlarge": aws_m5d_xlarge,
    "aws_m5d.2xlarge": aws_m5d_2xlarge,
    "aws_m5d.4xlarge": aws_m5d_4xlarge,
    "aws_m5d.8xlarge": aws_m5d_8xlarge,
    "aws_m5d.12xlarge": aws_m5d_12xlarge,
    "aws_m5d.16xlarge": aws_m5d_16xlarge,
    "aws_m5d.24xlarge": aws_m5d_24xlarge,
    "aws_m5d.metal": aws_m5d_metal,
    "aws_c5.18xlarge": aws_c5_18xlarge,
    "aws_p3.2xlarge": aws_p3_2xlarge,
    "aws_p3.8xlarge": aws_p3_8xlarge,
    "aws_p3.16xlarge": aws_p3_16xlarge,
    "aws_p3dn.24xlarge": aws_p3dn_24xlarge,
    "aws_p4d.24xlarge": aws_p4d_24xlarge,
    "aws_p4de.24xlarge": aws_p4de_24xlarge,
    "aws_p5.48xlarge": aws_p5_48xlarge,
    "aws_p5e.48xlarge": aws_p5e_48xlarge,
    "aws_p5en.48xlarge": aws_p5en_48xlarge,
    "aws_p6-b200.48xlarge": aws_p6_b200_48xlarge,
    "aws_p6-b300.48xlarge": aws_p6_b300_48xlarge,
    "aws_p6e-gb200.36xlarge": aws_p6e_gb200_36xlarge,
    "aws_g4dn.xlarge": aws_g4dn_xlarge,
    "aws_g4dn.2xlarge": aws_g4dn_2xlarge,
    "aws_g4dn.4xlarge": aws_g4dn_4xlarge,
    "aws_g4dn.8xlarge": aws_g4dn_8xlarge,
    "aws_g4dn.16xlarge": aws_g4dn_16xlarge,
    "aws_g4dn.12xlarge": aws_g4dn_12xlarge,
    "aws_g4dn.metal": aws_g4dn_metal,
    "aws_g5.xlarge": aws_g5_xlarge,
    "aws_g5.2xlarge": aws_g5_2xlarge,
    "aws_g5.4xlarge": aws_g5_4xlarge,
    "aws_g5.8xlarge": aws_g5_8xlarge,
    "aws_g5.16xlarge": aws_g5_16xlarge,
    "aws_g5.12xlarge": aws_g5_12xlarge,
    "aws_g5.24xlarge": aws_g5_24xlarge,
    "aws_g5.48xlarge": aws_g5_48xlarge,
    "aws_g6e.xlarge": aws_g6e_xlarge,
    "aws_g6e.2xlarge": aws_g6e_2xlarge,
    "aws_g6e.4xlarge": aws_g6e_4xlarge,
    "aws_g6e.8xlarge": aws_g6e_8xlarge,
    "aws_g6e.16xlarge": aws_g6e_16xlarge,
    "aws_g6e.12xlarge": aws_g6e_12xlarge,
    "aws_g6e.24xlarge": aws_g6e_24xlarge,
    "aws_g6e.48xlarge": aws_g6e_48xlarge,
    "aws_trn1.2xlarge": aws_trn1_2xlarge,
    "aws_trn1.32xlarge": aws_trn1_32xlarge,
    "aws_inf2.xlarge": aws_inf2_xlarge,
    "aws_inf2.8xlarge": aws_inf2_8xlarge,
    "aws_inf2.24xlarge": aws_inf2_24xlarge,
    "aws_inf2.48xlarge": aws_inf2_48xlarge,
}
