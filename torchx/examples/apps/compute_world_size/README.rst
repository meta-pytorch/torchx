Compute World Size Example
############################

This is a minimal "hello world" style  example application that uses
PyTorch Distributed to compute the world size. It is a minimal example
in that it initializes the ``torch.distributed`` process group and
performs a single collective operation (all_reduce) which is enough to
validate the infrastructure and scheduler setup.

This example is compatible with the ``dist.torchrun``. To run from CLI:

.. code-block:: shell-session

 $ cd $torchx-git-repo-root/torchx/examples/apps
 $ torchx run dist.torchrun --script compute_world_size/main.py -j 1x2
