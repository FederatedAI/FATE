import torch

from ..tensor import Tensor as FPTensor
from ..unify import device


class TensorKit:
    def __init__(self, computing, device: device) -> None:
        self.computing = computing
        self.device = device

    def random_tensor(self, shape, num_partition=1) -> FPTensor:
        from fate.arch.tensor.impl.tensor.distributed import FPTensorDistributed

        parts = []
        first_dim_approx = shape[0] // num_partition
        last_part_first_dim = shape[0] - (num_partition - 1) * first_dim_approx
        assert first_dim_approx > 0
        for i in range(num_partition):
            if i == num_partition - 1:
                parts.append(
                    torch.rand(
                        (
                            last_part_first_dim,
                            *shape[1:],
                        )
                    )
                )
            else:
                parts.append(torch.rand((first_dim_approx, *shape[1:])))
        return FPTensor(
            FPTensorDistributed(
                self.computing.parallelize(
                    parts, include_key=False, partition=num_partition
                )
            ),
        )

    def create_tensor(self, tensor: torch.Tensor) -> "FPTensor":
        return FPTensor(tensor)
