import torch

from ..tensor import FPTensor
from ..unify import Backend, Device


class TensorKit:
    def __init__(self, backend: Backend, device: Device) -> None:
        self.backend = backend
        self.device = device

    def random_tensor(self, shape, num_partition=1) -> FPTensor:
        if self.backend == Backend.LOCAL:
            return FPTensor(torch.rand(shape))
        else:
            from fate.arch.tensor.impl.tensor.distributed import FPTensorDistributed

            from ..session import computing_session

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
                self,
                FPTensorDistributed(
                    computing_session.parallelize(
                        parts, include_key=False, partition=num_partition
                    )
                ),
            )

    def create_tensor(self, tensor: torch.Tensor) -> "FPTensor":
        return FPTensor(tensor)
