import torch
import logging

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(torch.broadcast_tensors)
def broadcast_tensors(*input: DTensor):
    for t in input:
        if not isinstance(t, DTensor):
            raise TypeError("broadcast_tensors expects all inputs to be tensors")
    shapes = input[0].shardings.shapes
    for t in input[1:]:
        if t.shardings.shapes != shapes:
            raise RuntimeError("broadcast_tensors expects all inputs to be of the same shape")
    return input
