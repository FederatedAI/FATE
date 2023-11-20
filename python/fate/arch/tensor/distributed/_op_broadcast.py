import torch
import logging

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(torch.broadcast_tensors)
def broadcast_tensors(*input: DTensor):
    for t in input[1:]:
        if isinstance(t, DTensor):
            if t.shardings != input[0].shardings:
                raise RuntimeError("broadcast_tensors expects all inputs to be broadcastable to the first input")
        if torch.broadcast_shapes(input[0].shape, t.shape) != input[0].shape:
            raise RuntimeError("broadcast_tensors expects all inputs to be broadcastable to the first input")
    return input
