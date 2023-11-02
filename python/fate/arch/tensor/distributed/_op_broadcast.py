import torch
import logging

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(torch.broadcast_tensors)
def broadcast_tensors(*input: DTensor):
    logger.warning("broadcast_tensors is not implemented")
    return input
