import typing

import torch
import logging

from ._tensor import DTensor, implements

logger = logging.getLogger(__name__)


@implements(torch.stack)
def stack(
    tensors: typing.Union[typing.Tuple[DTensor, ...], typing.List[DTensor]],
    dim: int = 0,
    *,
    out: DTensor = None,
):
    logger.warning("stack DTensors may be slow")
    raise ValueError("stack DTensors may be slow")
    if out is not None:
        raise NotImplementedError("stack does not support out")

    out = tensors[0]
    for tensor in tensors[1:]:
        # TODO: check shapes
        out = DTensor(
            out.shardings.join_shard(
                tensor.shardings,
                func=lambda x, y: torch.stack([x, y], dim=dim),
            )
        )
    return out
