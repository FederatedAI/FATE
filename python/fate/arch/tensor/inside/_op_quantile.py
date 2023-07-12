from typing import List, Union

import numpy
import torch
from fate_utils import quantile


def quantile_fi(input: torch.Tensor, q, epsilon):
    if input.dtype == torch.float64:
        if len(input.shape) == 1:
            return quantile.quantile_f64_ix1(input.numpy(), q, epsilon)
        elif len(input.shape) == 2:
            return quantile.quantile_f64_ix2(input.numpy(), q, epsilon)
    raise NotImplementedError()


class GKSummary:
    """
    GKSummary is a summary of a stream of numbers, which can be used to estimate quantiles.

    Examples:
        >>> summary = GKSummary(0.01)
        >>> summary += torch.tensor([1.0, 2.0, 3.0])
        >>> summary += torch.tensor([4.0, 5.0, 6.0])
        >>> summary += torch.tensor([7.0, 8.0, 9.0])
        >>> summary.queries([0.1, 0.2])
        [2.0, 3.0]
    """

    def __init__(self, epsilon: float) -> None:
        self._summary = quantile.QuantileSummaryStream(epsilon)

    def merge(self, other: "GKSummary"):
        """merge other summary into self."""
        self._summary.merge(other._summary)
        return self

    def push(self, array: Union[torch.Tensor, numpy.ndarray]):
        """push elements in array into summary."""
        if isinstance(array, torch.Tensor):
            array = array.numpy()
        self._summary.insert_array(array)
        return self

    def __add__(self, other: "GKSummary"):
        if isinstance(other, GKSummary):
            return self.merge(other)
        return NotImplemented

    def __iadd__(self, other: Union[torch.Tensor, numpy.ndarray]):
        if isinstance(other, torch.Tensor) or isinstance(other, numpy.ndarray):
            return self.push(other)
        return NotImplemented

    def queries(self, q: List[float]):
        """return quantile values of q."""
        return torch.tensor(self._summary.queries(q))
