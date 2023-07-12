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
    def __init__(self, summary=None) -> None:
        if summary is None:
            summary = quantile.QuantileSummaryStream()
        self._summary = summary

    def __add__(self, other: "GKSummary"):
        if isinstance(other, GKSummary):
            return GKSummary(self._summary.merge(other._summary))
        return NotImplemented

    def __iadd__(self, other: torch.Tensor):
        if isinstance(other, torch.Tensor):
            self._summary.insert_array(other.numpy())
            return self
        return NotImplemented
