import logging
from typing import Optional

import numpy as np
import torch as t
from fate.arch import Context
from fate.arch.protocol.secure_aggregation._secure_aggregation import (
    SecureAggregatorClient as sa_client,
)
from fate.arch.protocol.secure_aggregation._secure_aggregation import (
    SecureAggregatorServer as sa_server,
)

logger = logging.getLogger(__name__)


AGGREGATE_TYPE = ["mean", "sum", "weighted_mean"]
TORCH_TENSOR_PRECISION = ["float32", "float64"]


class AutoSuffix(object):

    """
    A auto suffix that will auto increase count
    """

    def __init__(self, suffix_str=""):
        self._count = 0
        self.suffix_str = suffix_str

    def __call__(self):
        concat_suffix = self.suffix_str + "_" + str(self._count)
        self._count += 1
        return concat_suffix


class Aggregator:
    def __init__(self, ctx: Context, aggregator_name: Optional[str] = None):

        if aggregator_name is not None:
            agg_name = "_" + aggregator_name
        else:
            agg_name = ""
        self.suffix = {
            "local_loss": AutoSuffix("local_loss" + agg_name),
            "agg_loss": AutoSuffix("agg_loss" + agg_name),
            "local_model": AutoSuffix("local_model" + agg_name),
            "agg_model": AutoSuffix("agg_model" + agg_name),
            "converge_status": AutoSuffix("converge_status" + agg_name),
            "local_weight": AutoSuffix("local_weight" + agg_name),
            "computed_weight": AutoSuffix("agg_weight" + agg_name),
        }

    def model_aggregation(self, *args, **kwargs):
        raise NotImplementedError("model_aggregation should be implemented in subclass")

    def loss_aggregation(self, *args, **kwargs):
        raise NotImplementedError("loss_aggregation should be implemented in subclass")


class BaseAggregatorClient(Aggregator):
    def __init__(
        self,
        ctx: Context,
        aggregator_name: str = None,
        aggregate_type="mean",
        sample_num=1,
        is_mock=True,
        require_grad=True,
        float_p="float64",
    ) -> None:

        super().__init__(ctx, aggregator_name)
        self._weight = 1.0
        self.aggregator_name = "default" if aggregator_name is None else aggregator_name
        self.require_grad = require_grad

        assert float_p in TORCH_TENSOR_PRECISION, "float_p should be one of {}".format(TORCH_TENSOR_PRECISION)
        self.float_p = float_p

        if sample_num <= 0 and not isinstance(sample_num, int):
            raise ValueError("sample_num should be int greater than 0")

        logger.info("computing weights")
        if aggregate_type not in AGGREGATE_TYPE:
            raise ValueError("aggregate_type should be one of {}".format(AGGREGATE_TYPE))
        elif aggregate_type == "mean":
            ctx.arbiter.put(self.suffix["local_weight"](), 1.0)
            self._weight = ctx.arbiter.get(self.suffix["computed_weight"]())
        elif aggregate_type == "sum":
            ctx.arbiter.put(self.suffix["local_weight"](), sample_num)
            self._weight = 1.0
        elif aggregate_type == "weighted_mean":
            if sample_num <= 0 or sample_num is None:
                raise ValueError("sample_num should be int greater than 0")
            ctx.arbiter.put(self.suffix["local_weight"](), sample_num)
            self._weight = ctx.arbiter.get(self.suffix["computed_weight"]())

        logger.info("aggregate weight is {}".format(self._weight))

        self.model_aggregator = sa_client(prefix=self.aggregator_name + "_model", is_mock=is_mock)
        self.model_aggregator.dh_exchange(ctx, [ctx.guest.rank, *ctx.hosts.ranks])
        self.loss_aggregator = sa_client(prefix=self.aggregator_name + "_loss", is_mock=is_mock)
        self.loss_aggregator.dh_exchange(ctx, [ctx.guest.rank, *ctx.hosts.ranks])

    def _convert_type(self, data, dtype="float32"):

        if isinstance(data, t.Tensor):
            if dtype == "float32":
                data = data.float()
            elif dtype == "float64":
                data = data.double()
            else:
                raise ValueError("Invalid dtype. Choose either 'float32' or 'float64'")

            numpy_array = data.detach().cpu().numpy()

        elif isinstance(data, np.ndarray):
            if dtype == "float32":
                numpy_array = data.astype(np.float32)
            elif dtype == "float64":
                numpy_array = data.astype(np.float64)
            else:
                raise ValueError("Invalid dtype. Choose either 'float32' or 'float64'")
        else:
            raise ValueError("Invalid data type. Only numpy ndarray and PyTorch tensor are supported.")

        return numpy_array

    def _process_model(self, model):

        to_agg = None
        if isinstance(model, np.ndarray) or isinstance(model, t.Tensor):
            to_agg = self._convert_type(model, self.float_p)
            return [to_agg]

        if isinstance(model, t.nn.Module):
            parameters = list(model.parameters())
            if self.require_grad:
                agg_list = [
                    self._convert_type(p.cpu().detach().numpy(), self.float_p) for p in parameters if p.requires_grad
                ]
            else:
                agg_list = [self._convert_type(p.cpu().detach().numpy(), self.float_p) for p in parameters]

        elif isinstance(model, list):
            to_agg = []
            for p in model:
                to_agg.append(self._convert_type(p, self.float_p))
            agg_list = to_agg

        return agg_list

    def _recover_model(self, model, agg_model):

        if isinstance(model, np.ndarray) or isinstance(model, t.Tensor):
            return agg_model
        elif isinstance(model, t.nn.Module):
            if self.require_grad:
                for agg_p, p in zip(agg_model, [p for p in model.parameters() if p.requires_grad]):
                    p.data.copy_(t.Tensor(agg_p))
            else:
                for agg_p, p in zip(agg_model, model.parameters()):
                    p.data.copy_(t.Tensor(agg_p))
            return model
        else:
            return agg_model

    """
    User API
    """

    def model_aggregation(self, ctx, model):

        to_send = self._process_model(model)
        agg_model = self.model_aggregator.secure_aggregate(ctx, to_send, self._weight)
        return self._recover_model(model, agg_model)

    def loss_aggregation(self, ctx, loss):
        if isinstance(loss, t.Tensor):
            loss = loss.detach.cpu().numpy()
        else:
            loss = np.array(loss)
        loss = [loss]
        agg_loss = self.loss_aggregator.secure_aggregate(ctx, loss, self._weight)
        return agg_loss


class BaseAggregatorServer(Aggregator):
    def __init__(self, ctx: Context, aggregator_name: str = None, is_mock=True) -> None:

        super().__init__(ctx, aggregator_name)

        weight_list = self._collect(ctx, self.suffix["local_weight"]())
        weight_sum = sum(weight_list)
        ret_weight = []
        for w in weight_list:
            ret_weight.append(w / weight_sum)

        ret_suffix = self.suffix["computed_weight"]()
        for idx, w in enumerate(ret_weight):
            self._broadcast(ctx, w, ret_suffix, idx)

        self.aggregator_name = "default" if aggregator_name is None else aggregator_name
        self.model_aggregator = sa_server(
            prefix=self.aggregator_name + "_model", is_mock=is_mock, ranks=[ctx.guest.rank, *ctx.hosts.ranks]
        )
        self.loss_aggregator = sa_server(
            prefix=self.aggregator_name + "_loss", is_mock=is_mock, ranks=[ctx.guest.rank, *ctx.hosts.ranks]
        )

    def _check_party_id(self, party_id):
        # party idx >= -1, int
        if not isinstance(party_id, int):
            raise ValueError("party_id should be int")
        if party_id < -1:
            raise ValueError("party_id should be greater than -1")

    def _collect(self, ctx, suffix):
        guest_item = [ctx.guest.get(suffix)]
        host_item = ctx.hosts.get(suffix)
        combine_list = guest_item + host_item
        return combine_list

    def _broadcast(self, ctx, data, suffix, party_idx=-1):
        self._check_party_id(party_idx)
        if party_idx == -1:
            ctx.guest.put(suffix, data)
            ctx.hosts.put(suffix, data)
        elif party_idx == 0:
            ctx.guest.put(suffix, data)
        else:
            ctx.hosts[party_idx - 1].put(suffix, data)

    """
    User API
    """

    def model_aggregation(self, ctx, ranks=None):
        self.model_aggregator.secure_aggregate(ctx, ranks=ranks)

    def loss_aggregation(self, ctx, ranks=None):
        self.loss_aggregator.secure_aggregate(ctx, ranks=ranks)
