import torch as t
import numpy as np
from fate.arch import Context
from typing import Union
from .base import Aggregator
import logging
from fate.arch.protocol._dh import SecureAggregatorClient as sa_client
from fate.arch.protocol._dh import SecureAggregatorServer as sa_server


logger = logging.getLogger(__name__)


AGGREGATE_TYPE = ['mean', 'sum', 'weighted_mean']


class PlainTextAggregatorClient(Aggregator):

    """
    PlainTextAggregatorClient is used to aggregate plain text data
    """

    def __init__(self, ctx: Context, aggregator_name: str = None, aggregate_type='mean', sample_num=1) -> None:
        
        super().__init__(ctx, aggregator_name)
        self.ctx = ctx
        self._weight = 1.0
        self.aggregator_name = 'default' if aggregator_name is None else aggregator_name
        
        if sample_num <= 0 and not isinstance(sample_num, int):
            raise ValueError("sample_num should be int greater than 0")

        logger.info('computing weights')
        if aggregate_type not in AGGREGATE_TYPE:
            raise ValueError("aggregate_type should be one of {}".format(AGGREGATE_TYPE))
        elif aggregate_type == 'mean':
            self.ctx.arbiter.put(self.suffix["local_weight"](), 1.0)
            self._weight = self.ctx.arbiter.get(self.suffix["computed_weight"]())
        elif aggregate_type == 'sum':
            self.ctx.arbiter.put(self.suffix["local_weight"](), sample_num)
            self._weight = 1.0
        elif aggregate_type == 'weighted_mean':
            if sample_num <= 0 or sample_num is None:
                raise ValueError("sample_num should be int greater than 0")
            self.ctx.arbiter.put(self.suffix["local_weight"](), sample_num)
            self._weight = self.ctx.arbiter.get(self.suffix["computed_weight"]())

        logger.info("aggregate weight is {}".format(self._weight))

        self.model_aggregator = sa_client(prefix=self.aggregator_name+'_model', is_mock=True)
        self.model_aggregator.dh_exchange(ctx, [ctx.guest.rank, *ctx.hosts.ranks])
        self.loss_aggregator = sa_client(prefix=self.aggregator_name+'_loss', is_mock=True)
        self.loss_aggregator.dh_exchange(ctx, [ctx.guest.rank, *ctx.hosts.ranks])

    def _process_model(self, model):

        to_agg = None
        if isinstance(model, np.ndarray) or isinstance(model, t.Tensor):
            to_agg = model * self._weight
            return [to_agg]

        if isinstance(model, t.nn.Module):
            parameters = list(model.parameters())
            agg_list = [p.cpu().detach().numpy() for p in parameters if p.requires_grad]

        elif isinstance(model, list):
            for p in model:
                assert isinstance(
                    p, np.ndarray), 'expecting List[np.ndarray], but got {}'.format(p)
            agg_list = model

        return agg_list

    def _recover_model(self, model, agg_model):
        
        if isinstance(model, np.ndarray) or isinstance(model, t.Tensor):
            return agg_model
        elif isinstance(model, t.nn.Module):
            for agg_p, p in zip(agg_model, [p for p in model.parameters() if p.requires_grad]):
                p.data.copy_(t.Tensor(agg_p))
            return model
        else:
            return agg_model

    """
    User API
    """

    def model_aggregation(self, model):

        to_send = self._process_model(model)
        print('model is ', to_send)
        agg_model = self.model_aggregator.secure_aggregate(self.ctx, to_send, self._weight)
        return self._recover_model(model, agg_model)

    def loss_aggregation(self, loss):
        if isinstance(loss, t.Tensor):
            loss = loss.detach.cpu().numpy()
        else:
            loss = np.array(loss)
        loss = [loss]
        agg_loss = self.loss_aggregator.secure_aggregate(self.ctx, loss, self._weight)
        return agg_loss


class PlainTextAggregatorServer(Aggregator):

    """
    PlainTextAggregatorServer is used to aggregate plain text data
    """

    def __init__(self, ctx: Context, aggregator_name: str = None) -> None:

        super().__init__(ctx, aggregator_name)

        weight_list = self._collect(self.suffix["local_weight"]())
        weight_sum = sum(weight_list)
        ret_weight = []
        for w in weight_list:
            ret_weight.append(w / weight_sum)
        
        ret_suffix = self.suffix["computed_weight"]()
        for idx, w in enumerate(ret_weight):
            self._broadcast(w, ret_suffix, idx)

        self.aggregator_name = 'default' if aggregator_name is None else aggregator_name
        self.model_aggregator = sa_server(prefix=self.aggregator_name+'_model', is_mock=True, ranks=[ctx.guest.rank, *ctx.hosts.ranks])
        self.loss_aggregator = sa_server(prefix=self.aggregator_name+'_loss', is_mock=True, ranks=[ctx.guest.rank, *ctx.hosts.ranks])

    def _check_party_id(self, party_id):
        # party idx >= -1, int
        if not isinstance(party_id, int):
            raise ValueError("party_id should be int")
        if party_id < -1:
            raise ValueError("party_id should be greater than -1")
        
    def _collect(self, suffix):
        guest_item = [self.ctx.guest.get(suffix)]
        host_item = self.ctx.hosts.get(suffix)
        combine_list = guest_item + host_item
        return combine_list
        
    def _broadcast(self, data, suffix, party_idx=-1):
        self._check_party_id(party_idx)
        if party_idx == -1:
            self.ctx.guest.put(suffix, data)
            self.ctx.hosts.put(suffix, data)
        elif party_idx == 0:
            self.ctx.guest.put(suffix, data)
        else:
            self.ctx.hosts[party_idx - 1].put(suffix, data)

    """
    User API
    """

    def model_aggregation(self, ranks=None):
        self.model_aggregator.secure_aggregate(self.ctx, ranks=ranks)

    def loss_aggregation(self, ranks=None):
        self.loss_aggregator.secure_aggregate(self.ctx, ranks=ranks)
        