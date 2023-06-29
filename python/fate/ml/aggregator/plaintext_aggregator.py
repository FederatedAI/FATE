import torch as t
import numpy as np
from fate.arch import Context
from typing import Union
from .base import Aggregator
import logging


logger = logging.getLogger(__name__)


AGGREGATE_TYPE = ['mean', 'sum', 'weighted_mean']


class PlainTextAggregatorClient(Aggregator):

    """
    PlainTextAggregatorClient is used to aggregate plain text data
    """

    def __init__(self, ctx: Context, aggregator_name=None, aggregate_type='mean', sample_num=1) -> None:
        
        super().__init__(ctx, aggregator_name)
        self.ctx = ctx
        self._weight = 1.0

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

    def _process_model(self, model):

        to_agg = None
        if isinstance(model, np.ndarray) or isinstance(model, t.Tensor):
            to_agg = model * self._weight
            return to_agg

        if isinstance(model, t.nn.Module):
            parameters = list(model.parameters())
            tmp_list = [[p.cpu().detach().numpy() for p in parameters if p.requires_grad]]
        elif isinstance(model, list):
            for p in model:
                assert isinstance(
                    p, np.ndarray), 'expecting List[np.ndarray], but got {}'.format(p)
            tmp_list = [model]

        to_agg = [[arr * self._weight for arr in arr_list]
                    for arr_list in tmp_list]
        return to_agg

    def _recover_model(self, model, agg_model):
        
        if isinstance(model, np.ndarray) or isinstance(model, t.Tensor):
            return agg_model
        elif isinstance(model, t.nn.Module):
            for agg_p, p in zip(agg_model, [p for p in model.parameters() if p.requires_grad]):
                p.data.copy_(t.Tensor(agg_p))
            return model
        else:
            return agg_model

    def _send_loss(self, loss):
        assert isinstance(loss, float) or isinstance(
            loss, np.ndarray), 'illegal loss type {}, loss should be a float or a np array'.format(type(loss))
        loss_suffix = self.suffix['local_loss']()
        self.ctx.arbiter.put(loss_suffix, loss)

    def _send_model(self, model: Union[np.ndarray, t.Tensor, t.nn.Module]):
        """Sending model to arbiter for aggregation

        Parameters
        ----------
        model : model can be:
                A numpy array
                A Weight instance(or subclass of Weights), see federatedml.framework.weights
                List of numpy array
                A pytorch model, is the subclass of torch.nn.Module
                A pytorch optimizer, will extract param group from this optimizer as weights to aggregate
        """
        # judge model type
        to_agg_model = self._process_model(model)
        suffix = self.suffix['local_model']()
        self.ctx.arbiter.put(suffix, to_agg_model)

    def _get_aggregated_model(self):
        return self.ctx.arbiter.get(self.suffix['agg_model']())[0]
    
    def _get_aggregated_loss(self):
        return self.ctx.arbiter.get(self.suffix['agg_loss']())[0]
    
    """
    User API
    """

    def model_aggregation(self, model):

        self._send_model(model)
        agg_model = self._get_aggregated_model()
        return self._recover_model(model, agg_model)

    def loss_aggregation(self, loss):
        self._send_loss(loss)



class PlainTextAggregatorServer(Aggregator):

    """
    PlainTextAggregatorServer is used to aggregate plain text data
    """

    def __init__(self, ctx: Context, aggregator_name=None) -> None:

        super().__init__(ctx, aggregator_name)
        weight_list = self._collect(self.suffix["local_weight"]())
        weight_sum = sum(weight_list)
        ret_weight = []
        for w in weight_list:
            ret_weight.append(w / weight_sum)
        
        ret_suffix = self.suffix["computed_weight"]()
        for idx, w in enumerate(ret_weight):
            self._broadcast(w, ret_suffix, idx)

    def _check_party_id(self, party_id):
        # party idx >= -1, int
        if not isinstance(party_id, int):
            raise ValueError("party_id should be int")
        if party_id < -1:
            raise ValueError("party_id should be greater than -1")
        
    def _collect(self, suffix, party_idx=-1):
        self._check_party_id(party_idx)
        guest_item = [self.ctx.guest.get(suffix)]
        host_item = self.ctx.hosts.get(suffix)
        combine_list = guest_item + host_item
        if party_idx == -1:
            return combine_list
        else:
            return combine_list[party_idx]
        
    def _broadcast(self, data, suffix, party_idx=-1):
        self._check_party_id(party_idx)
        if party_idx == -1:
            self.ctx.guest.put(suffix, data)
            self.ctx.hosts.put(suffix, data)
        elif party_idx == 0:
            self.ctx.guest.put(suffix, data)
        else:
            self.ctx.hosts[party_idx - 1].put(suffix, data)

    def _aggregate_model(self, party_idx=-1):

        # get suffix
        suffix = self.suffix['local_model']()
        # recv params for aggregation
        models = self._collect(suffix=suffix, party_idx=party_idx)
        agg_result = None
        # Aggregate numpy groups
        if isinstance(models[0], list):
            # aggregation
            agg_result = models[0]
            # aggregate numpy model weights from all clients
            for params_group in models[1:]:
                for agg_params, params in zip(
                        agg_result, params_group):
                    for agg_p, p in zip(agg_params, params):
                        # agg_p: NumpyWeights or numpy array
                        agg_p += p
        else:
            raise ValueError('invalid aggregation format: {}'.format(models))

        if agg_result is None:
            raise ValueError(
                'can not aggregate receive model, format is illegal: {}'.format(models))
        
        return agg_result

    def _aggregate_loss(self, party_idx=-1):

        # get loss
        loss_suffix = self.suffix['local_loss']()
        losses = self._collect(suffix=loss_suffix, party_idx=-1)
        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss += loss

        return total_loss

    """
    User API
    """

    def model_aggregation(self, party_idx=-1):
        agg_model = self._aggregate_model(party_idx=party_idx)
        suffix = self.suffix['agg_model']()
        self._broadcast(agg_model, suffix=suffix, party_idx=party_idx)
        return agg_model

    def loss_aggregation(self, party_idx=-1):
        agg_loss = self._aggregate_loss(party_idx=party_idx)
        return agg_loss