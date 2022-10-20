import numpy as np
import torch as t
from typing import List
from torch.optim import Optimizer
from federatedml.framework.homo.blocks import random_padding_cipher
from federatedml.framework.homo.blocks.random_padding_cipher import RandomPaddingCipherTransVar
from federatedml.secureprotol.encrypt import PadsCipher
from federatedml.framework.homo.aggregator.agg_base import AggregatorBaseClient, AggregatorBaseServer
from federatedml.framework.homo.aggregator.agg_base import aggregator_client, aggregator_server
from federatedml.framework.weights import NumpyWeights
from federatedml.optim.convergence import converge_func_factory
from federatedml.util import LOGGER
from federatedml.util import consts


@aggregator_client('secure_agg')
class SecureAggregatorClient(AggregatorBaseClient):

    def __init__(self, max_aggregate_round: int, secure_aggregate=True, check_convergence=False,
                 convergence_type: str = 'diff', eps=0.0001, aggregate_type: str = 'fedavg',
                 sample_number=None):

        super(SecureAggregatorClient, self).__init__(max_aggregate_round)

        self.secure_aggregate = secure_aggregate
        self.check_convergence = check_convergence
        self.eps = eps
        if check_convergence is True:
            assert convergence_type in ['diff', 'abs']
        self.convergence_type = convergence_type
        self.support_agg_type = ['fedavg', 'sum']
        if aggregate_type not in self.support_agg_type:
            raise ValueError('supported aggregate type {}, but got {}'.format(self.support_agg_type, aggregate_type))
        self.aggregate_type = aggregate_type
        self.send_to_server({'secure_aggregate': self.secure_aggregate, 'check_convergence': self.check_convergence,
                             'eps': self.eps, 'aggregate_type': self.aggregate_type,
                             'convergence_type': self.convergence_type
                             }, suffix='param')

        self._cur_agg_round = 0

        # dh key exchange for secure aggregation
        if self.secure_aggregate:
            trans_var = RandomPaddingCipherTransVar(server=(consts.ARBITER,), clients=(consts.HOST,))
            self._random_padding_cipher: PadsCipher = \
                random_padding_cipher.Client(trans_var=trans_var).create_cipher()
            LOGGER.info('initialize secure aggregator done')

        # compute weights for this party
        self.sample_num = 1 if sample_number is None else sample_number
        if self.aggregate_type == 'fedavg':
            self.send_to_server(np.array(self.sample_num),
                                suffix=('sample_agg',))
            total_number = self.get_from_server(suffix=('sample_agg_result',))
            self._weight = float(self.sample_num / total_number)
            LOGGER.debug('total sample number is {}, weight is {}'.format(total_number, self._weight))
        elif self.aggregate_type == 'sum':
            self._weight = 1.0

    def inc_agg_round(self):
        self._cur_agg_round += 1

    def aggregate_model(self, model: List[List[np.ndarray]]):

        # if secure aggregation, add random mask
        if self.secure_aggregate:
            model = [[NumpyWeights(arr * self._weight).encrypted(self._random_padding_cipher) for arr in arr_list]
                     for arr_list in model]
        else:
            model = [[arr * self._weight for arr in arr_list] for arr_list in model]

        self.send_to_server(model, suffix=(self._cur_agg_round, 'model'))
        agg_model = self.get_from_server(suffix=(self._cur_agg_round,))
        # if secure aggregation, convert back to np ndarray
        if self.secure_aggregate:
            return [[np_weight.unboxed for np_weight in arr_list] for arr_list in agg_model]
        else:
            return agg_model

    def aggregate_loss(self, loss):

        if self.secure_aggregate:
            sync_loss = NumpyWeights(np.array([loss * self._weight])).encrypted(self._random_padding_cipher)
        else:
            sync_loss = loss * self._weight
        self.send_to_server(sync_loss, suffix=(self._cur_agg_round, 'loss'))
        converge_status = self.get_from_server(suffix=(self._cur_agg_round, 'converge_status'))

        return converge_status

    def aggregate(self, model: List[List[np.ndarray]], loss: float):

        agg_model = self.aggregate_model(model)
        converge_status = self.aggregate_loss(loss)
        self.inc_agg_round()

        return agg_model, converge_status


@aggregator_server('secure_agg')
class SecureAggregatorServer(AggregatorBaseServer):

    def __init__(self):
        super(SecureAggregatorServer, self).__init__()
        self.param = self.get_from_clients(suffix='param')[0]
        self.secure_aggregate = self.param['secure_aggregate']
        self.eps = self.param['eps']
        self.aggregate_type = self.param['aggregate_type']
        self.check_convergence = self.param['check_convergence']
        self.convergence_type = self.param['convergence_type']
        self.converge_status = False
        LOGGER.debug('received parameter {}'.format(self.param))

        self.convergence = None
        if self.check_convergence:
            self.convergence = converge_func_factory(self.convergence_type, self.eps)

        if self.secure_aggregate:
            random_padding_cipher.Server(trans_var=RandomPaddingCipherTransVar(server=(consts.ARBITER,),
                                                                               clients=(consts.HOST,))) \
                .exchange_secret_keys()
            LOGGER.info('initialize secure aggregator done')

        self._cur_agg_round = 0

        if self.aggregate_type == 'fedavg':
            total_sample = None
            samples = self.get_from_clients(suffix=('sample_agg',))
            for s in samples:
                if total_sample is None:
                    total_sample = s
                    continue
                total_sample += s
            self.send_to_clients(total_sample, suffix=('sample_agg_result',))

    def inc_agg_round(self):
        self._cur_agg_round += 1

    def get_converge_status(self):
        return self.converge_status

    def aggregate_model(self):
        # recv params for aggregation
        params_groups = self.get_from_clients(suffix=(self._cur_agg_round, "model"))
        # aggregation
        aggregated_params_group = params_groups[0]
        # aggregate numpy model weights from all clients
        for params_group in params_groups[1:]:
            for agg_params, params in zip(aggregated_params_group, params_group):
                for agg_p, p in zip(agg_params, params):
                    # agg_p: NumpyWeights or numpy array
                    agg_p += p

        self.send_to_clients(aggregated_params_group, suffix=(self._cur_agg_round,))

    def aggregate_loss(self):

        # get loss & weight
        losses = self.get_from_clients(suffix=(self._cur_agg_round, "loss"))
        # aggregate loss
        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss += loss
        if self.secure_aggregate:
            total_loss = total_loss.unboxed
        self.save_loss(float(total_loss))
        if self.convergence:
            is_converge = self.convergence.is_converge(total_loss)
        else:
            is_converge = False
        LOGGER.info('aggregated loss is {}, converge is {}'.format(total_loss, is_converge))
        self.send_to_clients(is_converge, suffix=(self._cur_agg_round, 'converge_status'))
        self.converge_status = is_converge

    def aggregate(self):
        self.aggregate_model()
        self.aggregate_loss()
        self.callback_loss(self.loss_history[-1], self._cur_agg_round)
        self.inc_agg_round()


if __name__ == '__main__':
    pass
