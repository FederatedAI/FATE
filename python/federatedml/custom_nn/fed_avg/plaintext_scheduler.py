import torch
from torch.optim import Optimizer
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables
from federatedml.util import LOGGER


class FedAvgSchedulerTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.client_params = self._create_variable(
            "client_params", src=["guest", "host"], dst=["arbiter"]
        )
        self.aggregated_params = self._create_variable(
            "aggregated_params", dst=["guest", "host"], src=["arbiter"]
        )
        self.client_loss = self._create_variable(
            "client_loss", src=["guest", "host"], dst=["arbiter"]
        )
        self.aggregated_loss = self._create_variable(
            "aggregated_loss", dst=["guest", "host"], src=["arbiter"]
        )
        self.comm_round = self._create_variable(
            "comm_round", src=["guest", "host"], dst=["arbiter"]
        )


class FedAvgSchedulerClient(object):

    def __init__(self):
        self.optimizer = None
        self.transfer_variable = None
        self._step = 0
        self._comm_round = None

    def init_transfer_variable(self, flowid):
        self.transfer_variable = FedAvgSchedulerTransferVariable()
        self.transfer_variable.set_flowid(flowid)

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def set_comm_round(self, comm_round):
        self._comm_round = comm_round

    def has_optimizer(self):
        return self.optimizer is not None

    def get_step(self):
        return self._step

    def sync_communication_round(self, comm_round):
        self.transfer_variable.comm_round.remote(comm_round, suffix='comm-round')

    def model_fed_avg(self):
        # get optimizer params
        params_group = [[p.detach().numpy() for p in group["params"]] for group in self.optimizer.param_groups]

        # send for aggregation
        self.transfer_variable.client_params.remote(params_group, suffix=(self._step,))

        # recv aggregated params
        agg_params_group = self.transfer_variable.aggregated_params.get(idx=0, suffix=(self._step,))

        # LOGGER.debug('recv parameter {}'.format(agg_params_group))

        # set aggregated params
        for agg_group, group in zip(agg_params_group, self.optimizer.param_groups):
            for agg_p, p in zip(agg_group, group["params"]):
                p.data.copy_(torch.Tensor(agg_p))

    def inc_step(self):
        self._step += 1
        if self._step > self._comm_round:
            raise ValueError('communication round exceeds pre-set communication round: cur round {}, fed avg round {}'
                             .format(self._step, self._comm_round))


class FedAvgSchedulerAggregator(object):
    def __init__(self, flowid=''):
        self.transfer_variable = FedAvgSchedulerTransferVariable()
        self._step = 0
        self._comm_round = None
        self.transfer_variable.set_flowid(flowid)

    def sync_communication_round(self):
        self._comm_round = self.transfer_variable.comm_round.get(idx=-1, suffix='comm-round')[0]

    def get_comm_round(self):
        return self._comm_round

    def model_fed_avg(self):
        # recv params for aggregation
        params_groups = self.transfer_variable.client_params.get(suffix=(self._step,))

        # aggregated
        aggregated_params_group = params_groups[0]
        n = len(params_groups)

        for params_group in params_groups[1:]:
            for agg_params, params in zip(aggregated_params_group, params_group):
                for agg_p, p in zip(agg_params, params):
                    agg_p += p

        for agg_params in aggregated_params_group:
            for agg_p in agg_params:
                agg_p /= n

        # send aggregated
        self.transfer_variable.aggregated_params.remote(aggregated_params_group, suffix=(self._step,))

    def inc_step(self):
        self._step += 1
        if self._step > self._comm_round:
            raise ValueError('communication round exceeds pre-set communication round: cur round {}, fed avg round {}'
                             .format(self._step, self._comm_round))
