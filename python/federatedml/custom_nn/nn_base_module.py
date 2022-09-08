import abc
from torch.optim import Optimizer
from federatedml.custom_nn.fed_avg.plaintext_scheduler import FedAvgSchedulerClient


class NNBaseModule(object):

    def __init__(self):
        self.fed_avg_client = FedAvgSchedulerClient()
        self.run_local_mode = False
        self.fed_avg_comm_round = None
        self.role = None
        self.party_id = None
        self._flowid = None

    def set_flowid(self, flowid):
        """
        Set flow id, and initialize transfer variable
        """
        self._flowid = flowid
        self.fed_avg_client.init_transfer_variable(self._flowid)

    def set_role(self, role):
        """
        set self role
        """
        self.role = role

    def local_mode(self):
        """
        set model to local model for local testing
        """
        self.run_local_mode = True

    def fed_mode(self):
        """
        recover federated mode
        """
        self.run_local_mode = False

    def set_fed_avg_round_num(self, comm_round: int = None):
        """
        set the total rounds of federated model averaging, please make sure that every side has same
        communication rounds
        """
        self.fed_avg_comm_round = comm_round
        if self.run_local_mode:
            return
        self.fed_avg_client.set_comm_round(comm_round)
        self.fed_avg_client.sync_communication_round(comm_round)

    def fed_avg_model(self, optimizer: Optimizer, loss: float, loss_weight: float = None, model_weight: float = None):
        """
        Running federation model average to get an aggregated model

        optimizer: pytorch optimizer, this optimizer is with model parameter
        loss: float, loss to aggregate
        loss_weight: float, default is 1, weight for loss aggregation
        model_weight: float, default is 1, weight for model weight aggregation
        """
        if self.run_local_mode:
            return

        if self.fed_avg_comm_round is None:
            raise ValueError('model federation averaging failed, please use "set_fed_avg_round_num()" to set'
                             'communication round before federation')

        if not self.fed_avg_client.has_optimizer():
            self.fed_avg_client.set_optimizer(optimizer)

        self.fed_avg_client.model_fed_avg()
        self.fed_avg_client.inc_step()

    def get_binary_classification_predict_result(self):
        pass

    def get_multi_classification_predict_result(self):
        pass

    def get_regression_predict_result(self):
        pass

    @abc.abstractmethod
    def train(self, cpn_input, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, cpn_input, **kwargs):
        pass
