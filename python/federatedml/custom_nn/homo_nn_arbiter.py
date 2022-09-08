from federatedml.model_base import ModelBase
from federatedml.custom_nn.fed_avg.plaintext_scheduler import FedAvgSchedulerAggregator
from federatedml.custom_nn.homo_nn_param import CustNNParam


class HomoNNArbiter(ModelBase):

    def __init__(self):
        super(HomoNNArbiter, self).__init__()
        self.model_param = CustNNParam()
        self.fed_avg_aggregator = FedAvgSchedulerAggregator(self.flowid)

    def _init_model(self, param: CustNNParam):
        self.class_name = param.class_name
        self.class_file_name = param.class_file_name
        self.nn_params = param.nn_params

    def fit(self, cpn_input):

        self.fed_avg_aggregator.sync_communication_round()
        for i in range(self.fed_avg_aggregator.get_comm_round()):
            self.fed_avg_aggregator.model_fed_avg()
            self.fed_avg_aggregator.inc_step()

    def predict(self, data_inst):
        return None
