import importlib
import os
from pathlib import Path

from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.framework.homo.aggregator.agg_base import arbiter_get_client_agg_class, get_aggregator_pairs, \
    AggregatorBaseServer
from federatedml.model_base import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.model_base import serialize_models
from federatedml.nn.homo.client import HomoNNTransferVariable
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.protobuf.generated.homo_nn_model_meta_pb2 import HomoNNMeta as HomoMeta
from federatedml.protobuf.generated.homo_nn_model_param_pb2 import HomoNNParam as HomoParam
from federatedml.util import LOGGER

_ml_base = Path(__file__).resolve().parent.parent.parent
IMPORT_PATH = 'federatedml.framework.homo.aggregator'
AGG_PATH = 'framework/homo/aggregator/'


def import_aggregator_modules():
    path_l = os.listdir(str(_ml_base) + '/' + AGG_PATH)
    for f in path_l:
        if f.endswith('.py'):
            importlib.import_module(IMPORT_PATH + '.' + f.replace('.py', ''))


class HomoNNServer(ModelBase):

    def __init__(self):
        super(HomoNNServer, self).__init__()
        self.model_param = HomoNNParam()
        self.transfer_variable = HomoNNTransferVariable(self.flowid)

    def _init_model(self, param: HomoNNParam()):
        pass

    def fit(self, cpn_input=None, cpn_input_2=None):

        import_aggregator_modules()
        LOGGER.debug('class map is {}'.format(get_aggregator_pairs()))
        server_class = arbiter_get_client_agg_class()
        LOGGER.debug('server class is {}'.format(server_class))
        server_agg: AggregatorBaseServer = server_class()
        server_agg.set_tracker(self.tracker)

        # fate loss callback setting
        self.callback_meta("loss", "train",
                           MetricMeta(name="train", metric_type="LOSS", extra_metas={"unit_name": "aggregate_round"}))

        for i in range(server_agg.get_agg_round()):
            server_agg.aggregate()
            if server_agg.get_converge_status():
                LOGGER.info('training converged, stop training')
                break
            LOGGER.info('aggregate round {} done'.format(i))

        ckp_histories = self.transfer_variable.ckp_history.get(idx=-1, )
        ckp_history = ckp_histories[0]

        if len(ckp_history) == 0:
            return

        for h in ckp_histories[1:]:
            assert ckp_history == h, 'all clients must have same checkpoint history, but got {}'.format(ckp_histories)

        # sever saves empty models, to match with client model checkpoint history
        model_checkpoint = ModelCheckpoint(self, save_freq=1)
        LOGGER.debug('checkpoint history is {}'.format(ckp_history))
        empty_model = {'param': HomoParam(), 'meta': HomoMeta()}
        for step_idx in ckp_history:
            # save empty check point
            model_checkpoint.add_checkpoint(step_index=step_idx, to_save_model=serialize_models(empty_model))

    def predict(self, data_inst):
        return None


if __name__ == '__main__':
    import_aggregator_modules()
