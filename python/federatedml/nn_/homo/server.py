import os
from pathlib import Path
from federatedml.model_base import ModelBase
import importlib
from federatedml.param.homo_cust_nn_param import HomoCustNNParam
from federatedml.framework.homo.aggregator.agg_base import arbiter_get_client_agg_class, get_aggregator_pairs, \
    AggregatorBaseServer
from federatedml.model_base import Metric
from federatedml.model_base import MetricMeta
from federatedml.util import LOGGER

_ml_base = Path(__file__).resolve().parent.parent.parent
IMPORT_PATH = 'federatedml.framework.homo.aggregator'
AGG_PATH = 'framework/homo/aggregator/'


def import_aggregator_modules():
    path_l = os.listdir(str(_ml_base)+'/'+AGG_PATH)
    for f in path_l:
        if f.endswith('.py'):
            importlib.import_module(IMPORT_PATH+'.'+f.replace('.py', ''))


class HomoCustNNServer(ModelBase):

    def __init__(self):
        super(HomoCustNNServer, self).__init__()
        self.model_param = HomoCustNNParam()

    def _init_model(self, param: HomoCustNNParam()):
        pass

    def fit(self, cpn_input=None, cpn_input_2=None):

        import_aggregator_modules()
        LOGGER.debug('class map is {}'.format(get_aggregator_pairs()))
        server_class = arbiter_get_client_agg_class()
        LOGGER.debug('server class is {}'.format(server_class))
        server_agg: AggregatorBaseServer = server_class()
        server_agg.set_tracker(self.tracker)
        
        # fate loss callback setting
        self.callback_meta("loss", "train", MetricMeta(name="train", metric_type="LOSS", extra_metas={"unit_name": "aggregate_round"}))
        
        for i in range(server_agg.get_agg_round()):
            server_agg.aggregate()
            if server_agg.get_converge_status():
                LOGGER.info('training converged, stop training')
                break
            LOGGER.info('aggregate round {} done'.format(i))

    def predict(self, data_inst):
        return None


if __name__ == '__main__':

    import_aggregator_modules()