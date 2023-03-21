from federatedml.model_base import ModelBase
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.nn.homo.trainer.trainer_base import get_trainer_class
from federatedml.model_base import MetricMeta
from federatedml.util import LOGGER
from federatedml.nn.homo.client import NNModelExporter
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.nn.backend.utils.common import get_homo_param_meta, recover_model_bytes


class HomoNNServer(ModelBase):

    def __init__(self):
        super(HomoNNServer, self).__init__()
        self.model_param = HomoNNParam()
        self.trainer = None
        self.trainer_param = None

        # arbiter side models
        self.model = None
        self.model_loaded = False

        # arbiter saved extra status
        self.exporter = NNModelExporter()
        self.extra_data = {}
        # warm start
        self.warm_start_iter = None

    def export_model(self):

        if self.model is None:
            LOGGER.debug('export an empty model')
            return self.exporter.export_model_dict()  # return an exporter

        return self.model

    def load_model(self, model_dict):

        if model_dict is not None:
            model_dict = list(model_dict["model"].values())[0]
            self.model = model_dict
            param, meta = get_homo_param_meta(self.model)
            # load extra data
            self.extra_data = recover_model_bytes(param.extra_data_bytes)
            self.warm_start_iter = param.epoch_idx

    def _init_model(self, param: HomoNNParam()):
        train_param = param.trainer.to_dict()
        self.trainer = train_param['trainer_name']
        self.trainer_param = train_param['param']
        LOGGER.debug('trainer and trainer param {} {}'.format(
            self.trainer, self.trainer_param))

    def fit(self, data_instance=None, validate_data=None):

        # fate loss callback setting
        self.callback_meta(
            "loss", "train", MetricMeta(
                name="train", metric_type="LOSS", extra_metas={
                    "unit_name": "aggregate_round"}))

        # display warmstart iter
        if self.component_properties.is_warm_start:
            self.callback_warm_start_init_iter(self.warm_start_iter)

        # initialize trainer
        trainer_class = get_trainer_class(self.trainer)
        LOGGER.info('trainer class is {}'.format(trainer_class))
        # init trainer
        trainer_inst = trainer_class(**self.trainer_param)
        # set tracker for fateboard callback
        trainer_inst.set_tracker(self.tracker)
        # set exporter
        trainer_inst.set_model_exporter(self.exporter)
        # set chceckpoint
        trainer_inst.set_checkpoint(ModelCheckpoint(self, save_freq=1))
        # run trainer server procedure
        trainer_inst.server_aggregate_procedure(self.extra_data)

        # aggregation process is done, get exported model if any
        self.model = trainer_inst.get_cached_model()
        self.set_summary(trainer_inst.get_summary())

    def predict(self, data_inst):
        return None
