from federatedml.model_base import ModelBase
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.nn.homo.trainer.trainer_base import get_trainer_class
from federatedml.model_base import MetricMeta
from federatedml.util import LOGGER
from federatedml.nn.homo.client import NNModelExporter
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.nn.backend.utils.common import get_homo_param_meta, recover_model_bytes
from federatedml.nn.homo._init import init
from federatedml.util import consts
from federatedml.nn.backend.utils.common import global_seed


class HomoNNServer(ModelBase):

    def __init__(self):
        super(HomoNNServer, self).__init__()
        self.model_param = HomoNNParam()
        self.trainer = consts.FEDAVG_TRAINER
        self.trainer_param = None

        # arbiter side models
        self.model = None
        self.model_loaded = False

        # arbiter saved extra status
        self.exporter = NNModelExporter()
        self.extra_data = {}
        # warm start
        self.model_loaded = False
        self.warm_start_iter = None
        # server init: if arbiter need to load model, loss, optimizer from config
        self.server_init = False

        self.dataset_module = None
        self.dataset = None
        self.dataset_param = {}
        self.torch_seed = None
        self.loss = None
        self.optimizer = None
        self.nn_define = None
        self.ds_config = None

    def export_model(self):

        if self.model is None:
            LOGGER.debug('export an empty model')
            return self.exporter.export_model_dict()  # return an empyty model

        return self.model

    def load_model(self, model_dict):

        if model_dict is not None:
            model_dict = list(model_dict["model"].values())[0]
            self.model = model_dict
            param, meta = get_homo_param_meta(self.model)
            # load extra data
            self.extra_data = recover_model_bytes(param.extra_data_bytes)
            self.warm_start_iter = param.epoch_idx
            self.model_loaded = True

    def _init_model(self, param: HomoNNParam()):

        train_param = param.trainer.to_dict()
        dataset_param = param.dataset.to_dict()
        self.trainer = train_param['trainer_name']
        self.dataset = dataset_param['dataset_name']
        self.trainer_param = train_param['param']
        self.torch_seed = param.torch_seed
        self.nn_define = param.nn_define
        self.loss = param.loss
        self.optimizer = param.optimizer
        self.ds_config = param.ds_config

        LOGGER.debug('trainer and trainer param {} {}'.format(
            self.trainer, self.trainer_param))
        self.server_init = param.server_init

    def fit(self, data_instance=None, validate_data=None):

        # fate loss callback setting
        self.callback_meta(
            "loss", "train", MetricMeta(
                name="train", metric_type="LOSS", extra_metas={
                    "unit_name": "aggregate_round"}))

        # display warmstart iter
        if self.component_properties.is_warm_start:
            self.callback_warm_start_init_iter(self.warm_start_iter)

        if self.server_init:
            LOGGER.info('server try to load model, loss, optimizer from config')
            # init
            global_seed(self.torch_seed)

            trainer_inst, model, optimizer, loss_fn, extra_data, optimizer, loss, self.warm_start_iter = init(
                trainer=self.trainer, trainer_param=self.trainer_param, nn_define=self.nn_define,
                config_optimizer=self.optimizer, config_loss=self.loss, torch_seed=self.torch_seed, model_loaded_flag=self.model_loaded,
                loaded_model=self.model, ds_config=self.ds_config
            )
            trainer_inst.set_model(model)

        else:
            # initialize trainer only
            trainer_class = get_trainer_class(self.trainer)
            trainer_inst = trainer_class(**self.trainer_param)
            LOGGER.info('trainer class is {}'.format(trainer_class))

        # set tracker for fateboard callback
        trainer_inst.set_tracker(self.tracker)
        # set exporter
        trainer_inst.set_model_exporter(self.exporter)
        # set party info
        party_id_list = [self.component_properties.guest_partyid]
        if self.component_properties.host_party_idlist is not None:
            for i in self.component_properties.host_party_idlist:
                party_id_list.append(i)
        trainer_inst.set_party_id_list(party_id_list)
        # set chceckpoint
        trainer_inst.set_checkpoint(ModelCheckpoint(self, save_freq=1))

        # run trainer server procedure
        trainer_inst.server_aggregate_procedure(self.extra_data)

        # aggregation process is done, get exported model if any
        self.model = trainer_inst.get_cached_model()
        self.set_summary(trainer_inst.get_summary())

    def predict(self, data_inst):
        return None
