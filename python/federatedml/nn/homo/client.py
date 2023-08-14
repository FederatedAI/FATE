import json
import torch
import inspect
from fate_arch.computing.non_distributed import LocalData
from fate_arch.computing import is_table
from federatedml.model_base import ModelBase
from federatedml.nn.homo.trainer.trainer_base import get_trainer_class, TrainerBase
from federatedml.nn.backend.utils.data import load_dataset
from federatedml.nn.backend.utils import deepspeed_util
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.nn.backend.torch import serialization as s
from federatedml.nn.backend.torch.base import FateTorchOptimizer
from federatedml.model_base import MetricMeta
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.nn.homo.trainer.trainer_base import StdReturnFormat
from federatedml.nn.backend.utils.common import global_seed, get_homo_model_dict, get_homo_param_meta, recover_model_bytes, get_torch_model_bytes
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.statistic.data_overview import check_with_inst_id
from federatedml.nn.homo.trainer.trainer_base import ExporterBase
from fate_arch.session import computing_session
from federatedml.nn.backend.utils.data import get_ret_predict_table
from federatedml.nn.backend.utils.data import add_match_id
from federatedml.protobuf.generated.homo_nn_model_param_pb2 import HomoNNParam as HomoNNParamPB
from federatedml.protobuf.generated.homo_nn_model_meta_pb2 import HomoNNMeta as HomoNNMetaPB


class NNModelExporter(ExporterBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export_model_dict(
            self,
            model=None,
            optimizer=None,
            model_define=None,
            optimizer_define=None,
            loss_define=None,
            epoch_idx=-1,
            converge_status=False,
            loss_history=None,
            best_epoch=-1,
            local_save_path='',
            extra_data={}):

        if issubclass(type(model), torch.nn.Module):
            model_statedict = model.state_dict()
        else:
            model_statedict = None

        opt_state_dict = None
        if optimizer is not None:
            assert isinstance(optimizer, torch.optim.Optimizer), \
                'optimizer must be an instance of torch.optim.Optimizer'
            opt_state_dict = optimizer.state_dict()

        model_status = {
            'model': model_statedict,
            'optimizer': opt_state_dict,
        }

        model_saved_bytes = get_torch_model_bytes(model_status)
        extra_data_bytes = get_torch_model_bytes(extra_data)

        param = HomoNNParamPB()
        meta = HomoNNMetaPB()

        # save param
        param.model_bytes = model_saved_bytes
        param.extra_data_bytes = extra_data_bytes
        param.epoch_idx = epoch_idx
        param.converge_status = converge_status
        param.best_epoch = best_epoch
        param.local_save_path = local_save_path
        if loss_history is None:
            loss_history = []
        param.loss_history.extend(loss_history)

        # save meta
        meta.nn_define.append(json.dumps(model_define))
        meta.optimizer_define.append(json.dumps(optimizer_define))
        meta.loss_func_define.append(json.dumps(loss_define))

        return get_homo_model_dict(param, meta)


class HomoNNClient(ModelBase):

    def __init__(self):
        super(HomoNNClient, self).__init__()
        self.model_param = HomoNNParam()
        self.trainer = consts.FEDAVG_TRAINER
        self.trainer_param = {}
        self.dataset_module = None
        self.dataset = None
        self.dataset_param = {}
        self.torch_seed = None
        self.loss = None
        self.optimizer = None
        self.nn_define = None

        # running varialbles
        self.trainer_inst = None

        # export model
        self.exporter = NNModelExporter()
        self.model_loaded = False
        self.model = None

        # cache dataset
        self.cache_dataset = {}

        # dtable partitions
        self.partitions = 4

        # warm start display iter
        self.warm_start_iter = None

        # deepspeed
        self.ds_config = None
        self._ds_stage = -1
        self.model_save_flag = False

    def _init_model(self, param: HomoNNParam):

        train_param = param.trainer.to_dict()
        dataset_param = param.dataset.to_dict()
        self.trainer = train_param['trainer_name']
        self.dataset = dataset_param['dataset_name']
        self.trainer_param = train_param['param']
        self.dataset_param = dataset_param['param']
        self.torch_seed = param.torch_seed
        self.nn_define = param.nn_define
        self.loss = param.loss
        self.optimizer = param.optimizer
        self.ds_config = param.ds_config

    def init(self):

        # set random seed
        global_seed(self.torch_seed)

        if self.ds_config:
            deepspeed_util.init_deepspeed_env(self.ds_config)

        # load trainer class
        if self.trainer is None:
            raise ValueError(
                'Trainer is not specified, please specify your trainer')

        trainer_class = get_trainer_class(self.trainer)
        LOGGER.info('trainer class is {}'.format(trainer_class))

        # recover model from model config / or recover from saved model param
        loaded_model_dict = None

        # if has model protobuf, load model config from protobuf
        load_opt_state_dict = False
        if self.model_loaded:

            param, meta = get_homo_param_meta(self.model)
            LOGGER.info('save path is {}'.format(param.local_save_path))
            if param.local_save_path == '':
                LOGGER.info('Load model from model protobuf')
                self.warm_start_iter = param.epoch_idx
                if param is None or meta is None:
                    raise ValueError(
                        'model protobuf is None, make sure'
                        'that your trainer calls export_model() function to save models')

                if meta.nn_define[0] is None:
                    raise ValueError(
                        'nn_define is None, model protobuf has no nn-define, make sure'
                        'that your trainer calls export_model() function to save models')

                self.nn_define = json.loads(meta.nn_define[0])
                loss = json.loads(meta.loss_func_define[0])
                optimizer = json.loads(meta.optimizer_define[0])
                loaded_model_dict = recover_model_bytes(param.model_bytes)
                extra_data = recover_model_bytes(param.extra_data_bytes)
            else:
                LOGGER.info('Load model from local save path')
                save_dict = torch.load(open(param.local_save_path, 'rb'))
                self.warm_start_iter = save_dict['epoch_idx']
                self.nn_define = save_dict['model_define']
                loss = save_dict['loss_define']
                optimizer = save_dict['optimizer_define']
                loaded_model_dict = save_dict
                extra_data = save_dict['extra_data']

            if self.optimizer is not None and optimizer != self.optimizer:
                LOGGER.info('optimizer updated')
            else:
                self.optimizer = optimizer
                load_opt_state_dict = True

            if self.loss is not None and self.loss != loss:
                LOGGER.info('loss updated')
            else:
                self.loss = loss
        else:
            extra_data = {}

        # check key param
        if self.nn_define is None:
            raise ValueError(
                'Model structure is not defined, nn_define is None, please check your param')

        # get model from nn define
        model = s.recover_sequential_from_dict(self.nn_define)
        if loaded_model_dict:
            model.load_state_dict(loaded_model_dict['model'])
            LOGGER.info('load model state dict from check point')

        LOGGER.info('model structure is {}'.format(model))
        # init optimizer
        if self.optimizer is not None and not self.ds_config:
            optimizer_: FateTorchOptimizer = s.recover_optimizer_from_dict(
                self.optimizer)
            # pass model parameters to optimizer
            optimizer = optimizer_.to_torch_instance(model.parameters())
            if load_opt_state_dict:
                LOGGER.info('load optimizer state dict')
                optimizer.load_state_dict(loaded_model_dict['optimizer'])
            LOGGER.info('optimizer is {}'.format(optimizer))
        else:
            optimizer = None
            LOGGER.info('optimizer is not specified')

        # init loss
        if self.loss is not None:
            loss_fn = s.recover_loss_fn_from_dict(self.loss)
            LOGGER.info('loss function is {}'.format(loss_fn))
        else:
            loss_fn = None
            LOGGER.info('loss function is not specified')

        # init trainer
        trainer_inst: TrainerBase = trainer_class(**self.trainer_param)
        LOGGER.info('trainer class is {}'.format(trainer_class))

        trainer_train_args = inspect.getfullargspec(trainer_inst.train).args
        args_format = [
            'self',
            'train_set',
            'validate_set',
            'optimizer',
            'loss',
            'extra_data'
        ]
        if len(trainer_train_args) < 6:
            raise ValueError(
                'Train function of trainer should take 6 arguments :{}, but current trainer.train '
                'only takes {} arguments: {}'.format(
                    args_format, len(trainer_train_args), trainer_train_args))

        trainer_inst.set_nn_config(self.nn_define, self.optimizer, self.loss)
        trainer_inst.fed_mode = True

        if self.ds_config:
            model, optimizer = deepspeed_util.deepspeed_init(model, self.ds_config)
            trainer_inst.enable_deepspeed(is_zero_3=deepspeed_util.is_zero3(self.ds_config))
            if deepspeed_util.is_zero3(self.ds_config):
                model.train()

        return trainer_inst, model, optimizer, loss_fn, extra_data

    def fit(self, train_input, validate_input=None):

        LOGGER.debug('train input is {}'.format(train_input))

        # train input & validate input are DTables or path str
        if not is_table(train_input):
            if isinstance(train_input, LocalData):
                train_input = train_input.path
                assert train_input is not None, 'input train path is None!'

        if not is_table(validate_input):
            if isinstance(validate_input, LocalData):
                validate_input = validate_input.path
                assert validate_input is not None, 'input validate path is None!'

        # fate loss callback setting
        self.callback_meta(
            "loss",
            "train",
            MetricMeta(
                name="train",
                metric_type="LOSS",
                extra_metas={
                    "unit_name": "epochs"}))

        # set random seed
        global_seed(self.torch_seed)

        self.trainer_inst, model, optimizer, loss_fn, extra_data = self.init()
        self.trainer_inst.set_model(model)
        self.trainer_inst.set_tracker(self.tracker)
        self.trainer_inst.set_model_exporter(self.exporter)
        party_id_list = [self.component_properties.guest_partyid]
        for i in self.component_properties.host_party_idlist:
            party_id_list.append(i)
        self.trainer_inst.set_party_id_list(party_id_list)

        # load dataset class
        dataset_inst = load_dataset(
            dataset_name=self.dataset,
            data_path_or_dtable=train_input,
            dataset_cache=self.cache_dataset,
            param=self.dataset_param
        )
        # set dataset prefix
        dataset_inst.set_type('train')
        LOGGER.info('train dataset instance is {}'.format(dataset_inst))

        if validate_input:
            val_dataset_inst = load_dataset(
                dataset_name=self.dataset,
                data_path_or_dtable=validate_input,
                dataset_cache=self.cache_dataset,
                param=self.dataset_param
            )
            if id(val_dataset_inst) != id(dataset_inst):
                dataset_inst.set_type('validate')
            LOGGER.info('validate dataset instance is {}'.format(dataset_inst))
        else:
            val_dataset_inst = None

        # display warmstart iter
        if self.component_properties.is_warm_start:
            self.callback_warm_start_init_iter(self.warm_start_iter)

        # set model check point
        self.trainer_inst.set_checkpoint(ModelCheckpoint(self, save_freq=1))
        # training
        self.trainer_inst.train(
            dataset_inst,
            val_dataset_inst,
            optimizer,
            loss_fn,
            extra_data
        )

        # training is done, get exported model
        self.model = self.trainer_inst.get_cached_model()
        self.set_summary(self.trainer_inst.get_summary())

    def predict(self, cpn_input):

        with_inst_id = False
        schema = None
        if not is_table(cpn_input):
            if isinstance(cpn_input, LocalData):
                cpn_input = cpn_input.path
                assert cpn_input is not None, 'input path is None!'
        elif is_table(cpn_input):
            with_inst_id = check_with_inst_id(cpn_input)
            schema = cpn_input.schema

        LOGGER.info('running predict')
        if self.trainer_inst is None:
            # init model
            self.trainer_inst, model, optimizer, loss_fn, _ = self.init()
            self.trainer_inst.set_model(model)
            self.trainer_inst.set_tracker(self.tracker)

        dataset_inst = load_dataset(
            dataset_name=self.dataset,
            data_path_or_dtable=cpn_input,
            dataset_cache=self.cache_dataset,
            param=self.dataset_param)

        if not dataset_inst.has_dataset_type():
            dataset_inst.set_type('predict')

        trainer_ret = self.trainer_inst.predict(dataset_inst)
        if trainer_ret is None or not isinstance(trainer_ret, StdReturnFormat):
            LOGGER.info(
                'trainer did not return formatted predicted result, skip predict')
            return None

        id_table, pred_table, classes = trainer_ret()

        if with_inst_id:  # set match id
            add_match_id(id_table=id_table, dataset_inst=dataset_inst)

        id_dtable, pred_dtable = get_ret_predict_table(
            id_table, pred_table, classes, self.partitions, computing_session)
        ret_table = self.predict_score_to_output(
            id_dtable, pred_dtable, classes)
        if schema is not None:
            self.set_predict_data_schema(ret_table, schema)

        return ret_table

    def export_model(self):
        if self.model is None:
            LOGGER.debug('export an empty model')
            return self.exporter.export_model_dict()  # return an empty model

        return self.model

    def load_model(self, model_dict):

        model_dict = list(model_dict["model"].values())[0]
        self.model = model_dict
        self.model_loaded = True

    # override function
    @staticmethod
    def set_predict_data_schema(predict_datas, schemas):
        if predict_datas is None:
            return predict_datas
        if isinstance(predict_datas, list):
            predict_data = predict_datas[0]
            schema = schemas[0]
        else:
            predict_data = predict_datas
            schema = schemas
        if predict_data is not None:
            predict_data.schema = {
                "header": [
                    "label",
                    "predict_result",
                    "predict_score",
                    "predict_detail",
                    "type",
                ],
                "sid": 'id',
                "content_type": "predict_result"
            }
            if schema.get("match_id_name") is not None:
                predict_data.schema["match_id_name"] = schema.get(
                    "match_id_name")
        return predict_data
