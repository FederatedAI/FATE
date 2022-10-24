import json
import torch
import tempfile
import inspect
from fate_arch.session import computing_session
from federatedml.model_base import ModelBase
from federatedml.nn.dataset.base import get_dataset_class, Dataset
from federatedml.nn.homo.trainer.trainer_base import get_trainer_class, TrainerBase
from federatedml.nn.dataset.table import TableDataset
from federatedml.nn.dataset.image import ImageDataset
from federatedml.param.homo_nn_param import HomoNNParam
from federatedml.nn.backend.torch import serialization as s
from federatedml.nn.backend.torch.base import FateTorchOptimizer
from federatedml.model_base import MetricMeta
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.nn.backend.util import global_seed, get_homo_model_dict, get_homo_param_meta
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


class HomoNNTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        # checkpoint history
        self.ckp_history = self._create_variable(name='ckp_history', src=['host'], dst=['arbiter'])


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
        self.validation_freq = None
        self.nn_define = None

        # running varialbles
        self.trainer_inst = None

        # export model
        self.model_loaded = False
        self.model = None

        # cache dataset
        self.cache_dataset = {}

        # dtable partitions
        self.partitions = 4

        # transfer var
        self.transfer_variable = HomoNNTransferVariable(self.flowid)

    def _init_model(self, param: HomoNNParam):

        train_param = param.trainer.to_dict()
        dataset_param = param.dataset.to_dict()
        self.trainer = train_param['trainer_name']
        self.dataset = dataset_param['dataset_name']
        self.trainer_param = train_param['param']
        self.dataset_param = dataset_param['param']
        self.torch_seed = param.torch_seed
        self.validation_freq = param.validation_freq
        self.nn_define = param.nn_define
        self.loss = param.loss
        self.optimizer = param.optimizer

    def try_dataset_class(self, dataset_class, path):
        # try default dataset
        try:
            dataset_inst: Dataset = dataset_class(**self.dataset_param)
            dataset_inst.load(path)
            return dataset_inst
        except Exception as e:
            LOGGER.warning('try to load dataset failed, exception :{}'.format(e))
            return None

    def load_dataset(self, data_path_or_dtable):

        # load dataset class
        if isinstance(data_path_or_dtable, str):
            cached_id = data_path_or_dtable
        else:
            cached_id = id(data_path_or_dtable)

        if cached_id in self.cache_dataset:
            LOGGER.debug('use cached dataset, cached id {}'.format(cached_id))
            return self.cache_dataset[cached_id]

        if self.dataset is None or self.dataset == '':
            # automatically match default dataset
            LOGGER.info('dataset is not specified, use auto inference')
            
            for ds_class in [TableDataset, ImageDataset]:
                dataset_inst = self.try_dataset_class(ds_class, data_path_or_dtable)
                if dataset_inst is not None:
                    break
            if dataset_inst is None:
                raise ValueError('cannot find default dataset that can successfully load data from path {}, '
                                 'please check the warning message for error details'.
                                 format(data_path_or_dtable))
        else:
            # load specified dataset
            dataset_class = get_dataset_class(self.dataset)
            dataset_inst = dataset_class(**self.dataset_param)
            dataset_inst.load(data_path_or_dtable)

        if isinstance(data_path_or_dtable, str):
            self.cache_dataset[data_path_or_dtable] = dataset_inst
        else:
            self.cache_dataset[id(data_path_or_dtable)] = dataset_inst

        return dataset_inst

    # read model from model bytes
    @staticmethod
    def recover_model_bytes(model_bytes):
        with tempfile.TemporaryFile() as f:
            f.write(model_bytes)
            f.seek(0)
            model_dict = torch.load(f)
        return model_dict

    def init(self):

        # set random seed
        global_seed(self.torch_seed)

        # load trainer class
        if self.trainer is None:
            raise ValueError('Trainer is not specified, please specify your trainer')
            
        trainer_class = get_trainer_class(self.trainer)
        LOGGER.info('trainer class is {}'.format(trainer_class))

        # recover model from model config / or recover from saved model param
        loaded_model_dict = None
        
        # if has model protobuf, load model config from protobuf
        load_opt_state_dict = False
        if self.model_loaded:
            param, meta = self.model

            if param is None or meta is None:
                raise ValueError('model protobuf is None, make sure'
                                 'that your trainer calls export_model() function to save models')

            if meta.nn_define[0] is None:
                raise ValueError('nn_define is None, model protobuf has no nn-define, make sure'
                                 'that your trainer calls export_model() function to save models')

            self.nn_define = json.loads(meta.nn_define[0])
            loss = json.loads(meta.loss_func_define[0])
            optimizer = json.loads(meta.optimizer_define[0])
            loaded_model_dict = self.recover_model_bytes(param.model_bytes)
            
            if self.optimizer is not None and optimizer != self.optimizer:
                LOGGER.info('optimizer updated')
            else:
                self.optimizer = optimizer
                load_opt_state_dict = True
                
            if self.loss is not None and self.loss != loss:
                LOGGER.info('loss updated')
            else:
                self.loss = loss

        # check key param
        if self.nn_define is None:
            raise ValueError('Model structure is not defined, nn_define is None, please check your param')
        
        # get model from nn define
        model = s.recover_sequential_from_dict(self.nn_define)
        if loaded_model_dict:
            model.load_state_dict(loaded_model_dict['model'])
            LOGGER.info('load model state dict from check point')

        LOGGER.info('model structure is {}'.format(model))
        # init optimizer
        if self.optimizer is not None:
            optimizer_: FateTorchOptimizer = s.recover_optimizer_from_dict(self.optimizer)
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

        trainer_train_args = inspect.getfullargspec(trainer_inst.train).args
        args_format = ['self', 'train_set', 'validate_set', 'optimizer', 'loss']
        if len(trainer_train_args) < 5:
            raise ValueError('Train function of trainer should take 5 arguments :{}, but current trainer.train '
                             'only takes {} arguments: {}'.format(args_format, len(trainer_train_args),
                                                                  trainer_train_args))

        trainer_inst.set_nn_config(self.nn_define, self.optimizer, self.loss)

        return trainer_inst, model, optimizer, loss_fn

    def fit(self, train_input, validate_input=None):
        
        # train input & validate input are DTables or path str
        
        # fate loss callback setting
        self.callback_meta("loss", "train", MetricMeta(name="train", metric_type="LOSS", extra_metas={"unit_name": "iters"}))

        # set random seed
        global_seed(self.torch_seed)

        self.trainer_inst, model, optimizer, loss_fn = self.init()
        self.trainer_inst.set_model(model)
        self.trainer_inst.set_tracker(self.tracker)

        # load dataset class
        dataset_inst = self.load_dataset(train_input)
        LOGGER.info('train dataset instance is {}'.format(dataset_inst))
        
        if validate_input:
            val_dataset_inst = self.load_dataset(validate_input)
            LOGGER.info('validate dataset instance is {}'.format(dataset_inst))
        else:
            val_dataset_inst = None

        # set dataset prefix
        dataset_inst.set_type('train')
        # set model check point
        self.callback_list.callback_list.append(ModelCheckpoint(self, save_freq=1))
        self.trainer_inst.init_checkpoint(self.callback_list.callback_list[0])
        self.trainer_inst.train(dataset_inst, val_dataset_inst, optimizer, loss_fn)
                               
        # training is done, get exported model
        self.model = self.trainer_inst.get_cached_model()

        # sync check point history
        ckp_history = self.trainer_inst.get_checkpoint_history()
        self.transfer_variable.ckp_history.remote(ckp_history)

    def predict(self, cpn_input):

        LOGGER.info('running predict')
        if self.trainer_inst is None:
            # init model
            self.trainer_inst, model, optimizer, loss_fn = self.init()
            self.trainer_inst.set_model(model)
            self.trainer_inst.set_tracker(self.tracker)

        dataset_inst = self.load_dataset(cpn_input)
        if not dataset_inst.has_dataset_type():
            dataset_inst.set_type('predict')
        trainer_ret = self.trainer_inst.predict(dataset_inst)
        if trainer_ret is None:
            LOGGER.info('trainer did not return formatted predicted result, skip predict')
            return None

        id_table, pred_table, classes = trainer_ret
        id_dtable = computing_session.parallelize(id_table, partition=self.partitions, include_key=True)
        pred_dtable = computing_session.parallelize(pred_table, partition=self.partitions, include_key=True)

        return self.predict_score_to_output(id_dtable, pred_dtable, classes)

    def export_model(self):

        if self.model is None:
            return

        return get_homo_model_dict(self.model[0], self.model[1])

    def load_model(self, model_dict):

        model_dict = list(model_dict["model"].values())[0]
        param, meta = get_homo_param_meta(model_dict)
        self.model = (param, meta)
        self.model_loaded = True
