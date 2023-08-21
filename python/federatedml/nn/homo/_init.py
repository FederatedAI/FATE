import json
import torch
import inspect
from federatedml.nn.homo.trainer.trainer_base import get_trainer_class, TrainerBase
from federatedml.util import LOGGER
from federatedml.nn.backend.torch import serialization as s
from federatedml.nn.backend.torch.base import FateTorchOptimizer
from federatedml.nn.backend.utils.common import recover_model_bytes
from federatedml.nn.backend.utils import deepspeed_util


def init(trainer, trainer_param, nn_define, config_optimizer, config_loss, torch_seed, model_loaded_flag, loaded_model, ds_config):
    
    warm_start_iter = None

    if ds_config:
        deepspeed_util.init_deepspeed_env(ds_config)

    # load trainer class
    if trainer is None:
        raise ValueError(
            'Trainer is not specified, please specify your trainer')

    trainer_class = get_trainer_class(trainer)
    LOGGER.info('trainer class is {}'.format(trainer_class))

    # recover model from model config / or recover from saved model param
    loaded_model_dict = None

    # if has model protobuf, load model config from protobuf
    load_opt_state_dict = False

    if model_loaded_flag:
        param, meta = get_homo_param_meta(loaded_model)
        LOGGER.info('save path is {}'.format(param.local_save_path))
        if param.local_save_path == '':
            LOGGER.info('Load model from model protobuf')
            warm_start_iter = param.epoch_idx
            if param is None or meta is None:
                raise ValueError(
                    'model protobuf is None, make sure'
                    'that your trainer calls export_model() function to save models')

            if meta.nn_define[0] is None:
                raise ValueError(
                    'nn_define is None, model protobuf has no nn-define, make sure'
                    'that your trainer calls export_model() function to save models')

            nn_define = json.loads(meta.nn_define[0])
            loss = json.loads(meta.loss_func_define[0])
            optimizer = json.loads(meta.optimizer_define[0])
            loaded_model_dict = recover_model_bytes(param.model_bytes)
            extra_data = recover_model_bytes(param.extra_data_bytes)

        else:
            LOGGER.info('Load model from local save path')
            save_dict = torch.load(open(param.local_save_path, 'rb'))
            warm_start_iter = save_dict['epoch_idx']
            nn_define = save_dict['model_define']
            loss = save_dict['loss_define']
            optimizer = save_dict['optimizer_define']
            loaded_model_dict = save_dict
            extra_data = save_dict['extra_data']

        if config_optimizer is not None and optimizer != config_optimizer:
            LOGGER.info('optimizer updated')
        else:
            config_optimizer = optimizer
            load_opt_state_dict = True

        if config_loss is not None and config_loss != loss:
            LOGGER.info('loss updated')
        else:
            config_loss = loss
    else:
        extra_data = {}

    # check key param
    if nn_define is None:
        raise ValueError(
            'Model structure is not defined, nn_define is None, please check your param')

    # get model from nn define
    model = s.recover_sequential_from_dict(nn_define)
    if loaded_model_dict:
        model.load_state_dict(loaded_model_dict['model'])
        LOGGER.info('load model state dict from check point')

    LOGGER.info('model structure is {}'.format(model))
    # init optimizer
    if config_optimizer is not None and not ds_config:
        optimizer_: FateTorchOptimizer = s.recover_optimizer_from_dict(
            config_optimizer)
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
    if config_loss is not None:
        loss_fn = s.recover_loss_fn_from_dict(config_loss)
        LOGGER.info('loss function is {}'.format(loss_fn))
    else:
        loss_fn = None
        LOGGER.info('loss function is not specified')

    # init trainer
    trainer_inst: TrainerBase = trainer_class(**trainer_param)
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

    trainer_inst.set_nn_config(nn_define, config_optimizer, config_loss)
    trainer_inst.fed_mode = True

    if ds_config:
        model, optimizer = deepspeed_util.deepspeed_init(model, ds_config)
        trainer_inst.enable_deepspeed(is_zero_3=deepspeed_util.is_zero3(ds_config))
        if deepspeed_util.is_zero3(ds_config):
            model.train()

    return trainer_inst, model, optimizer, loss_fn, extra_data, config_optimizer, config_loss, warm_start_iter