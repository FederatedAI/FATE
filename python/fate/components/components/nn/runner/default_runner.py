import torch as t
import os
from fate.components.components.nn.nn_runner import NNRunner
from fate.ml.nn.algo.homo.fedavg import FedAVG, FedAVGArguments, FedAVGCLient, FedAVGServer, TrainingArguments
from typing import Optional, Dict, Union
from fate.components.components.nn.loader import Loader
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import _LRScheduler
from fate.ml.nn.trainer.trainer_base import FedArguments, TrainingArguments, FedTrainerClient, FedTrainerServer
from typing import Union, Type, Callable, Optional
from transformers.trainer_utils import get_last_checkpoint
from fate.ml.nn.dataset.table import TableDataset
from typing import Literal
import logging
from fate.components.components.utils import consts
from fate.ml.nn.dataset.table import TableDataset
from fate.arch.dataframe import DataFrame


logger = logging.getLogger(__name__)


SUPPORTED_ALGO = ['fedavg']


def load_model_dict_from_path(path):
    # Ensure that the path is a string
    assert isinstance(
        path, str), "Path must be a string, but got {}".format(
        type(path))

    # Append the filename to the path
    model_path = os.path.join(path, 'pytorch_model.bin')

    # Check if the file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"No 'pytorch_model.bin' file found at {model_path}, no saved model found")

    # Load the state dict from the specified path
    model_dict = t.load(model_path)

    return model_dict


def dir_warning(train_args):
    if 'output_dir' in train_args or 'logging_dir' in train_args or 'resume_from_checkpoint' in train_args:
        logger.warning(
            "The output_dir, logging_dir, and resume_from_checkpoint arguments are not supported in the "
            "DefaultRunner when running the Pipeline. These arguments will be replaced by FATE provided paths.")


class SetupReturn:
    """
    Class to encapsulate the return objects from the setup.
    """

    def __init__(self,
                 trainer: Union[Type[FedTrainerClient],
                                Type[FedTrainerServer]] = None,
                 model: Type[nn.Module] = None,
                 optimizer: Type[optim.Optimizer] = None,
                 loss: Callable = None,
                 scheduler: Type[_LRScheduler] = None,
                 train_args: TrainingArguments = None,
                 fed_args: FedArguments = None,
                 data_collator: Callable = None) -> None:

        if trainer is not None and not (
            issubclass(
                type(trainer),
                FedTrainerClient) or issubclass(
                type(trainer),
                FedTrainerServer)):
            raise TypeError(
                f"SetupReturn Error: trainer must be a subclass of either FedTrainerClient or FedTrainerServer but got {type(trainer)}")

        if model is not None and not issubclass(type(model), nn.Module):
            raise TypeError(
                f"SetupReturn Error: model must be a subclass of torch.nn.Module but got {type(model)}")

        if optimizer is not None and not issubclass(
                type(optimizer), optim.Optimizer):
            raise TypeError(
                f"SetupReturn Error: optimizer must be a subclass of torch.optim.Optimizer but got {type(optimizer)}")

        if loss is not None and not callable(loss):
            raise TypeError(
                f"SetupReturn Error: loss must be callable but got {type(loss)}")

        if scheduler is not None and not issubclass(
                type(scheduler), _LRScheduler):
            raise TypeError(
                f"SetupReturn Error: scheduler must be a subclass of torch.optim.lr_scheduler._LRScheduler but got {type(scheduler)}")

        if train_args is not None and not isinstance(
                train_args, TrainingArguments):
            raise TypeError(
                f"SetupReturn Error: train_args must be an instance of TrainingArguments but got {type(train_args)}")

        if fed_args is not None and not isinstance(fed_args, FedArguments):
            raise TypeError(
                f"SetupReturn Error: fed_args must be an instance of FedArguments but got {type(fed_args)}")

        if data_collator is not None and not callable(data_collator):
            raise TypeError(
                f"SetupReturn Error: data_collator must be callable but got {type(data_collator)}")

        self.trainer = trainer
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.train_args = train_args
        self.fed_args = fed_args
        self.data_collator = data_collator

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        repr_string = "SetupReturn(\n"
        for key, value in self.__dict__.items():
            repr_string += f"  {key}={type(value)},\n"
        repr_string = repr_string.rstrip(',\n')
        repr_string += "\n)"
        return repr_string


class DefaultRunner(NNRunner):

    def __init__(self,
                 algo: str = 'fedavg',
                 model_conf: Optional[Dict] = None,
                 dataset_conf: Optional[Dict] = None,
                 optimizer_conf: Optional[Dict] = None,
                 training_args_conf: Optional[Dict] = None,
                 fed_args_conf: Optional[Dict] = None,
                 loss_conf: Optional[Dict] = None,
                 data_collator_conf: Optional[Dict] = None,
                 tokenizer_conf: Optional[Dict] = None,
                 task_type: Literal['binary',
                                    'multi',
                                    'regression',
                                    'others'] = 'binary',
                 threshold: float = 0.5,
                 local_mode: bool = False) -> None:

        super().__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.dataset_conf = dataset_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.fed_args_conf = fed_args_conf
        self.loss_conf = loss_conf
        self.data_collator_conf = data_collator_conf
        self.local_mode = local_mode
        self.tokenizer_conf = tokenizer_conf
        self.task_type = task_type
        self.threshold = threshold

        # check param
        if self.algo not in SUPPORTED_ALGO:
            raise ValueError('algo should be one of [fedavg]')
        if self.task_type not in ['binary', 'multi', 'regression', 'others']:
            raise ValueError(
                'task_type should be one of [binary, multi, regression, others]')
        assert self.threshold >= 0 and self.threshold <= 1, 'threshold should be in [0, 1]'
        assert isinstance(self.local_mode, bool), 'local should be bool'

        # setup var
        self.trainer = None

    def _loader_load_from_conf(self, conf, return_class=False):
        if conf is None:
            return None
        if return_class:
            return Loader.from_dict(conf).load_item()
        return Loader.from_dict(conf).call_item()

    def _prepare_data(self, data, data_name) -> SetupReturn:

        if data is None:
            return None
        if isinstance(data, DataFrame) and self.dataset_conf is None:
            logger.info(
                'Input data {} is FATE DataFrame and dataset conf is None, will automatically handle the input data'.format(data_name))
            if self.task_type == consts.MULTI:
                dataset = TableDataset(
                    flatten_label=True,
                    label_dtype='long',
                    to_tensor=True)
            else:
                dataset = TableDataset(to_tensor=True)
            dataset.load(data)
        else:
            dataset = self._loader_load_from_conf(self.dataset_conf)
            if hasattr(dataset, 'load'):
                dataset.load(data)
            else:
                raise ValueError(
                    f"The dataset {dataset} lacks a load() method, which is required for data parsing in the DefaultRunner. \
                                Please implement this method in your dataset class. You can refer to the base class 'Dataset' in 'fate.ml.nn.dataset.base' \
                                for the necessary interfaces to implement.")
        if dataset is not None and not issubclass(
                type(dataset), data_utils.Dataset):
            raise TypeError(
                f"SetupReturn Error: {data_name}_set must be a subclass of torch.utils.data.Dataset but got {type(dataset)}")

        return dataset

    def client_setup(
            self,
            train_set=None,
            validate_set=None,
            output_dir=None,
            saved_model=None,
            stage='train'):

        if stage == 'predict':
            self.local_mode = True

        if self.algo == 'fedavg':
            client_class: FedAVGCLient = FedAVG.client
        else:
            raise ValueError(f"algo {self.algo} not supported")

        ctx = self.get_context()
        print(self.model_conf)
        model = self._loader_load_from_conf(self.model_conf)
        if model is None:
            raise ValueError(
                f"model is None, cannot load model from conf {self.model_conf}")

        if output_dir is None:
            output_dir = './'

        resume_path = None
        if saved_model is not None:
            model_dict = load_model_dict_from_path(saved_model)
            model.load_state_dict(model_dict)
            logger.info(f"loading model dict from {saved_model} to model done")
            if get_last_checkpoint(saved_model) is not None:
                resume_path = saved_model
                logger.info(
                    f"checkpoint detected, resume_path set to {resume_path}")
        # load optimizer
        optimizer_loader = Loader.from_dict(self.optimizer_conf)
        optimizer_ = optimizer_loader.load_item()
        optimizer_params = optimizer_loader.kwargs
        optimizer = optimizer_(model.parameters(), **optimizer_params)
        # load loss
        loss = self._loader_load_from_conf(self.loss_conf)
        # load collator func
        data_collator = self._loader_load_from_conf(self.data_collator_conf)
        # load tokenizer if import conf provided
        tokenizer = self._loader_load_from_conf(self.tokenizer_conf)
        # args
        dir_warning(self.training_args_conf)
        training_args = TrainingArguments(**self.training_args_conf)
        # reset to default, saving to arbitrary path is not allowed in
        # DefaultRunner
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path
        fed_args = FedAVGArguments(**self.fed_args_conf)

        # prepare trainer
        trainer = client_class(
            ctx=ctx,
            model=model,
            loss_fn=loss,
            optimizer=optimizer,
            training_args=training_args,
            fed_args=fed_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_set=train_set,
            val_set=validate_set,
            local_mode=self.local_mode)

        return SetupReturn(
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_args=training_args,
            fed_args=fed_args,
            data_collator=data_collator)

    def server_setup(self, stage='train'):

        if stage == 'predict':
            self.local_mode = True
        if self.algo == 'fedavg':
            server_class: FedAVGServer = FedAVG.server
        else:
            raise ValueError(f"algo {self.algo} not supported")
        ctx = self.get_context()
        trainer = server_class(ctx=ctx, local_mode=self.local_mode)
        return SetupReturn(trainer=trainer)

    def train(self,
              train_data: Optional[Union[str,
                                         DataFrame]] = None,
              validate_data: Optional[Union[str,
                                            DataFrame]] = None,
              output_dir: str = None,
              saved_model_path: str = None):

        if self.is_client():
            train_set = self._prepare_data(train_data, 'train_data')
            validate_set = self._prepare_data(validate_data, 'val_data')
            setup = self.client_setup(
                train_set=train_set,
                validate_set=validate_set,
                output_dir=output_dir,
                saved_model=saved_model_path)
            trainer = setup['trainer']
            self.trainer = trainer
            trainer.train()
            if output_dir is not None:
                trainer.save_model(output_dir)
        elif self.is_server():
            setup = self.server_setup()
            trainer = setup['trainer']
            trainer.train()

    def _run_dataset_func(self, dataset, func_name):

        if hasattr(dataset, func_name):
            output = getattr(dataset, func_name)()
            if output is None:
                logger.info(
                    f'dataset {type(dataset)}: {func_name} returns None, this will influence the output of predict')
            return output
        else:
            logger.info(
                f'dataset {type(dataset)} not implemented {func_name}, classes set to None, this will influence the output of predict')
            return None

    def predict(self,
                test_data: Union[str,
                                 DataFrame],
                saved_model_path: str = None) -> Union[DataFrame,
                                                       None]:

        if self.is_client():
            test_set = self._prepare_data(test_data, 'test_data')
            if self.trainer is not None:
                trainer = self.trainer
                logger.info('trainer found, skip setting up')
            else:
                setup = self.client_setup(
                    saved_model=saved_model_path, stage='predict')
                trainer = setup['trainer']

            classes = self._run_dataset_func(test_set, 'get_classes')
            match_ids = self._run_dataset_func(test_set, 'get_match_ids')
            sample_ids = self._run_dataset_func(test_set, 'get_sample_ids')
            match_id_name = self._run_dataset_func(
                test_set, 'get_match_id_name')
            sample_id_name = self._run_dataset_func(
                test_set, 'get_sample_id_name')
            pred_rs = trainer.predict(test_set)
            rs_df = self.get_nn_output_dataframe(
                self.get_context(),
                pred_rs.predictions,
                pred_rs.label_ids if hasattr(pred_rs, 'label_ids') else None,
                match_ids,
                sample_ids,
                match_id_name=match_id_name,
                sample_id_name=sample_id_name,
                dataframe_format='fate_std',
                task_type=self.task_type,
                classes=classes)
            return rs_df
        else:
            # server not predict
            return
