import torch as t
import os
from fate.components.components.nn.nn_runner import NNInput, NNRunner, NNOutput
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
from typing import Literal
import logging


logger = logging.getLogger(__name__)


SUPPORTED_ALGO = ['fedavg']


def load_model_dict_from_path(path):
    # Ensure that the path is a string
    assert isinstance(path, str), "Path must be a string"

    # Append the filename to the path
    model_path = os.path.join(path, 'pytorch_model.bin')

    # Check if the file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No 'pytorch_model.bin' file found at {model_path}, no saved model found")

    # Load the state dict from the specified path
    model_dict = t.load(model_path)

    return model_dict


def dir_warning(train_args):
    if 'output_dir' in train_args or 'logging_dir' in train_args or 'resume_from_checkpoint' in train_args:
        logger.warning("The output_dir, logging_dir, and resume_from_checkpoint arguments are not supported in the "
                       "DefaultRunner when running the Pipeline. These arguments will be replaced by FATE provided paths.")


class SetupReturn:
    """
    Class to encapsulate the return objects from the setup.
    """

    def __init__(self,
             trainer: Union[Type[FedTrainerClient], Type[FedTrainerServer]] = None,
             model: Type[nn.Module] = None,
             train_set: Type[data_utils.Dataset] = None,
             validate_set: Type[data_utils.Dataset] = None,
             test_set: Type[data_utils.Dataset] = None,
             optimizer: Type[optim.Optimizer] = None,
             loss: Callable = None,
             scheduler: Type[_LRScheduler] = None,
             train_args: TrainingArguments = None,
             fed_args: FedArguments = None,
             data_collator: Callable = None) -> None:

        if trainer is not None and not (issubclass(type(trainer), FedTrainerClient) or issubclass(type(trainer), FedTrainerServer)):
            raise TypeError(f"SetupReturn Error: trainer must be a subclass of either FedTrainerClient or FedTrainerServer but got {type(trainer)}")
            
        if model is not None and not issubclass(type(model), nn.Module):
            raise TypeError(f"SetupReturn Error: model must be a subclass of torch.nn.Module but got {type(model)}")
            
        if train_set is not None and not issubclass(type(train_set), data_utils.Dataset):
            raise TypeError(f"SetupReturn Error: train_set must be a subclass of torch.utils.data.Dataset but got {type(train_set)}")
            
        if validate_set is not None and not issubclass(type(validate_set), data_utils.Dataset):
            raise TypeError(f"SetupReturn Error: validate_set must be a subclass of torch.utils.data.Dataset but got {type(validate_set)}")
            
        if test_set is not None and not issubclass(type(test_set), data_utils.Dataset):
            raise TypeError(f"SetupReturn Error: test_set must be a subclass of torch.utils.data.Dataset but got {type(test_set)}")
            
        if optimizer is not None and not issubclass(type(optimizer), optim.Optimizer):
            raise TypeError(f"SetupReturn Error: optimizer must be a subclass of torch.optim.Optimizer but got {type(optimizer)}")
            
        if loss is not None and not callable(loss):
            raise TypeError(f"SetupReturn Error: loss must be callable but got {type(loss)}")
            
        if scheduler is not None and not issubclass(type(scheduler), _LRScheduler):
            raise TypeError(f"SetupReturn Error: scheduler must be a subclass of torch.optim.lr_scheduler._LRScheduler but got {type(scheduler)}")
            
        if train_args is not None and not isinstance(train_args, TrainingArguments):
            raise TypeError(f"SetupReturn Error: train_args must be an instance of TrainingArguments but got {type(train_args)}")
            
        if fed_args is not None and not isinstance(fed_args, FedArguments):
            raise TypeError(f"SetupReturn Error: fed_args must be an instance of FedArguments but got {type(fed_args)}")
            
        if data_collator is not None and not callable(data_collator):
            raise TypeError(f"SetupReturn Error: data_collator must be callable but got {type(data_collator)}")

        self.trainer = trainer
        self.model = model
        self.train_set = train_set
        self.validate_set = validate_set
        self.test_set = test_set
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
                 task_type: Literal['binary', 'multi', 'regression', 'others'] = 'binary',
                 use_hf_default_behavior: bool = False,
                 local_mode: bool = False
                ) -> None:
        
        super().__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.dataset_conf = dataset_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.fed_args_conf = fed_args_conf
        self.loss_conf = loss_conf
        self.data_collator_conf = data_collator_conf
        self.use_hf_default_behavior = use_hf_default_behavior
        self.local_mode = local_mode
        self.tokenizer_conf = tokenizer_conf
        self.task_type = task_type

    def _loader_load_from_conf(self, conf, return_class=False):
        if conf is None:
            return None
        if return_class:
            return Loader.from_dict(conf).load_item()
        return Loader.from_dict(conf).call_item()

    def _prepare_dataset(self, dataset_conf, cpn_input_data):
        dataset = self._loader_load_from_conf(dataset_conf)
        if hasattr(dataset, 'load'):
            if cpn_input_data is not None:
                dataset.load(cpn_input_data)
                return dataset
            else:
                return None
        else:
            raise ValueError(f"dataset {dataset} has no load() method")

    def setup(self, cpn_input_data: NNInput, stage='train'):

        if stage == 'predict':
            self.local_mode = True

        if self.algo == 'fedavg':
            client_class: FedAVGCLient = FedAVG.client
            server_class: FedAVGServer = FedAVG.server
        else:
            raise ValueError(f"algo {self.algo} not supported")

        ctx = self.get_context()
            
        if self.is_client():

            # load arguments, models, etc
            # prepare datatset
            # dataet
            train_set = self._prepare_dataset(self.dataset_conf, cpn_input_data.get_train_data())
            validate_set = self._prepare_dataset(self.dataset_conf, cpn_input_data.get_validate_data())
            test_set = self._prepare_dataset(self.dataset_conf, cpn_input_data.get_test_data())
            # load model
            model = self._loader_load_from_conf(self.model_conf)
            if model is None:
                raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")
            # save path: path to save provided by fate framework
            save_path = cpn_input_data.get_fate_save_path()
            # if have input model for warm-start 
            model_path = cpn_input_data.get_saved_model_path()
            # resume_from checkpoint path
            resume_path = None
            
            if model_path is not None:
                model_dict = load_model_dict_from_path(model_path)
                model.load_state_dict(model_dict)
                logger.info(f"loading model dict from {model_path} to model done")
                if get_last_checkpoint(model_path) is not None:
                    resume_path = model_path
                    logger.info(f"checkpoint detected, resume_path set to {resume_path}")

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
            training_args.output_dir = save_path  # reset to default, saving to arbitrary path is not allowed in NN component
            training_args.resume_from_checkpoint = resume_path  # resume path
            fed_args = FedAVGArguments(**self.fed_args_conf)

            # prepare trainer
            trainer = client_class(ctx=ctx, model=model, loss_fn=loss,
                                   optimizer=optimizer, training_args=training_args,
                                   fed_args=fed_args, data_collator=data_collator,
                                   tokenizer=tokenizer, train_set=train_set, val_set=validate_set, local_mode=self.local_mode)
            
            return SetupReturn(trainer=trainer, model=model, optimizer=optimizer, loss=loss, 
                               train_args=training_args, fed_args=fed_args, data_collator=data_collator,
                               train_set=train_set, validate_set=validate_set, test_set=test_set)

        elif self.is_server():
            trainer = server_class(ctx=ctx, local_mode=self.local_mode)
            return SetupReturn(trainer=trainer)
        
    def train(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        
        
        setup = self.setup(input_data, stage='train')
        trainer = setup['trainer']
        if self.is_client():

            trainer.train()
            trainer.save_model(input_data.get('fate_save_path'))
            # predict the dataset when training is done
            train_rs = trainer.predict(setup['train_set']) if setup['train_set'] else None
            validate_rs = trainer.predict(setup['validate_set']) if setup['validate_set'] else None
        
            ret = self.generate_std_nn_output(input_data=input_data,
                                    train_eval_prediction=train_rs,
                                    validate_eval_prediction=validate_rs,
                                    task_type=self.task_type,
                                    threshold=0.5)
            
            logger.debug(f"train output: {ret}")

            return ret
            
        elif self.is_server():
            trainer.train()

    def predict(self, input_data: NNInput = None) -> Union[NNOutput, None]:

        setup = self.setup(input_data, stage='predict')
        test_set = setup['test_set']
        trainer = setup['trainer']
        pred_rs = trainer.predict(test_set)
        ret = self.generate_std_nn_output(input_data=input_data, test_eval_prediction=pred_rs, task_type=self.task_type, threshold=0.5)
        return ret


    
