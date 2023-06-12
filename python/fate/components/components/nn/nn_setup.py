import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import _LRScheduler
from typing import Union, Type, Callable, Optional, List, Tuple
from fate.components import Role
from fate.interface import Context
from fate.ml.nn.trainer.trainer_base import FedArguments, TrainingArguments, FedTrainerClient, FedTrainerServer


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


class ComponentInputData:
    """
    Class to encapsulate input data for a component.
    
    Parameters:
        train_data (Union[pd.DataFrame, str]): The training data as a pandas DataFrame or the file path to it.
        validate_data (Union[pd.DataFrame, str]): The validation data as a pandas DataFrame or the file path to it.
        test_data (Union[pd.DataFrame, str]): The testing data as a pandas DataFrame or the file path to it.
    """
    
    def __init__(self, train_data: Union[pd.DataFrame, str] = None,
                  validate_data: Union[pd.DataFrame, str] = None,
                    test_data: Union[pd.DataFrame, str] = None) -> None:
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data

    def get(self, key: str) -> Union[pd.DataFrame, str]:
        return getattr(self, key)
    
    def get_train_data(self) -> Union[pd.DataFrame, str]:
        return self.train_data
    
    def get_validate_data(self) -> Union[pd.DataFrame, str]:
        return self.validate_data
    
    def get_test_data(self) -> Union[pd.DataFrame, str]:
        return self.test_data
    

class ComponentInputModel:

    def __init__(self, model_conf: dict) -> None:
        self.model_conf = model_conf

    def get_model_dict(self):
        return self.model_conf
    

class ComponentOutputModel:
    pass


class ComponentOutputData:
    pass


class NNRunner(object):

    def __init__(self) -> None:
        
        self._role = None
        self._party_id = None
        self._cpn_input_data = None
        self._cpn_input_model = None
        self._ctx: Context = None

    def set_context(self, context: Context):
        assert isinstance(context, Context)
        self._ctx = context

    def get_context(self) -> Context:
        return self._ctx

    def set_role(self, role: Role):
        assert isinstance(role, Role)
        self._role = role

    def is_client(self) -> bool:
        return self._role.is_guest or self._role.is_host
    
    def is_server(self) -> bool:
        return self._role.is_arbiter
    
    def set_party_id(self, party_id: int):
        assert isinstance(self._party_id, int)
        self._party_id = party_id

    def set_cpn_input_data(self, cpn_input : ComponentInputData):
        self._cpn_input_data = cpn_input

    def set_cpn_input_model(self, cpn_input):
        self._cpn_input_model = cpn_input

    def get_cpn_input_data(self) -> ComponentInputData:
        return self._cpn_input_data

    def get_cpn_input_model(self):
        return self._cpn_input_model

    def train(self) -> Optional[Union[Tuple[Data, Model], None]]:
        pass

    def predict(self):
        pass

