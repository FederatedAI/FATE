import pandas as pd
from typing import Union, Optional
from fate.components import Role
from fate.interface import Context



class NNInput:
    """
    Class to encapsulate input data for NN Runner.
    
    Parameters:
        train_data (Union[pd.DataFrame, str]): The training data as a pandas DataFrame or the file path to it.
        validate_data (Union[pd.DataFrame, str]): The validation data as a pandas DataFrame or the file path to it.
        test_data (Union[pd.DataFrame, str]): The testing data as a pandas DataFrame or the file path to it.
        model_path (str): The path of a saved model.
        fate_save_path (str): The path for you to save your trained model in current task.
    """
    
    def __init__(self, train_data: Union[pd.DataFrame, str] = None,
                       validate_data: Union[pd.DataFrame, str] = None,
                       test_data: Union[pd.DataFrame, str] = None,
                       model_path: str = None,
                       fate_save_path: str = None
                       ) -> None:
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.model_path = model_path
        self.fate_save_path = fate_save_path

    def get(self, key: str) -> Union[pd.DataFrame, str]:
        return getattr(self, key)
    
    def get_train_data(self) -> Union[pd.DataFrame, str]:
        return self.train_data
    
    def get_validate_data(self) -> Union[pd.DataFrame, str]:
        return self.validate_data
    
    def get_test_data(self) -> Union[pd.DataFrame, str]:
        return self.test_data
    
    def get_model_path(self) -> str:
        return self.model_path
    
    def get_fate_save_path(self) -> str:
        return self.fate_save_path
    
    def __repr__(self) -> str:
        return f"NNInput(train_data={self.train_data}, validate_data={self.validate_data}, \
                test_data={self.test_data}, model_path={self.model_path}, fate_save_path={self.fate_save_path})"


class NNOutput:
    
    def __init__(self, data=None) -> None:
        self.data = data


class NNRunner(object):

    def __init__(self) -> None:
        
        self._role = None
        self._party_id = None
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

    def get_fateboard_tracker(self):
        pass

    def train(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        pass

    def predict(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        pass

