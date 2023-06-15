import numpy as np
import torch 
import pandas as pd
from typing import Union, Optional
from fate.components import Role
from fate.interface import Context
from typing import Optional, Callable, Tuple
from transformers import EvalPrediction
import numpy as np


def get_output_dataframe(predict_prob: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor],
                         pred_label: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor],
                         label: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor],
                         sample_id: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]] = None,
                         match_id: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]] = None,
                         match_id_name: str = 'match_id'
                         ) -> pd.DataFrame:
    
    # Convert inputs to numpy arrays
    predict_prob = _convert_to_numpy_array(predict_prob)
    pred_label = _convert_to_numpy_array(pred_label)
    label = _convert_to_numpy_array(label)
    
    # Generate id arrays if both sample_id and match_id are None

    if sample_id is not None:
        sample_id = _convert_to_numpy_array(sample_id)
    else:
        sample_id = np.arange(0, len(label))

    if match_id is not None:
        match_id = _convert_to_numpy_array(match_id)
    else:
        match_id = np.arange(0, len(label))

    # Check if the lengths along axis=0 are consistent
    _check_consistency(predict_prob, pred_label, label, sample_id, match_id)
    
    columns = ['sample_id', match_id_name, 'predict_prob', 'predict_label', 'label']
    # Concatenate into a DataFrame
    return _concatenate_to_dataframe(sample_id, match_id, predict_prob, pred_label, label, columns)


def _convert_to_numpy_array(data: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return np.array(data)


def _check_consistency(predict_prob: np.ndarray, pred_label: np.ndarray, label: np.ndarray, 
                       sample_id: Optional[np.ndarray], match_id: Optional[np.ndarray]) -> None:
    
    arrays = [predict_prob, pred_label, label, sample_id, match_id]
    
    lengths = [arr.shape[0] for arr in arrays if arr is not None]
    
    if not all(length == lengths[0] for length in lengths):
        raise ValueError(f"Inconsistent lengths: predict_prob({lengths[0]}), pred_label({lengths[1]}), label({lengths[2]})"
                         + (f", sample_id({lengths[3]})" if sample_id is not None else "")
                         + (f", match_id({lengths[4]})" if match_id is not None else ""))


def _concatenate_to_dataframe(sample_id: Optional[np.ndarray],
                              match_id: Optional[np.ndarray],
                              predict_prob: Optional[np.ndarray],
                              pred_label: Optional[np.ndarray],
                              label: Optional[np.ndarray],
                              columns
                              ) -> pd.DataFrame:
    
    data = {}
    arrays = [sample_id, match_id, predict_prob.tolist(), pred_label, label]
    
    # Loop over each column
    for col, array in zip(columns, arrays):
        if array is not None:
            # Except for predict_prob, reshape other columns to (sample_num, -1)
            if col != 'predict_prob':
                array = array.flatten()
            data[col] = array
            
    return pd.DataFrame(data)


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
                       fate_save_path: str = None,
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

    def __getitem__(self, key: str):
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"NNInput(\ntrain_data={self.train_data},\nvalidate_data={self.validate_data}, \
        \ntest_data={self.test_data},\nmodel_path={self.model_path},\nfate_save_path={self.fate_save_path}\n)"



class NNOutput:
    
    def __init__(self, train_result: Optional[pd.DataFrame] = None,
                 validate_result: Optional[pd.DataFrame] = None,
                 test_result: Optional[pd.DataFrame] = None) -> None:
        
        # Columns that are expected in the DataFrames
        expected_columns = ['sample_id', 'predict_prob', 'predict_label', 'label']

        self.train_result = None
        self.validate_result = None
        self.test_result = None
        
        # Check if the DataFrames have all the required columns
        if train_result is not None:
            self._check_columns(train_result, expected_columns)
            self.train_result = train_result
            
        if validate_result is not None:
            self._check_columns(validate_result, expected_columns)
            self.validate_result = validate_result
            
        if test_result is not None:
            self._check_columns(test_result, expected_columns)
            self.test_result = test_result
    
    def _check_columns(self, dataframe: pd.DataFrame, expected_columns) -> None:
        """
        Check if the DataFrame has all the expected columns.
        """
        missing_columns = set(expected_columns) - set(dataframe.columns)
        if len(missing_columns) > 0:
            raise ValueError(f"Missing columns: {missing_columns}")

    def __repr__(self) -> str:
        return f"NNOutput(train_result=\n{self.train_result}\n, validate_result=\n{self.validate_result}\n, test_result=\n{self.test_result}\n)"


def extract_dataset_info(input_data_method: Callable[[], Optional[pd.DataFrame]]) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    data = input_data_method()
    if isinstance(data, pd.DataFrame):
        if hasattr(data, 'match_id_name'):
            match_id_name = data.match_id_name
        else:
            match_id_name = 'match_id'
        return data.get('sample_id'), data.get(match_id_name), match_id_name
    return None, None, None



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
    
    @staticmethod
    def generate_std_nn_output(input_data: NNInput,
                           train_eval_prediction: Optional[EvalPrediction] = None,
                           validate_eval_prediction: Optional[EvalPrediction] = None,
                           test_eval_prediction: Optional[EvalPrediction] = None,
                           task_type: str = "classification",
                           threshold: float = 0.5) -> NNOutput:

        def compute_pred_label(predictions, task_type, threshold):
            if task_type == "classification":
                # Check if binary classification
                if predictions.shape[-1] == 1:
                    return (predictions >= threshold).astype(int)
                # Otherwise assume multi-class classification
                return np.argmax(predictions, axis=-1)
            # If not classification, return argmax as default
            return np.argmax(predictions, axis=-1)
        
        def generate_result(eval_prediction, data_getter, task_type, threshold):
            if eval_prediction is not None:
                sample_id, match_id, match_id_name = extract_dataset_info(data_getter)
                pred_label = compute_pred_label(eval_prediction.predictions, task_type, threshold)
                df = get_output_dataframe(predict_prob=eval_prediction.predictions,
                                        pred_label=pred_label,
                                        label=eval_prediction.label_ids,
                                        sample_id=sample_id,
                                        match_id=match_id,
                                        match_id_name=match_id_name
                                        )
                df.match_id_name = match_id_name
                return df

            return None

        # Using a dictionary to iterate through the datasets
        eval_predictions = {
            "train": train_eval_prediction,
            "validate": validate_eval_prediction,
            "test": test_eval_prediction
        }
        
        data_getters = {
            "train": input_data.get_train_data,
            "validate": input_data.get_validate_data,
            "test": input_data.get_test_data
        }

        # Generate results
        results = {}
        for key in eval_predictions.keys():
            results[key] = generate_result(eval_predictions[key], data_getters[key], task_type, threshold)

        # Create and return an NNOutput object
        return NNOutput(train_result=results["train"], validate_result=results["validate"], test_result=results["test"])
    
    def get_fateboard_tracker(self):
        pass

    def train(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        pass

    def predict(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        pass

