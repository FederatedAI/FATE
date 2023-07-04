import numpy as np
import torch 
import pandas as pd
from typing import Union, Optional
from fate.components.core import Role
from fate.arch import Context
from typing import Optional, Callable, Tuple
from transformers.trainer_utils import PredictionOutput
import numpy as np
from fate.arch.dataframe._dataframe import DataFrame
from fate.arch.dataframe.manager.schema_manager import Schema
from fate.components.components.utils import consts
from fate.components.components.utils.predict_format import get_output_pd_df, LABEL, PREDICT_SCORE
import logging


logger = logging.getLogger(__name__)


def _convert_to_numpy_array(data: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return np.array(data)


class SampleIDs:

    def __init__(self, sample_id=None, match_id=None, sample_id_name='sample_id', match_id_name='id') -> None:
        self.sample_id = sample_id
        self.match_id = match_id
        self.sample_id_name = sample_id_name
        self.match_id_name = match_id_name

    def maybe_generate_ids(self, sample_num: int) -> None:
        if self.sample_id is None:
            self.sample_id = np.arange(0, sample_num)
        if self.match_id is None:
            self.match_id = np.arange(0, sample_num)

    def get_id_df(self) -> pd.DataFrame:
        return pd.DataFrame({self.sample_id_name: self.sample_id, self.match_id_name: self.match_id})

    def __repr__(self) -> str:
        return f"{self.sample_id_name}: {self.sample_id} \n {self.match_id_name}: {self.match_id}"
    

class NNInput:
    """
    Class to encapsulate input data for NN Runner.
    
    Parameters:
        train_data (Union[pd.DataFrame, str]): The training data as a pandas DataFrame or the file path to it.
        validate_data (Union[pd.DataFrame, str]): The validation data as a pandas DataFrame or the file path to it.
        test_data (Union[pd.DataFrame, str]): The testing data as a pandas DataFrame or the file path to it.
        saved_model_path (str): The path of a saved model.
        fate_save_path (str): The path for you to save your trained model in current task.
    """
    
    def __init__(self, train_data: Union[pd.DataFrame, str, DataFrame] = None,
                       validate_data: Union[pd.DataFrame, str, DataFrame] = None,
                       test_data: Union[pd.DataFrame, str, DataFrame] = None,
                       saved_model_path: str = None,
                       fate_save_path: str = None,
                       ) -> None:
        
        self.schema = None
        self.train_ids = None
        self.validate_ids = None
        self.test_ids = None
        if isinstance(train_data, DataFrame):
            self.train_data, self.train_ids, self.schema = self._extract_fate_df(train_data)
        else:
            self.train_data = train_data
            self.train_ids = SampleIDs()

        if isinstance(validate_data, DataFrame):
            self.validate_data, self.validate_ids, _ = self._extract_fate_df(validate_data)
        else:
            self.validate_data = validate_data
            self.validate_ids = SampleIDs()

        if isinstance(test_data, DataFrame):
            self.test_data, self.test_ids, self.schema = self._extract_fate_df(test_data)
        else:
            self.test_data = test_data
            self.test_ids = SampleIDs()

        self.saved_model_path = saved_model_path
        self.fate_save_path = fate_save_path

    def _extract_fate_df(self, df: DataFrame):
        schema = df.schema
        pd_df = df.as_pd_df()
        sample_id = schema.sample_id_name
        match_id = schema.match_id_name
        ids = SampleIDs(sample_id=pd_df[sample_id].to_numpy(), match_id=pd_df[match_id].to_numpy(), 
                        sample_id_name=sample_id, match_id_name=match_id)
        features = pd_df.drop(columns=[sample_id, match_id])
        return features, ids, schema

    def get(self, key: str) -> Union[pd.DataFrame, str]:
        return getattr(self, key)
    
    def get_train_data(self) -> Union[pd.DataFrame, str]:
        return self.train_data
    
    def get_validate_data(self) -> Union[pd.DataFrame, str]:
        return self.validate_data
    
    def get_test_data(self) -> Union[pd.DataFrame, str]:
        return self.test_data
    
    def get_saved_model_path(self) -> str:
        return self.saved_model_path
    
    def get_fate_save_path(self) -> str:
        return self.fate_save_path

    def get_train_ids(self) -> SampleIDs:
        return self.train_ids
    
    def get_validate_ids(self) -> SampleIDs:
        return self.validate_ids
    
    def get_test_ids(self) -> SampleIDs:
        return self.test_ids

    def get_schema(self) -> Schema:
        return self.schema

    def __getitem__(self, key: str):
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"NNInput(\ntrain_data={self.train_data},\nvalidate_data={self.validate_data}, \
        \ntest_data={self.test_data},\nmodel_path={self.saved_model_path},\nfate_save_path={self.fate_save_path}\n)"



class NNOutput:
    
    def __init__(self, 
                 train_result: Optional[pd.DataFrame] = None,
                 validate_result: Optional[pd.DataFrame] = None,
                 test_result: Optional[pd.DataFrame] = None,
                 sample_id_name="sample_id",
                 match_id_name="id",
                 ) -> None:

        assert isinstance(train_result, pd.DataFrame) or train_result is None
        assert isinstance(validate_result, pd.DataFrame) or validate_result is None
        assert isinstance(test_result, pd.DataFrame) or test_result is None
        self.sample_id_name = sample_id_name
        self.match_id_name = match_id_name

        self.train_result = self._check_ids(train_result)
        self.validate_result = self._check_ids(validate_result)
        self.test_result = self._check_ids(test_result)
    
    def _check_ids(self, dataframe: pd.DataFrame):
        if dataframe is None:
            return None
        if self.sample_id_name in dataframe.columns and self.match_id_name in dataframe.columns:
            return dataframe
        id_ = SampleIDs(sample_id_name=self.sample_id_name, match_id_name=self.match_id_name)
        id_.maybe_generate_ids(len(dataframe))
        id_df = id_.get_id_df()
        if self.sample_id_name not in dataframe.columns:
            # concat id_df and dataframe
            dataframe = pd.concat([id_df[[self.sample_id_name]], dataframe], axis=1)
        if self.match_id_name not in dataframe.columns:
            dataframe = pd.concat([id_df[[self.match_id_name]], dataframe], axis=1)
        return dataframe

    def __repr__(self) -> str:
        return f"NNOutput(train_result=\n{self.train_result}\n, validate_result=\n{self.validate_result}\n, test_result=\n{self.test_result}\n)"


def task_type_infer(predict_result, true_label):
    
    pred_shape = predict_result.shape

    if true_label.max() == 1.0 and true_label.min() == 0.0:
        return consts.BINARY

    if (len(pred_shape) > 1) and (pred_shape[1] > 1):
        if np.isclose(
            predict_result.sum(
                axis=1), np.array(
                [1.0])).all():
            return consts.MULTI
        else:
            return None
    elif (len(pred_shape) == 1) or (pred_shape[1] == 1):
        return consts.REGRESSION

    return None


def get_formatted_output_df(predict_rs: PredictionOutput, id_: SampleIDs, dataset_type, task_type=None, 
                            classes=None, threshold=0.5):
    
    logger.info("Start to format output dataframe {}".format(type(predict_rs)))
    if isinstance(predict_rs, PredictionOutput):
        predict_score = predict_rs.predictions
        if hasattr(predict_rs, 'label_ids'):
            label = predict_rs.label_ids
        else:
            raise ValueError("predict_rs should be PredictionOutput and label ids should be included in it, but got {}".format(predict_rs))

        predict_score = _convert_to_numpy_array(predict_score)
        label = _convert_to_numpy_array(label)
        df = pd.DataFrame()
        df[PREDICT_SCORE] = predict_score.tolist()
        id_.maybe_generate_ids(len(df))
        id_df = id_.get_id_df()
        df = pd.concat([id_df, df], axis=1)
        if task_type is None:
            task_type = task_type_infer(predict_score, label)
        if task_type == consts.BINARY or task_type == consts.MULTI:
            if task_type == consts.BINARY:
                classes = [0, 1]
            else:
                classes = np.unique(label).tolist()
        if task_type is not None:
            return get_output_pd_df(df, label, id_.match_id_name, id_.sample_id_name, dataset_type, task_type, classes, threshold)
        else:
            df[LABEL] = label
            return df
    else:
        raise ValueError("predict_rs should be PredictionOutput")



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
                           train_eval_prediction: Optional[PredictionOutput] = None,
                           validate_eval_prediction: Optional[PredictionOutput] = None,
                           test_eval_prediction: Optional[PredictionOutput] = None,
                           task_type: str = consts.BINARY,
                           threshold: float = 0.5) -> NNOutput:

        results = {}
        match_id_name, sample_id_name = 'id', 'sample_id'
        if train_eval_prediction is not None:
            ids = input_data.get_train_ids()
            match_id_name, sample_id_name = ids.match_id_name, ids.sample_id_name
        elif test_eval_prediction is not None:
            ids = input_data.get_test_ids()
            match_id_name, sample_id_name = ids.match_id_name, ids.sample_id_name
        else:
            raise ValueError('You need to provide either train_eval_prediction or test_eval_prediction')
        
        if train_eval_prediction is not None:
            results["train"] = get_formatted_output_df(train_eval_prediction, input_data.get_train_ids(), consts.TRAIN_SET, task_type, threshold=threshold)
        if validate_eval_prediction is not None:
            results["validate"] = get_formatted_output_df(validate_eval_prediction, input_data.get_validate_ids(), consts.VALIDATE_SET, task_type, threshold=threshold)
        if test_eval_prediction is not None:
            results["test"] = get_formatted_output_df(test_eval_prediction, input_data.get_test_ids(), consts.TEST_SET, task_type, threshold=threshold)

        return NNOutput(train_result=results.get("train"), validate_result=results.get("validate"), test_result=results.get("test"), 
                        match_id_name=match_id_name, sample_id_name=sample_id_name)
    
    def get_fateboard_tracker(self):
        pass

    def train(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        pass

    def predict(self, input_data: NNInput = None) -> Optional[Union[NNOutput, None]]:
        pass

