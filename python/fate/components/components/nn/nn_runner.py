#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import torch
import torch as t
import logging
import numpy as np
from torch import optim, nn
import pandas as pd
from typing import Literal, Type, Callable
from fate.ml.nn.trainer.trainer_base import TrainingArguments
from fate.components.core import Role
from fate.arch import Context
from typing import Optional, Union
from transformers.trainer_utils import PredictionOutput
from fate.arch.dataframe._dataframe import DataFrame
from fate.components.components.utils import consts
from torch.optim.lr_scheduler import _LRScheduler
from fate.ml.utils.predict_tools import to_dist_df, array_to_predict_df
from fate.ml.utils.predict_tools import BINARY, MULTI, REGRESSION, OTHER, CAUSAL_LM, LABEL, PREDICT_SCORE
from fate.components.components.nn.loader import Loader


logger = logging.getLogger(__name__)


def convert_to_numpy_array(data: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.to_numpy()
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return np.array(data)


def task_type_infer(predict_result, true_label):
    pred_shape = predict_result.shape

    if true_label.max() == 1.0 and true_label.min() == 0.0:
        return consts.BINARY

    if (len(pred_shape) > 1) and (pred_shape[1] > 1):
        if np.isclose(predict_result.sum(axis=1), np.array([1.0])).all():
            return consts.MULTI
        else:
            return None
    elif (len(pred_shape) == 1) or (pred_shape[1] == 1):
        return consts.REGRESSION

    return None


def load_model_dict_from_path(path):
    # Ensure that the path is a string
    assert isinstance(path, str), "Path must be a string, but got {}".format(type(path))

    # Append the filename to the path
    model_path = os.path.join(path, "pytorch_model.bin")

    # Check if the file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No 'pytorch_model.bin' file found at {model_path}, no saved model found")

    # Load the state dict from the specified path
    model_dict = t.load(model_path)

    return model_dict


def dir_warning(train_args):
    if "output_dir" in train_args or "resume_from_checkpoint" in train_args:
        logger.warning(
            "The output_dir, logging_dir, and resume_from_checkpoint arguments are not supported in the "
            "DefaultRunner when running the Pipeline. These arguments will be replaced by FATE provided paths."
        )


def loader_load_from_conf(conf, return_class=False):
    if conf is None:
        return None
    if return_class:
        return Loader.from_dict(conf).load_item()
    return Loader.from_dict(conf).call_item()


def run_dataset_func(dataset, func_name):
    if hasattr(dataset, func_name):
        output = getattr(dataset, func_name)()
        if output is None:
            logger.info(
                f"dataset {type(dataset)}: {func_name} returns None, this will influence the output of predict"
            )
        return output
    else:
        logger.info(
            f"dataset {type(dataset)} not implemented {func_name}, classes set to None, this will influence the output of predict"
        )
        return None


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

    def is_guest(self) -> bool:
        return self._role.is_guest

    def is_host(self) -> bool:
        return self._role.is_host

    def set_party_id(self, party_id: int):
        assert isinstance(self._party_id, int)
        self._party_id = party_id

    def get_fateboard_tracker(self):
        pass

    def get_nn_output_dataframe(
        self,
        ctx,
        predictions: Union[np.ndarray, torch.Tensor, DataFrame, PredictionOutput],
        labels: Union[np.ndarray, torch.Tensor, DataFrame, PredictionOutput] = None,
        match_ids: Union[pd.DataFrame, np.ndarray] = None,
        sample_ids: Union[pd.DataFrame, np.ndarray] = None,
        match_id_name: str = None,
        sample_id_name: str = None,
        dataframe_format: Literal["default", "dist_df"] = "default",
        task_type: Literal["binary", "multi", "regression", "others"] = None,
        threshold: float = 0.5,
        classes: list = None,
    ) -> DataFrame:
        """
        Constructs a FATE DataFrame from predictions and labels. This Dataframe is able to flow through FATE components.

        Parameters:
            ctx (Context): The Context Instance.
            predictions (Union[np.ndarray, torch.Tensor, DataFrame, PredictionOutput]): The model's predictions, which can be numpy arrays, torch tensors, pandas DataFrames, or PredictionOutputs.
            labels (Union[np.ndarray, torch.Tensor, DataFrame, PredictionOutput]): The true labels, which can be numpy arrays, torch tensors, pandas DataFrames, or PredictionOutputs.
            match_ids (Union[pd.DataFrame, np.ndarray], optional): Match IDs, if applicable. Defaults to None. If None, will auto generate match_ids.
            sample_ids (Union[pd.DataFrame, np.ndarray], optional): Sample IDs, if applicable. Defaults to None. If None, will auto generate sample_ids.
            match_id_name (str, optional): Column name for match IDs in the resulting DataFrame. If None, Defaults to 'id'.
            sample_id_name (str, optional): Column name for sample IDs in the resulting DataFrame. If None, Defaults to 'sample_id'.
            dataframe_format (Literal['default', 'dist_df'], optional): Output format of the resulting DataFrame. If 'default', simply combines labels and predictions into a DataFrame.
                                                                         If 'dist_df', organizes output according to the FATE framework's format. Defaults to 'default'.
            task_type (Literal['binary', 'multi', 'regression', 'causal_lm', 'others'], optional):  This parameter is only needed when dataframe_format is 'dist_df'. Defaults to None.
                                                                                                    The type of machine learning task, which can be 'binary', 'multi', 'regression', 'causal', or 'others'.
            threshold (float, optional): This parameter is only needed when dataframe_format is 'dist_df' and task_type is 'binary'. Defaults to 0.5.
            classes (list, optional): This parameter is only needed when dataframe_format is 'dist_df'. List of classes.
        Returns:
            DataFrame: A DataFrame that contains the neural network's predictions and the true labels, possibly along with match IDs and sample IDs, formatted according to the specified format.
        """
        # check parameters
        assert task_type in [BINARY, MULTI, REGRESSION, CAUSAL_LM, OTHER], f"task_type {task_type} is not supported"
        assert dataframe_format in ["default", "dist_df"], f"dataframe_format {dataframe_format} is not supported"

        if match_id_name is None:
            match_id_name = "id"
        if sample_id_name is None:
            sample_id_name = "sample_id"

        if isinstance(predictions, PredictionOutput):
            predictions = predictions.predictions

        if labels is not None:
            if isinstance(labels, PredictionOutput):
                labels = labels.label_ids
            predictions = convert_to_numpy_array(predictions)
            labels = convert_to_numpy_array(labels)
            assert len(predictions) == len(
                labels
            ), f"predictions length {len(predictions)} != labels length {len(labels)}"

        # check match ids
        if match_ids is not None:
            match_ids = convert_to_numpy_array(match_ids).flatten()
        else:
            logger.info("match_ids is not provided, will auto generate match_ids")
            match_ids = np.array([i for i in range(len(predictions))]).flatten()

        # check sample ids
        if sample_ids is not None:
            sample_ids = convert_to_numpy_array(sample_ids).flatten()
        else:
            logger.info("sample_ids is not provided, will auto generate sample_ids")
            sample_ids = np.array([i for i in range(len(predictions))]).flatten()

        assert len(match_ids) == len(
            predictions
        ), f"match_ids length {len(match_ids)} != predictions length {len(predictions)}"
        assert len(sample_ids) == len(
            predictions
        ), f"sample_ids length {len(sample_ids)} != predictions length {len(predictions)}"

        # match id name and sample id name must be str
        assert isinstance(match_id_name, str), f"match_id_name must be str, but got {type(match_id_name)}"
        assert isinstance(sample_id_name, str), f"sample_id_name must be str, but got {type(sample_id_name)}"

        if dataframe_format == "default" or \
                (dataframe_format == "dist_df" and task_type == OTHER):
            df = pd.DataFrame()
            if labels is not None:
                df[LABEL] = labels.to_list()

            df[PREDICT_SCORE] = predictions.to_list()

            df[match_id_name] = match_ids.flatten()
            df[sample_id_name] = sample_ids.flatten()
            df = to_dist_df(ctx, sample_id_name, match_id_name, df)
            return df
        elif dataframe_format == "dist_df" and task_type in [BINARY, MULTI, REGRESSION]:
            df = array_to_predict_df(
                ctx,
                task_type,
                predictions,
                match_ids,
                sample_ids,
                match_id_name,
                sample_id_name,
                labels,
                threshold,
                classes,
            )
            return df

    def train(
        self,
        train_data: Optional[Union[str, DataFrame]] = None,
        validate_data: Optional[Union[str, DataFrame]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ) -> None:
        """
        Train interface.

        Parameters:
            train_data (Union[str, DataFrame]): The training data, which can be a FATE DataFrame containing the data, or a string path representing the bound data.Train data is Optional on the server side.
            validate_data (Optional[Union[str, DataFrame]]): The validation data, which can be a FATE DataFrame containing the data,  or a string path representing the bound data . This argument is optional.
            output_dir (str, optional): The path to the directory where the trained model should be saved. If this class is running in the fate pipeline, this path will provided by FATE framework.
            saved_model_path (str, optional): The path to the saved model that should be loaded before training starts.If this class is running in the fate pipeline, this path will provided by FATE framework.
        """
        pass

    def predict(
        self, test_data: Optional[Union[str, DataFrame]] = None, output_dir: str = None, saved_model_path: str = None
    ) -> DataFrame:
        """
        Predict interface.

        Parameters:
            test_data (Union[str, DataFrame]): The data to predict, which can be a FATE DataFrame containing the data, or a string path representing the bound data.Test data is Optional on the server side.
            output_dir (str, optional): The path to the directory where the trained model should be saved. If this class is running in the fate pipeline, this path will provided by FATE framework.
            saved_model_path (str, optional): The path to the saved model that should be loaded before training starts.If this class is running in the fate pipeline, this path will provided by FATE framework.
        """
