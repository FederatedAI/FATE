import json
import pandas as pd
from fate.arch.dataframe import PandasReader
import numpy as np
from typing import Union
from fate.arch.dataframe import DataFrame


TRAIN_SET = 'train_set'
VALIDATE_SET = 'validate_set'
TEST_SET = 'test_set'
LABEL = "label"
PREDICT_RESULT = "predict_result"
PREDICT_SCORE = "predict_score"
PREDICT_DETAIL = "predict_detail"

# TASK TYPE
BINARY = 'binary'
MULTI = 'multi'
REGRESSION = 'regression'
OTHER = 'other'


def predict_detail_dict_to_str(result_dict):
    return "\"" + json.dumps(result_dict).replace("\"", "\'") + "\""


def add_ids(df: pd.DataFrame, match_id: pd.DataFrame, sample_id: pd.DataFrame):
    df = pd.concat([df, match_id, sample_id], axis=1)
    return df


def to_fate_df(ctx, sample_id_name, match_id_name, result_df):

    if LABEL in result_df:
        reader = PandasReader(
            sample_id_name=sample_id_name,
            match_id_name=match_id_name,
            label_name=LABEL,
            dtype="object")
    else:
        reader = PandasReader(
            sample_id_name=sample_id_name,
            match_id_name=match_id_name,
            dtype="object")
    data = reader.to_frame(ctx, result_df)
    return data


def compute_predict_details(
        dataframe: Union[pd.DataFrame, DataFrame], task_type, classes: list = None, threshold=0.5):

    assert task_type in [BINARY, MULTI, REGRESSION, OTHER], 'task_type must be one of {} as a std task, but got {}'.format(
        [BINARY, MULTI, REGRESSION, OTHER], task_type)
    if isinstance(dataframe, DataFrame):
        df = dataframe.as_pd_df()
    else:
        df = dataframe

    pred = df[PREDICT_SCORE].values if PREDICT_SCORE in df else None
    if pred is None:
        raise ValueError('pred score is not found in input dataframe')

    if task_type == BINARY and task_type == MULTI and classes is None:
        raise ValueError('task_type is binary or multi, but classes is None')

    if task_type == BINARY:
        if len(classes) == 2:
            predict_score = np.array(pred)
            predict_result = (predict_score > threshold).astype(int)
            predict_details = [
                {
                    classes[0]: 1 -
                    float(
                        predict_score[i]),
                    classes[1]: float(
                        predict_score[i])} for i in range(
                    len(predict_score))]
        else:
            raise ValueError(
                'task_type is binary, but classes length is not 2: {}'.format(classes))

    elif task_type == MULTI:
        if len(classes) > 2:
            predict_score = np.array([max(i) for i in pred])
            predict_result = np.array([np.argmax(i) for i in pred])
            predict_details = [predict_detail_dict_to_str({classes[j]: float(
                pred[i][j]) for j in range(len(classes))}) for i in range(len(pred))]
        else:
            raise ValueError(
                'task_type is multi, but classes length is not greater than 2: {}'.format(classes))

    elif task_type == REGRESSION:
        # regression task
        predict_score = np.array(pred)
        predict_result = np.array(pred)
        predict_details = [{LABEL: float(pred[i])} for i in range(len(pred))]

    df[PREDICT_RESULT] = predict_result
    df[PREDICT_DETAIL] = predict_details
    if task_type == MULTI:
        df[PREDICT_SCORE] = predict_score

    return df


def std_output_df(
        task_type,
        pred: np.array,
        label: np.array = None,
        threshold=0.5,
        classes: list = None):

    df = pd.DataFrame()
    if len(pred.shape) == 1:
        df[PREDICT_SCORE] = np.array(pred)
    if len(pred.shape) == 2:
        if pred.shape[1] == 1:
            df[PREDICT_SCORE] = np.array(pred).flatten()
        else:
            df[PREDICT_SCORE] = np.array(pred).tolist()
    else:
        raise ValueError(
            'This is not a FATE std task, pred scores shape are {}'.format(
                pred.shape))

    if label is not None:
        if len(label.shape) == 1:
            label = label.flatten()
        elif len(label.shape) == 2 and label.shape[1] == 1:
            label = label.flatten()
        else:
            label = label.tolist()
        df[LABEL] = label

    df = compute_predict_details(df, task_type, classes, threshold)

    return df
