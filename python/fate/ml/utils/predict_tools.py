import json
from typing import Literal

import numpy as np
import pandas as pd

from fate.arch.dataframe import DataFrame
from fate.arch.dataframe import PandasReader

# DATA SET COLUMNS
TRAIN_SET = 'train_set'
VALIDATE_SET = 'validate_set'
TEST_SET = 'test_set'

# PREDICT RESULT COLUMNS
PREDICT_RESULT = "predict_result"
PREDICT_SCORE = "predict_score"
PREDICT_DETAIL = "predict_detail"
LABEL = "label"

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


def to_dist_df(ctx, sample_id_name, match_id_name, result_df: pd.DataFrame):

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


def compute_predict_details(dataframe: DataFrame, task_type: Literal['binary', 'multi', 'regression'], classes: list = None, threshold=0.5):

    assert task_type in [BINARY, MULTI, REGRESSION,
                         OTHER], 'task_type must be one of {} as a std task, but got {}'.format(
        [BINARY, MULTI, REGRESSION, OTHER], task_type)

    assert threshold >= 0 and threshold <= 1, 'threshold must be float in [0, 1], but got {}'.format(threshold)

    if not isinstance(dataframe, DataFrame):
        raise ValueError('dataframe must be a fate DataFrame, but got {}'.format(type(dataframe)))
    if dataframe.schema.label_name is not None and dataframe.schema.label_name != LABEL:
        dataframe.rename(label_name=LABEL)
    assert PREDICT_SCORE in dataframe.schema.columns, 'column {} is not found in input dataframe'.format(PREDICT_SCORE)

    if task_type == BINARY and task_type == MULTI:
        if classes is None or (not isinstance(classes, list) and len(classes) < 2):
            raise ValueError('task_type is binary or multi, but classes is None, or classes length is less than 2')

    if task_type == BINARY:
        if len(classes) == 2:
            neg_class, pos_class = classes[0], classes[1]
            dataframe[[PREDICT_RESULT, PREDICT_DETAIL]] = dataframe.apply_row( \
                lambda v: [int(v[PREDICT_SCORE] > threshold),
                           predict_detail_dict_to_str({neg_class: 1 - float(v[PREDICT_SCORE]),
                                                       pos_class: float(v[PREDICT_SCORE])})],
                enable_type_align_checking=False)
        else:
            raise ValueError(
                'task_type is binary, but classes length is not 2: {}'.format(classes))
        
    elif task_type == REGRESSION:
        dataframe[[PREDICT_RESULT, PREDICT_DETAIL]] = dataframe.apply_row( \
            lambda v: [v[PREDICT_SCORE], predict_detail_dict_to_str({PREDICT_SCORE: float(v[PREDICT_SCORE])})],
            enable_type_align_checking=False)

    elif task_type == MULTI:

        def handle_multi(v: pd.Series):
            predict_result = np.argmax(v[PREDICT_SCORE])
            assert len(v[PREDICT_SCORE]) == len(classes), 'predict score length is not equal to classes length,\
                predict score is {}, but classes are {}, please check the data you provided'.format(v[PREDICT_SCORE], classes)
            predict_details = {classes[j]: float(v[PREDICT_SCORE][j]) for j in range(len(classes))}
            return [predict_result, predict_detail_dict_to_str(predict_details)]

        dataframe[[PREDICT_RESULT, PREDICT_DETAIL]] = dataframe.apply_row(handle_multi, enable_type_align_checking=False)
        predict_score = dataframe[PREDICT_SCORE].apply_row(lambda v: max(v[PREDICT_SCORE]))
        dataframe[PREDICT_SCORE] = predict_score

    return dataframe


def array_to_predict_df(
        ctx,
        task_type: Literal['binary', 'multi', 'regression'],
        pred: np.ndarray,
        match_ids: np.ndarray,
        sample_ids: np.ndarray,
        match_id_name: str,
        sample_id_name: str,
        label: np.array = None,
        threshold=0.5,
        classes: list = None):

    df = pd.DataFrame()
    if len(pred.shape) == 1:
        df[PREDICT_SCORE] = np.array(pred)
    elif len(pred.shape) == 2:
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

    df[sample_id_name] = sample_ids.flatten()
    df[match_id_name] = match_ids.flatten()
    fate_df = to_dist_df(ctx, sample_id_name, match_id_name, df)
    predict_result = compute_predict_details(fate_df, task_type, classes, threshold)

    return predict_result
