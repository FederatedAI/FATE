import pandas as pd
from fate.arch.dataframe import PandasReader
import numpy as np


TRAIN_SET = 'train_set'
VALIDATE_SET = 'validate_set'
TEST_SET = 'test_set'
LABEL = "label"
PREDICT_LABEL = "predict_result"
PREDICT_SCORE = "predict_score"
PREDICT_DETAIL = "predict_detail"
TYPE = "type"

# TASK TYPE
BINARY = 'binary'
MULTI = 'multi'
REGRESSION = 'regression'
OTHER = 'other'


def add_ids(df: pd.DataFrame, match_id: pd.DataFrame, sample_id:pd.DataFrame):
    df = pd.concat([df, match_id, sample_id], axis=1)
    return df


def add_dataset_type(df: pd.DataFrame, ds_type):
    
    assert ds_type in [TRAIN_SET, VALIDATE_SET, TEST_SET], 'ds_type must be one of {}, but got {}'.format([TRAIN_SET, VALIDATE_SET, TEST_SET], ds_type)
    df[TYPE] = ds_type
    return df


def to_fate_df(ctx, sample_id_name, match_id_name, result_df):

    if LABEL in result_df:
        reader = PandasReader(sample_id_name=sample_id_name, match_id_name=match_id_name, label_name=LABEL, dtype="object")
    else:
        reader = PandasReader(sample_id_name=sample_id_name, match_id_name=match_id_name, dtype="object")
    data = reader.to_frame(ctx, result_df)
    return data


def std_output_df(task_type, pred: np.array, label: np.array=None, threshold=0.5, classes: list = None):

    assert task_type in [BINARY, MULTI, REGRESSION, OTHER], 'task_type must be one of {} as a std task, but got {}'.format([BINARY, MULTI, REGRESSION, OTHER], task_type)
    
    if task_type == BINARY:
        if len(classes) == 2:
            predict_score = pred
            predict_result = (predict_score > threshold).astype(int)
            predict_details = [{classes[0]: 1 - float(predict_score[i]), classes[1]: float(predict_score[i])} for i in range(len(predict_score))]
        else:
            raise ValueError('task_type is binary, but classes length is not 2: {}'.format(classes))

    elif task_type == MULTI: 
        if len(classes) > 2:
            predict_score = pred.max(axis=1)
            predict_result = np.argmax(pred, axis=1)
            predict_details = [{classes[j]: float(pred[i][j]) for j in range(len(classes))} for i in range(len(pred))]
        else:
            raise ValueError('task_type is multi, but classes length is not greater than 2: {}'.format(classes))
    
    elif task_type == REGRESSION:
        # regression task
        predict_score = pred
        predict_result = pred
        predict_details = [{LABEL: float(pred[i])} for i in range(len(pred))]

    if label is None:
        df = pd.DataFrame({
            PREDICT_SCORE: predict_score.flatten(),
            PREDICT_LABEL: predict_result.flatten(),
            PREDICT_DETAIL: predict_details
        })
    else:
        df = pd.DataFrame({
            PREDICT_SCORE: predict_score.flatten(),
            PREDICT_LABEL: predict_result.flatten(),
            LABEL: label.flatten(),
            PREDICT_DETAIL: predict_details
        })
    
    return df