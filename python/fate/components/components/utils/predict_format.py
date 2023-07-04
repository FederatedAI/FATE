import numpy as np
from fate.arch.dataframe._dataframe import DataFrame
import pandas as pd
from pandas import DataFrame as pd_DataFrame
from typing import Union
from fate.components.components.utils import consts
from fate.arch.dataframe import PandasReader
from fate.arch.dataframe.manager.schema_manager import Schema
import json

# variable:
LABEL = "label"
PREDICT_LABEL = "predict_result"
PREDICT_SCORE = "predict_score"
PREDICT_DETAIL = "predict_detail"
TYPE = "type"



def predict_detail_dict_to_str(result_dict):
    return "\"" + json.dumps(result_dict).replace("\"", "\'") + "\""


def predict_detail_str_to_dict(result_dict_str):
    return json.loads(json.loads(result_dict_str).replace("\'", "\""))


def get_output_pd_df(pred_table: pd_DataFrame, label: Union[pd_DataFrame, pd.Series, np.ndarray], match_id_name, sample_id_name, dataset_type=consts.TRAIN_SET, 
                     task_type=consts.BINARY, classes=None, threshold=0.5):
    
    df = pred_table

    if match_id_name not in pred_table.columns:
        raise ValueError(f"match_id_column {match_id_name} not in predict_table whose columns are {pred_table.columns}")
    
    if sample_id_name not in pred_table.columns:
        raise ValueError(f"sample_id_column {sample_id_name} not in predict_table whose columns are {pred_table.columns}")

    if PREDICT_SCORE not in pred_table.columns:
        raise ValueError(f"predict_score not in predict_table whose columns are {pred_table.columns}")
    
    pred_rs_df = pd.DataFrame()
    pred_rs_df[match_id_name] = df[match_id_name]
    pred_rs_df[sample_id_name] = df[sample_id_name]
    pred_rs_df[PREDICT_SCORE] = df[PREDICT_SCORE]
    if task_type == consts.BINARY:
        if classes is None:
            raise ValueError("classes must be specified positive and negative when task_type is binary, example: [0, 1] as negative, positive")
        class_neg, class_pos = classes[0], classes[1]
        pred_rs_df[PREDICT_SCORE] = pred_rs_df[PREDICT_SCORE].apply(lambda x: x[0])
        pred_rs_df[PREDICT_LABEL] = df[PREDICT_SCORE].apply(lambda x: class_pos if x[0] >= threshold else class_neg)
        pred_rs_df[PREDICT_DETAIL] = df[PREDICT_SCORE].apply(lambda x: predict_detail_dict_to_str({class_pos: x[0], class_neg: 1 - x[0]}))
    elif task_type == consts.MULTI:
        if classes is None:
            raise ValueError("classes must be specified when task_type is multi")
        pred_rs_df[PREDICT_LABEL] = df[PREDICT_SCORE].apply(lambda x: classes[x.index(max(x))])
        pred_rs_df[PREDICT_DETAIL] = df[PREDICT_SCORE].apply(lambda x: predict_detail_dict_to_str({classes[i]: x[i] for i in range(len(x))}))
    elif task_type == consts.REGRESSION:
        pred_rs_df[PREDICT_SCORE] = pred_rs_df[PREDICT_SCORE].apply(lambda x: x[0])
        pred_rs_df[PREDICT_LABEL] = pred_rs_df[PREDICT_SCORE]
        pred_rs_df[PREDICT_DETAIL] = df[PREDICT_SCORE].apply(lambda x: predict_detail_dict_to_str({"label": x[0]}))
    else:
        raise ValueError(f"task_type {task_type} is not supported")
    
    pred_rs_df[TYPE] = dataset_type
    pred_rs_df[LABEL] = label

    return pred_rs_df


def predict_score_to_output(ctx, pred_table:DataFrame, train_data: DataFrame, dataset_type=consts.TRAIN_SET, 
                            task_type=consts.BINARY, classes=None, threshold=0.5) -> DataFrame:

    assert dataset_type in [consts.TRAIN_SET, consts.TEST_SET, consts.VALIDATION_SET], f"dataset_type {dataset_type} is not supported"

    if isinstance(pred_table, DataFrame):
        df: pd_DataFrame = pred_table.as_pd_df()
    else:
        raise TypeError(f"predict_table type {type(pred_table)} is not supported")
    
    schema = train_data.schema
    label_name = schema.label_name
    label_df = train_data[label_name].as_pd_df()
    label = label_df[label_name]
    match_id_name = schema.match_id_name
    sample_id_name = schema.sample_id_name
    pred_rs_df = get_output_pd_df(df, label, match_id_name, sample_id_name, dataset_type, task_type, classes, threshold)
    reader = PandasReader(sample_id_name=sample_id_name, match_id_name=match_id_name, label_name=LABEL, dtype="object")
    output_df = reader.to_frame(ctx, pred_rs_df)

    return output_df
