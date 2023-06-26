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
PREDICT_RESULT = "predict_result"
PREDICT_SCORE = PREDICT_SCORE
PREDICT_DETAIL = "predict_detail"
TYPE = "type"


def predict_detail_dict_to_str(result_dict):
    return "\"" + json.dumps(result_dict).replace("\"", "\'") + "\""


def predict_detail_str_to_dict(result_dict_str):
    return json.loads(json.loads(result_dict_str).replace("\'", "\""))


def predict_score_to_output(pred_table: Union[DataFrame, pd_DataFrame], schema: Schema, dataset_type=consts.TRAIN_SET, 
                            task_type=consts.BINARY, classes=None, threshold=0.5) -> DataFrame:

    assert dataset_type in [consts.TRAIN_SET, consts.TEST_SET, consts.VALIDATION_SET], f"dataset_type {dataset_type} is not supported"

    if isinstance(pred_table, DataFrame):
        df: pd_DataFrame = pred_table.as_pd_df()
    
    elif isinstance(pred_table, pd_DataFrame):
        df: pd_DataFrame = pred_table

    else:
        raise TypeError(f"predict_table type {type(pred_table)} is not supported")
    
    match_id_column = schema.match_id_name
    sample_id_column = schema.sample_id_name

    if match_id_column not in pred_table.columns:
        raise ValueError(f"match_id_column {match_id_column} not in predict_table whose columns are {pred_table.columns}")
    
    if sample_id_column not in pred_table.columns:
        raise ValueError(f"sample_id_column {sample_id_column} not in predict_table whose columns are {pred_table.columns}")

    if PREDICT_SCORE not in pred_table.columns:
        raise ValueError(f"predict_score not in predict_table whose columns are {pred_table.columns}")
    
    pred_rs_df = pd.DataFrame()
    pred_rs_df[match_id_column] = df[match_id_column]
    pred_rs_df[sample_id_column] = df[sample_id_column]
    pred_rs_df[PREDICT_SCORE] = df[PREDICT_SCORE]
    if task_type == consts.BINARY:
        if classes is None:
            raise ValueError("classes must be specified positive and negative when task_type is binary, example: [0, 1] as negative, positive")
        class_neg, class_pos = classes[0], classes[1]
        pred_rs_df[PREDICT_RESULT] = df[PREDICT_SCORE].apply(lambda x: class_pos if x >= threshold else class_neg)
        pred_rs_df[PREDICT_DETAIL] = df[PREDICT_SCORE].apply(lambda x: predict_detail_dict_to_str({class_pos: x, class_neg: 1 - x}))
    elif task_type == consts.MULTY:
        if classes is None:
            raise ValueError("classes must be specified when task_type is multy")
        pred_rs_df[PREDICT_RESULT] = df[PREDICT_SCORE].apply(lambda x: classes[x.index(max(x))])
        pred_rs_df[PREDICT_DETAIL] = df[PREDICT_SCORE].apply(lambda x: predict_detail_dict_to_str({classes[i]: x[i] for i in range(len(x))}))
    elif task_type == consts.REGRESSION:
        pred_rs_df[PREDICT_RESULT] = df[PREDICT_SCORE]
        pred_rs_df[PREDICT_DETAIL] = df[PREDICT_SCORE].apply(lambda x: predict_detail_dict_to_str({"label": x}))
    else:
        raise ValueError(f"task_type {task_type} is not supported")
    
    pred_rs_df[TYPE] = dataset_type

    

