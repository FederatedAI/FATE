from fate.arch.dataframe._dataframe import DataFrame
import pandas as pd
from pandas import DataFrame as pd_DataFrame
from typing import Union
from fate.components.components.utils import consts
from fate.arch.dataframe import PandasReader
from typing import Literal
from fate.arch.dataframe.manager.schema_manager import Schema


PREDICT_DF_SCHEMA = [
    "label",
    "predict_result",
    "predict_score",
    "predict_detail",
    "type",
    "sample_id"
]


def pd_predict_score_to_fate_df(pred_score: pd_DataFrame, schema: Schema, sample_id: pd.DataFrame, match_id: pd.DataFrame):
    pass


def predict_score_to_output(pred_score: Union[DataFrame, pd_DataFrame], schema: pd.Series, dataset_type=consts.TRAIN_SET, 
                            task_type=consts.BINARY, classes=None, threshold=0.5) -> DataFrame:

    if isinstance(pred_score, DataFrame):
        pass
    
    elif isinstance(pred_score, pd_DataFrame):
        pass

    # reader = PandasReader(sample_id_name="sample_id", match_id_name=result_df.match_id_name, label_name="label", dtype="object")
    # data = reader.to_frame(ctx, result_df)