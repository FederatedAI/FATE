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
import logging
from typing import Dict
from fate.arch import Context
import numpy as np
import pandas as pd
from fate.arch import Context
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn
from fate.components.core.params import string_choice
from fate.ml.evaluation.tool import (
    get_binary_metrics,
    get_multi_metrics,
    get_regression_metrics,
    get_specified_metrics,
)
from fate.ml.utils.predict_tools import PREDICT_SCORE, PREDICT_RESULT, LABEL
from fate.components.components.utils.consts import BINARY, REGRESSION, MULTI

logger = logging.getLogger(__name__)


def split_dataframe_by_type(input_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    if "type" in input_df.columns:
        return {dataset_type: input_df[input_df["type"] == dataset_type] for dataset_type in input_df["type"].unique()}
    else:
        return {"origin": input_df}


@cpn.component(roles=[GUEST, HOST])
def evaluation(
    ctx: Context,
    role: Role,
    input_data: cpn.dataframe_inputs(roles=[GUEST, HOST]),
    default_eval_setting: cpn.parameter(
        type=string_choice(choice=["binary", "multi", "regression"]), default="binary", optional=True
    ),
    metrics: cpn.parameter(type=list, default=None, optional=True),
    predict_column_name: cpn.parameter(type=str, default=None, optional=True,
                                        desc="predict data column name, if None(default), will use \
                                        'predict_score' in the input dataframe when the default setting are binary and regression, \
                                        and use 'predict_result' if default setting is multi"),
    label_column_name: cpn.parameter(type=str, default=None, optional=True, desc="label data column namem if None(default), \
                                     will use 'label' in the input dataframe")
):

    if role.is_arbiter:
        return
    else:

        if metrics is not None:
            metrics_ensemble = get_specified_metrics(metrics)
            predict_col = predict_column_name if predict_column_name is not None else PREDICT_SCORE
            label_col = label_column_name if label_column_name is not None else LABEL
        else:
            if default_eval_setting == MULTI:
                metrics_ensemble = get_multi_metrics()
                predict_col = predict_column_name if predict_column_name is not None else PREDICT_RESULT
                label_col = label_column_name if label_column_name is not None else LABEL
            else:
                if default_eval_setting == BINARY:
                    metrics_ensemble = get_binary_metrics()
                elif default_eval_setting ==  REGRESSION:
                    metrics_ensemble = get_regression_metrics()
                else:
                    raise ValueError("default_eval_setting should be one of binary, multi, regression, got {}")
                predict_col = predict_column_name if predict_column_name is not None else PREDICT_SCORE
                label_col = label_column_name if label_column_name is not None else LABEL

        df_list = [_input.read() for _input in input_data]
        task_names = [_input.artifact.metadata.source.task_name for _input in input_data]
        eval_rs = {}
        logger.info('components names are {}'.format(task_names))
        for name, df in zip(task_names, df_list):
            rs_ = evaluate(df, metrics_ensemble, predict_col, label_col)
            eval_rs[name] = rs_

    ctx.metrics.log_metrics(eval_rs, name='evaluation', type='evaluation')
    logger.info("eval result: {}".format(eval_rs))


def evaluate(input_data, metrics, predict_col, label_col):

    data = input_data.as_pd_df()
    split_dict = split_dataframe_by_type(data)
    rs_dict = {}
    logger.info('eval dataframe is {}'.format(data))
    for name, df in split_dict.items():
        
        logger.info('eval dataframe is \n\n{}'.format(df))
        y_true = df[label_col]
        # in case is multi result, use tolist
        y_pred = df[predict_col]
        rs = metrics(predict=y_pred, label=y_true)
        rs_dict[name] = rs

    return rs_dict
