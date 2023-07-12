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
from fate.components.components.utils.predict_format import PREDICT_SCORE, LABEL

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
    default_eval_metrics: cpn.parameter(
        type=string_choice(choice=["binary", "multi", "regression"]), default="binary", optional=True
    ),
    metrics: cpn.parameter(type=list, default=None, optional=True)
):

    if role.is_arbiter:
        return
    else:
        if metrics is not None:
            metrics_ensemble = get_specified_metrics(metrics)
        else:
            if default_eval_metrics == "binary":
                metrics_ensemble = get_binary_metrics()
            elif default_eval_metrics == "multi":
                metrics_ensemble = get_multi_metrics()
            elif default_eval_metrics == "regression":
                metrics_ensemble = get_regression_metrics()

        df_list = [_input.read() for _input in input_data]
        component_name = [_input.artifact.metadata.source.component for _input in input_data]
        component_rs = {}
        for name, df in zip(component_name, df_list):
            rs_dict = evaluate(df, metrics_ensemble)
            component_rs[name] = rs_dict

    ctx.metrics.log_metrics(rs_dict, name='evaluation', type='evaluation')
    logger.info("eval result: {}".format(rs_dict))


def evaluate(input_data, metrics):

    data = input_data.as_pd_df()
    split_dict = split_dataframe_by_type(data)
    rs_dict = {}
    logger.info('eval dataframe is {}'.format(data))
    
    for name, df in split_dict.items():
        
        logger.info('eval dataframe is \n\n{}'.format(df))
        y_true = df[LABEL]
        # in case is multi result, use tolist
        y_pred = df[PREDICT_SCORE]
        rs = metrics(predict=y_pred, label=y_true)
        rs_dict[name] = rs

    return rs_dict
