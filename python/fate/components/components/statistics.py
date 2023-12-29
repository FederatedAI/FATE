#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

from typing import List, Union

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params


@cpn.component(roles=[GUEST, HOST], provider="fate")
def statistics(
    ctx: Context,
    role: Role,
    input_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    metrics: cpn.parameter(
        type=Union[
            List[Union[params.statistic_metrics_param(), params.legal_percentile()]],
            params.statistic_metrics_param(),
            params.legal_percentile(),
        ],
        default=["mean", "std", "min", "max"],
        desc="metrics to be computed, default ['count', 'mean', 'std', 'min', 'max']",
    ),
    ddof: cpn.parameter(
        type=params.conint(ge=0), default=1, desc="Delta Degrees of Freedom for std and var, default 1"
    ),
    bias: cpn.parameter(
        type=bool,
        default=True,
        desc="If False, the calculations of skewness and kurtosis are corrected for statistical bias.",
    ),
    relative_error: cpn.parameter(
        type=params.confloat(gt=0, le=1), default=1e-3, desc="float, error rate for quantile"
    ),
    skip_col: cpn.parameter(
        type=List[str],
        default=None,
        optional=True,
        desc="columns to be skipped, default None; if None, statistics will be computed over all columns",
    ),
    use_anonymous: cpn.parameter(
        type=bool, default=False, desc="bool, whether interpret `skip_col` as anonymous column names"
    ),
    output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
    from fate.ml.statistics.statistics import FeatureStatistics

    sub_ctx = ctx.sub_ctx("train")
    input_data = input_data.read()
    select_cols = get_to_compute_cols(
        input_data.schema.columns, input_data.schema.anonymous_columns, skip_col, use_anonymous
    )
    if isinstance(metrics, str):
        metrics = [metrics]
    if len(metrics) > 1:
        for metric in metrics:
            if metric == "describe":
                raise ValueError(f"'describe' should not be combined with additional metric names.")
    stat_computer = FeatureStatistics(list(set(metrics)), ddof, bias, relative_error)
    input_data = input_data[select_cols]
    stat_computer.fit(sub_ctx, input_data)

    model = stat_computer.get_model()
    output_model.write(model, metadata={})


def get_to_compute_cols(columns, anonymous_columns, skip_columns, use_anonymous):
    if skip_columns is None:
        skip_columns = []
    if use_anonymous and skip_columns is not None:
        skip_columns = [anonymous_columns[columns.index(col)] for col in skip_columns]
    skip_col_set = set(skip_columns)
    select_columns = [col for col in columns if col not in skip_col_set]

    return select_columns
