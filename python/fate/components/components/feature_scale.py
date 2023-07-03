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


@cpn.component(roles=[GUEST, HOST])
def feature_scale(ctx, role):
    ...


@feature_scale.train()
def feature_scale_train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        method: cpn.parameter(type=params.string_choice(["standard", "min_max"]), default="standard", optional=False),
        feature_range: cpn.parameter(
            type=Union[list, dict],
            default=[0, 1],
            optional=True,
            desc="Result feature value range for `min_max` method, "
                 "take either dict in format: {col_name: [min, max]} for specific columns "
                 "or [min, max] for all columns. Columns unspecified will be scaled to default range [0,1]",
    ),
    scale_col: cpn.parameter(
        type=List[str],
        default=None,
        optional=True,
        desc="list of column names to be scaled, if None, all columns will be scaled; "
             "only one of {scale_col, scale_idx} should be specified",
    ),
    scale_idx: cpn.parameter(
        type=List[params.conint(ge=0)],
        default=None,
        optional=True,
        desc="list of column index to be scaled, if None, all columns will be scaled; "
             "only one of {scale_col, scale_idx} should be specified",
    ),
    strict_range: cpn.parameter(
        type=bool,
        default=True,
        desc="whether transformed value to be strictly restricted within given range; "
        "effective for 'min_max' scale method only",
    ),
    use_anonymous: cpn.parameter(
        type=bool, default=False, desc="bool, whether interpret `scale_col` as anonymous column names"
    ),
    train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
    output_model: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    train(
        ctx,
        train_data,
        train_output_data,
        output_model,
        method,
        feature_range,
        scale_col,
        scale_idx,
        strict_range,
        use_anonymous,
    )


@feature_scale.predict()
def feature_scale_predict(
        ctx: Context,
        role: Role,
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    predict(ctx, input_model, test_data, test_output_data)


def train(
    ctx,
    train_data,
    train_output_data,
    output_model,
    method,
    feature_range,
    scale_col,
    scale_idx,
    strict_range,
    use_anonymous,
):
    from fate.ml.preprocessing import FeatureScale

    train_data = train_data.read()

    with ctx.sub_ctx("train") as sub_ctx:
        columns = train_data.schema.columns.to_list()
        anonymous_columns = None
        if use_anonymous:
            anonymous_columns = train_data.schema.anonymous_columns.to_list()
            if method != "min_max":
                feature_range = None
        scale_col, feature_range = get_to_scale_cols(columns, anonymous_columns, scale_col, scale_idx, feature_range)

        scaler = FeatureScale(method, scale_col, feature_range, strict_range)
        scaler.fit(sub_ctx, train_data)

        model = scaler.to_model()
        with output_model as model_writer:
            model_writer.write_model("feature_scale", model, metadata={})

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = scaler.transform(sub_ctx, train_data)
        train_output_data.write(output_data)


def predict(ctx, input_model, test_data, test_output_data):
    from fate.ml.preprocessing import FeatureScale

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        scaler = FeatureScale.from_model(model)
        test_data = test_data.read()
        output_data = scaler.transform(sub_ctx, test_data)
        test_output_data.write(output_data)


def get_to_scale_cols(columns, anonymous_columns, scale_col, scale_idx, feature_range):
    if anonymous_columns is not None:
        scale_col = [columns[anonymous_columns.index(col)] for col in scale_col]

    if scale_col is not None:
        if scale_idx is not None:
            raise ValueError(f"`scale_col` and `scale_idx` cannot be specified simultaneously, please check.")
        select_col = scale_col
    elif scale_idx is not None:
        select_col = [columns[i] for i in scale_idx]
    else:
        select_col = columns
    col_set = set(columns)
    if not all(col in col_set for col in select_col):
        raise ValueError(f"Given scale columns not found in data schema, please check.")

    if feature_range is not None:
        if isinstance(feature_range, dict):
            for col in select_col:
                if col not in feature_range:
                    feature_range[col] = [0, 1]
        else:
            feature_range = {col: feature_range for col in select_col}
    return select_col, feature_range
