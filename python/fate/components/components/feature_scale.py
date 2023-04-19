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

from typing import Union, List

from fate.components import (
    GUEST,
    HOST,
    DatasetArtifact,
    Input,
    ModelArtifact,
    Output,
    Role,
    cpn,
    params
)


@cpn.component(roles=[GUEST, HOST])
def feature_scale(ctx, role):
    ...


@feature_scale.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.parameter("method", type=params.string_choice(["standard", "min_max"]), default="standard", optional=False)
@cpn.parameter("feature_range", type=Union[tuple, dict], default=(0, 1), optional=True,
               desc="Result feature value range for `min_max` method, "
                    "take either dict in format: {col_name: (min, max)} for specific columns "
                    "or (min, max) for all columns. Columns unspecified will be scaled to default range (0,1)")
@cpn.parameter("scale_col", type=List[str], default=None,
               desc="list of column names to be scaled, if None, all columns will be scaled; "
                    "only one of {scale_col, scale_idx} should be specified")
@cpn.parameter("scale_idx", type=List[params.conint(ge=0)], default=None,
               desc="list of column index to be scaled, if None, all columns will be scaled; "
                    "only one of {scale_col, scale_idx} should be specified")
@cpn.parameter("use_anonymous", type=bool, default=False,
               desc="bool, whether interpret `scale_col` as anonymous column names")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def feature_scale_train(
        ctx,
        role: Role,
        train_data,
        method,
        feature_range,
        scale_col,
        scale_idx,
        use_anonymous,
        train_output_data,
        output_model,
):
    train(ctx, train_data, train_output_data, output_model, method, feature_range, scale_col, scale_idx, use_anonymous)


@feature_scale.predict()
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def feature_scale_predict(
        ctx,
        role: Role,
        test_data,
        input_model,
        test_output_data,
):
    predict(ctx, input_model, test_data, test_output_data)


def train(ctx, train_data, train_output_data, output_model, method, feature_range, scale_col, scale_idx, use_anonymous):
    from fate.ml.preprocessing import FeatureScale
    train_data = ctx.reader(train_data).read_dataframe().data

    with ctx.sub_ctx("train") as sub_ctx:
        columns = train_data.schema.columns
        anonymous_columns = None
        if use_anonymous:
            anonymous_columns = train_data.schema.anonymous_columns
            if method != "min_max":
                feature_range = None
        scale_col, feature_range = get_to_scale_cols(columns, anonymous_columns, scale_col, scale_idx, feature_range)

        scaler = FeatureScale(method, scale_col, feature_range)
        train_data = sub_ctx.reader(train_data).read_dataframe().data
        scaler.fit(sub_ctx, train_data)

        model = scaler.to_model()
        with output_model as model_writer:
            model_writer.write_model("preprocessing", model, metadata={})

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = scaler.transform(sub_ctx, train_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def predict(ctx, input_model, test_data, test_output_data):
    from fate.ml.preprocessing import FeatureScale

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        scaler = FeatureScale.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        output_data = scaler.transform(sub_ctx, test_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)


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
                    feature_range[col] = (0, 1)
        else:
            feature_range = {col: feature_range for col in select_col}
    return select_col, feature_range
