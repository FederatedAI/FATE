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
def feature_imputation(ctx, role):
    ...


@feature_imputation.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.parameter("missing_fill_method", type=str, default="designated",
               desc="NA's will be filled using given method")
@cpn.parameter("col_missing_fill_method", type=Union[str, dict], default=None,
               desc="NA's will be filled using given method for each column, "
                    "take dict in format: {col_name: method};"
                    "columns unspecified will take `missing_fill_method`.")
@cpn.parameter("missing_value", type=Union[int, float, List[int], List[float], dict], default=None,
               desc="values to be treated as NA's, if None, will use default NA's;"
                    "take dict in format: {col_name: List[missing values]};"
                    "columns unspecified will take default NA's.")
@cpn.parameter("designated_fill_value", type=Union[int, float, dict], default=0,
               desc="NA's will be filled using this value if missing_fill_method is 'designated',"
                    "take dict in format: {col_name: designated value};"
                    "columns unspecified will take default designated value 0.")
@cpn.parameter("imputation_col", type=List[str], default=None,
               desc="list of column names to be filled, if None, all columns will be filled; "
                    "only one of {imputation_col, imputation_idx} should be specified")
@cpn.parameter("imputation_idx", type=List[params.conint(ge=0)], default=None,
               desc="list of column index to be filled, if None, all columns will be filled; "
                    "only one of {imputation_col, imputation_idx} should be specified")
@cpn.parameter("use_anonymous", type=bool, default=False,
               desc="bool, whether interpret col names in "
                    "`imputation_col`, `missing_value`, `col_missing_fill_method`, and `designated_fill_value` "
                    "as anonymous column names")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def feature_imputation_train(
        ctx,
        role: Role,
        train_data,
        missing_fill_method,
        col_missing_fill_method,
        missing_value,
        designated_fill_value,
        imputation_col,
        imputation_idx,
        use_anonymous,
        train_output_data,
        output_model,
):
    train(ctx, train_data, train_output_data, output_model,
          missing_fill_method, col_missing_fill_method,
          missing_value, designated_fill_value,
          imputation_col, imputation_idx,
          use_anonymous)


@feature_imputation.predict()
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def feature_imputation_predict(
        ctx,
        role: Role,
        test_data,
        input_model,
        test_output_data,
):
    predict(ctx, input_model, test_data, test_output_data)


def train(ctx, train_data, train_output_data, output_model,
          missing_fill_method, col_missing_fill_method,
          missing_value, designated_fill_value,
          imputation_col, imputation_idx,
          use_anonymous):
    from fate.ml.preprocessing import FeatureImputation
    train_data = ctx.reader(train_data).read_dataframe().data
    if missing_value is not None:
        if not isinstance(missing_value, list):
            missing_value = [missing_value]

    with ctx.sub_ctx("train") as sub_ctx:
        columns = train_data.schema.columns.to_list()
        anonymous_columns = None
        if use_anonymous:
            anonymous_columns = train_data.schema.anonymous_columns.to_list()
        imputation_col, col_missing_fill_method = get_to_imputation_cols(columns,
                                                                         anonymous_columns,
                                                                         imputation_col,
                                                                         imputation_idx,
                                                                         missing_fill_method,
                                                                         col_missing_fill_method)
        missing_value, designated_fill_value = get_missing_fill_value(columns,
                                                                      anonymous_columns,
                                                                      imputation_col,
                                                                      missing_value,
                                                                      designated_fill_value)

        imputation_obj = FeatureImputation(imputation_col, col_missing_fill_method, missing_value,
                                           designated_fill_value)
        imputation_obj.fit(sub_ctx, train_data)

        model = imputation_obj.to_model()
        with output_model as model_writer:
            model_writer.write_model("feature_imputation", model, metadata={})

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = imputation_obj.transform(sub_ctx, train_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def predict(ctx, input_model, test_data, test_output_data):
    from fate.ml.preprocessing import FeatureImputation

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        imputation_obj = FeatureImputation.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        output_data = imputation_obj.transform(sub_ctx, test_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)


def get_to_imputation_cols(columns, anonymous_columns, imputation_col, imputation_idx, missing_fill_method,
                           col_missing_fill_method):
    if anonymous_columns is not None:
        imputation_col = [columns[anonymous_columns.index(col)] for col in imputation_col]

    if imputation_col is not None:
        if imputation_idx is not None:
            raise ValueError(f"`imputation_col` and `imputation_idx` cannot be specified simultaneously, please check.")
        select_col = imputation_col
    elif imputation_idx is not None:
        select_col = [columns[i] for i in imputation_idx]
    else:
        select_col = columns
    col_set = set(columns)
    if not all(col in col_set for col in select_col):
        raise ValueError(f"Given imputation columns not found in data schema, please check.")

    if col_missing_fill_method is not None:
        if anonymous_columns is not None:
            col_missing_fill_method = {columns[anonymous_columns.index(col)]:
                                           col_missing_fill_method[col] for col in col_missing_fill_method}
        for col in select_col:
            if col not in col_missing_fill_method:
                col_missing_fill_method[col] = missing_fill_method

    return select_col, col_missing_fill_method


def get_missing_fill_value(columns, anonymous_columns,
                           imputation_col, missing_value, designated_fill_value):
    if missing_value is not None:
        if not isinstance(missing_value, dict):
            if not isinstance(missing_value, list):
                missing_value = [missing_value]
            missing_value = {col: missing_value for col in imputation_col}
        elif anonymous_columns is not None:
            missing_value = {columns[anonymous_columns.index(col)]: missing_value[col] for col in missing_value}

    if designated_fill_value is not None:
        if not isinstance(designated_fill_value, dict):
            designated_fill_value = {col: designated_fill_value for col in imputation_col}
        elif anonymous_columns is not None:
            designated_fill_value = {columns[anonymous_columns.index(col)]: designated_fill_value[col] for col in
                                     designated_fill_value}
            for col in imputation_col:
                if col not in designated_fill_value:
                    designated_fill_value[col] = 0

    return missing_value, designated_fill_value
