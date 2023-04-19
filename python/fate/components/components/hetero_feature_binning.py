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

from typing import List

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
def hetero_feature_binning(ctx, role):
    ...


"""
@cpn.parameter("bins", type=dict, default=[],
               desc="dict of format {col_name: [bins]} which specifies bin edges for each feature, "
                    "including right edge of last bin")
"""


@hetero_feature_binning.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.parameter("method", type=params.string_choice(["quantile", "bucket"]),
               default="quantile", optional=False,
               desc="binning method, options: {quantile, bucket, manual}")
@cpn.parameter("n_bins", type=params.conint(gt=1), default=10,
               desc="max number of bins, should be no less than 2")
@cpn.parameter("split_pt_dict", type=dict, default=None,
               desc="dict, manually provided split points, only effective when `method`='manual'")
@cpn.parameter("bin_col", type=List[str], default=None,
               desc="list of column names to be binned, if None, all columns will be binned; "
                    "only one of {bin_col, bin_idx} should be specified")
@cpn.parameter("bin_idx", type=List[params.conint(ge=0)], default=None,
               desc="list of column index to be binned, if None, all columns will be binned; "
                    "only one of {bin_col, bin_idx} should be specified")
@cpn.parameter("category_col", type=List[str], default=None,
               desc="list of column names to be treated as categorical features and will not be binned; "
                    "only one of {category_col, category_idx} should be specified"
                    "note that metrics will be computed over categorical features if this param is specified")
@cpn.parameter("category_idx", type=List[params.conint(ge=0)], default=None,
               desc="list of column index to be treated as categorical features and will not be binned; "
                    "only one of {category_col, category_idx} should be specified"
                    "note that metrics will be computed over categorical features if this param is specified")
@cpn.parameter("use_anonymous", type=bool, default=False,
               desc="bool, whether interpret `bin_col` & `category_col` as anonymous column names")
@cpn.parameter("transform_method", type=str, default=None,  # may support user-provided dict in future release
               desc="str, values to which binned data will be transformed, select from {'woe', 'bin_idx'}; "
                    "note that host will not transform features to woe values regardless of setting")
@cpn.parameter("skip_metrics", type=bool, default=False, desc="bool, whether compute host's metrics or not")
@cpn.parameter("local_only", type=bool, default=False, desc="bool, whether compute host's metrics or not")
@cpn.parameter("error_rate", type=params.confloat(gt=0, le=1), default=1e-3, desc="float, error rate for quantile")
@cpn.parameter("adjustment_factor", type=params.confloat(gt=0), default=0.5,
               desc="float, useful when here is no event or non-event in a bin")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def feature_binning_train(
        ctx,
        role: Role,
        train_data,
        method,
        split_pt_dict,
        n_bins,
        bin_col,
        bin_idx,
        category_col,
        category_idx,
        use_anonymous,
        transform_method,
        skip_metrics,
        local_only,
        error_rate,
        adjustment_factor,
        train_output_data,
        output_model,
):
    train(ctx, train_data, train_output_data, output_model, role, method, n_bins, split_pt_dict,
          bin_col, bin_idx, category_col, category_idx, use_anonymous, transform_method,
          skip_metrics, local_only, error_rate, adjustment_factor)


@hetero_feature_binning.predict()
@cpn.parameter("transform_method", type=str, default=None,  # may support user-provided dict in future release
               desc="str, values to which binned data will be transformed, select from {'woe', 'bin_idx'}; "
                    "note that host will not transform features to woe values regardless of setting")
@cpn.parameter("skip_metrics", type=bool, default=False, desc="bool, whether compute host's metrics or not")
@cpn.parameter("local_only", type=bool, default=False, desc="bool, whether compute host's metrics or not")
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def feature_binning_predict(
        ctx,
        role: Role,
        test_data,
        input_model,
        transform_method,
        skip_metrics,
        local_only,
        test_output_data,
):
    predict(ctx, input_model, test_data, test_output_data, role, transform_method, skip_metrics, local_only)


def train(ctx, train_data, train_output_data, output_model, role, method, n_bins, split_pt_dict,
          bin_col, bin_idx, category_col, category_idx, use_anonymous, transform_method,
          skip_metrics, local_only, error_rate, adjustment_factor):
    from fate.ml.feature_binning import HeteroBinningModuleHost, HeteroBinningModuleGuest

    with ctx.sub_ctx("train") as sub_ctx:
        train_data = sub_ctx.reader(train_data).read_dataframe().data
        columns = train_data.schema.columns.to_list()
        anonymous_columns = None
        if use_anonymous:
            anonymous_columns = train_data.schema.anonymous_columns
            split_pt_dict = {columns[anonymous_columns.index(col)]: split_pt_dict[col] for col in split_pt_dict.keys()}
        to_bin_cols, merged_category_col = get_to_bin_cols(columns, anonymous_columns,
                                                           bin_col, bin_idx, category_col, category_idx)

        if role.is_guest:
            binning = HeteroBinningModuleGuest(method, n_bins, split_pt_dict, to_bin_cols, transform_method,
                                               merged_category_col, local_only, error_rate, adjustment_factor)
        elif role.is_host:
            binning = HeteroBinningModuleHost(method, n_bins, split_pt_dict, to_bin_cols, transform_method,
                                              merged_category_col, local_only, error_rate, adjustment_factor)
        binning.fit(sub_ctx, train_data)
        if not skip_metrics:
            binned_data = binning._bin_obj.bucketize_data(train_data)
            binning.compute_metrics(sub_ctx, binned_data)
        model = binning.to_model()
        with output_model as model_writer:
            model_writer.write_model("feature_binning", model, metadata={"transform_method": transform_method,
                                                                         "skip_metrics": skip_metrics,
                                                                         "local_only": local_only,
                                                                         "model_type": "binning"})

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = train_data
        if transform_method is not None:
            binned_data = binning._bin_obj.bucketize_data(train_data)
            output_data = binning.transform(sub_ctx, binned_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def predict(ctx, input_model, test_data, test_output_data, role, transform_method, skip_metrics, local_only):
    from fate.ml.feature_binning import HeteroBinningModuleHost, HeteroBinningModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        if role.is_guest:
            binning = HeteroBinningModuleGuest.from_model(model)
        elif role.is_host:
            binning = HeteroBinningModuleHost.from_model(model)
        model_meta = model["meta_data"]
        binning.local_only = local_only or model_meta["local_only"]
        transform_method = transform_method or model_meta["transform_method"]
        skip_metrics = skip_metrics or model_meta["skip_metrics"]

        binning.set_transform_method(transform_method)
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        if skip_metrics and transform_method is None:
            return test_data
        binned_data = binning._bin_obj.bucketize_data(test_data)
        if not skip_metrics:
            binning.compute_metrics(sub_ctx, binned_data)
        output_data = test_data
        if transform_method is not None:
            output_data = binning.transform(sub_ctx, binned_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)


def get_to_bin_cols(columns, anonymous_columns, bin_col, bin_idx, category_col, category_idx):
    if anonymous_columns is not None:
        bin_col = [columns[anonymous_columns.index(col)] for col in bin_col]
        category_col = [columns[anonymous_columns.index(col)] for col in category_col]

    if bin_col is not None:
        if bin_idx is not None:
            raise ValueError(f"`bin_col` and `bin_idx` cannot be specified simultaneously, please check.")
        select_col = bin_col
    elif bin_idx is not None:
        select_col = [columns[i] for i in bin_idx]
    else:
        select_col = columns
    col_set = set(columns)
    if not all(col in col_set for col in select_col):
        raise ValueError(f"Given bin columns not found in data schema, please check.")

    if category_col is not None:
        if category_idx is not None:
            raise ValueError(f"`category_col` and `category_idx` cannot be specified simultaneously, please check.")
    elif category_idx is not None:
        category_col = [columns[i] for i in category_idx]
    else:
        return select_col, []
    if not all(col in col_set for col in category_col):
        raise ValueError(f"Given category columns not found in data schema, please check.")
    category_col_set = set(category_col)
    select_col = [col for col in select_col if col not in category_col_set]
    return select_col, category_col
