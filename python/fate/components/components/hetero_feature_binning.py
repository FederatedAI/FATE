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

import logging
from typing import List

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params
from fate.ml.feature_binning import HeteroBinningModuleHost, HeteroBinningModuleGuest

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST], provider="fate")
def hetero_feature_binning(ctx, role):
    ...


"""
@cpn.parameter("bins", type=dict, default=[],
               desc="dict of format {col_name: [bins]} which specifies bin edges for each feature, "
                    "including right edge of last bin")
"""


@hetero_feature_binning.train()
def feature_binning_train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        method: cpn.parameter(type=params.string_choice(["quantile", "bucket", "manual"]),
                              default="quantile", optional=False,
                              desc="binning method, options: {quantile, bucket, manual}"),
        n_bins: cpn.parameter(type=params.conint(gt=1), default=10,
                              desc="max number of bins, should be no less than 2"),
        split_pt_dict: cpn.parameter(type=dict, default=None, optional=True,
                                     desc="dict, manually provided split points, "
                                          "only effective when `method`='manual'"),
        bin_col: cpn.parameter(type=List[str], default=None,
                               desc="list of column names to be binned, if None, all columns will be binned; "
                                    "only one of {bin_col, bin_idx} should be specified"),
        bin_idx: cpn.parameter(type=List[params.conint(ge=0)], default=None,
                               desc="list of column index to be binned, if None, all columns will be binned; "
                                    "only one of {bin_col, bin_idx} should be specified"),
        category_col: cpn.parameter(type=List[str], default=None,
                                    desc="list of column names to be treated as categorical "
                                         "features and will not be binned; "
                                         "only one of {category_col, category_idx} should be specified"
                                         "note that metrics will be computed over categorical features "
                                         "if this param is specified"),
        category_idx: cpn.parameter(type=List[params.conint(ge=0)], default=None,
                                    desc="list of column index to be treated as categorical features "
                                         "and will not be binned; "
                                         "only one of {category_col, category_idx} should be specified"
                                         "note that metrics will be computed over categorical features "
                                         "if this param is specified"),
        use_anonymous: cpn.parameter(type=bool, default=False,
                                     desc="bool, whether interpret `bin_col` & `category_col` "
                                          "as anonymous column names"),
        transform_method: cpn.parameter(type=params.string_choice(['woe', 'bin_idx']),
                                        default=None,  # may support user-provided dict in future release
                                        desc="str, values to which binned data will be transformed, "
                                             "select from {'woe', 'bin_idx'}; "
                                             "note that host will not transform features "
                                             "to woe values regardless of setting"),
        skip_metrics: cpn.parameter(type=bool, default=False,
                                    desc="bool, whether compute host's metrics or not"),
        local_only: cpn.parameter(type=bool, default=False, desc="bool, whether compute host's metrics or not"),
        relative_error: cpn.parameter(type=params.confloat(gt=0, le=1), default=1e-6,
                                      desc="float, error rate for quantile"),
        adjustment_factor: cpn.parameter(type=params.confloat(gt=0), default=0.5,
                                         desc="float, useful when here is no event or non-event in a bin"),
        he_param: cpn.parameter(type=params.he_param(), default=params.HEParam(kind="paillier", key_length=1024),
                                desc="homomorphic encryption param"),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
    ctx.cipher.set_phe(ctx.device, he_param.dict())
    train(ctx, train_data, train_output_data, output_model, role, method, n_bins, split_pt_dict,
          bin_col, bin_idx, category_col, category_idx, use_anonymous, transform_method,
          skip_metrics, local_only, relative_error, adjustment_factor)


@hetero_feature_binning.predict()
def feature_binning_predict(
        ctx: Context,
        role: Role,
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        transform_method: cpn.parameter(type=params.string_choice(['woe', 'bin_idx']),
                                        default=None,  # may support user-provided dict in future release
                                        desc="str, values to which binned data will be transformed, "
                                             "select from {'woe', 'bin_idx'}; "
                                             "note that host will not transform features "
                                             "to woe values regardless of setting"),
        skip_metrics: cpn.parameter(type=bool, default=False,
                                    desc="bool, whether compute host's metrics or not"),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    predict(ctx, input_model, test_data, test_output_data, role, transform_method, skip_metrics)


def train(ctx, train_data, train_output_data, output_model, role, method, n_bins, split_pt_dict,
          bin_col, bin_idx, category_col, category_idx, use_anonymous, transform_method,
          skip_metrics, local_only, relative_error, adjustment_factor):
    logger.info(f"start binning train")
    sub_ctx = ctx.sub_ctx("train")
    train_data = train_data.read()
    columns = train_data.schema.columns.to_list()
    anonymous_columns = None
    if use_anonymous:
        anonymous_columns = train_data.schema.anonymous_columns.to_list()
        split_pt_dict = {columns[anonymous_columns.index(col)]: split_pt_dict[col] for col in split_pt_dict.keys()}
    to_bin_cols, merged_category_col = get_to_bin_cols(columns, anonymous_columns,
                                                       bin_col, bin_idx, category_col, category_idx)
    if split_pt_dict:
        to_bin_cols = list(set(to_bin_cols).intersection(split_pt_dict.keys()))

    if role.is_guest:
        binning = HeteroBinningModuleGuest(method, n_bins, split_pt_dict, to_bin_cols, transform_method,
                                           merged_category_col, local_only, relative_error, adjustment_factor)
    elif role.is_host:
        binning = HeteroBinningModuleHost(method, n_bins, split_pt_dict, to_bin_cols, transform_method,
                                          merged_category_col, local_only, relative_error, adjustment_factor)
    else:
        raise ValueError(f"unknown role: {role}")
    binning.fit(sub_ctx, train_data)
    binned_data = None
    if not skip_metrics:
        binned_data = binning._bin_obj.bucketize_data(train_data)
        binning.compute_metrics(sub_ctx, binned_data)
    model = binning.get_model()
    output_model.write(model)

    sub_ctx = ctx.sub_ctx("predict")
    output_data = train_data
    if transform_method is not None:
        if binned_data is None:
            binned_data = binning._bin_obj.bucketize_data(train_data)
        output_data = binning.transform(sub_ctx, binned_data)
    train_output_data.write(output_data)


def predict(ctx, input_model, test_data, test_output_data, role, transform_method, skip_metrics):
    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    if role.is_guest:
        binning = HeteroBinningModuleGuest.from_model(model)
    elif role.is_host:
        binning = HeteroBinningModuleHost.from_model(model)
    # model_meta = model["meta_data"]
    else:
        raise ValueError(f"unknown role: {role}")

    binning.set_transform_method(transform_method)
    test_data = test_data.read()
    if skip_metrics and transform_method is None:
        return test_data
    binned_data = binning._bin_obj.bucketize_data(test_data)
    if not skip_metrics:
        binning.compute_metrics(sub_ctx, binned_data)
    output_data = test_data
    if transform_method is not None:
        output_data = binning.transform(sub_ctx, binned_data)
    test_output_data.write(output_data)


def get_to_bin_cols(columns, anonymous_columns, bin_col, bin_idx, category_col, category_idx):
    if anonymous_columns is not None:
        if bin_col is not None:
            bin_col = [columns[anonymous_columns.index(col)] for col in bin_col]
        if category_col is not None:
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
