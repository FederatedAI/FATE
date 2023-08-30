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

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST], provider="fate")
def one_hot_encoder(ctx, role):
    ...


@one_hot_encoder.train()
def one_hot_encoder_train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        drop: cpn.parameter(type=params.string_choice(["first", "if_binary"]),
                            default=None, optional=True,
                            desc="whether the first category should be dropped from the encoded features;"
                                 "None means all encoded columns will be retained;"
                                 "if 'if_binary', "
                                 "only drop the first category if feature contains exactly two categories; "
                                 "if 'first', then the first category will always be dropped, "
                                 "and so single-valued features will be dropped;"
                                 "default None"),
        encode_col: cpn.parameter(
            type=List[str],
            default=None,
            optional=True,
            desc="list of column names to be encoded, if None, all columns will be encoded; "
                 "only one of {encode_col, encode_idx} should be specified",
        ),
        encode_idx: cpn.parameter(
            type=List[params.conint(ge=0)],
            default=None,
            optional=True,
            desc="list of column index to be encoded, if None, all columns will be encoded; "
                 "only one of {encode_col, encode_idx} should be specified",
        ),
        handle_unknown: cpn.parameter(type=params.string_choice(["error", "ignore"]), default="error",
                                      desc="Whether to raise an error or ignore if an unknown categorical "
                                           "feature is present during transform; if 'ignore', "
                                           "this entry will be zero in all one-hot-encoded columns; "
                                           "default is to raise an error"),
        use_anonymous: cpn.parameter(
            type=bool, default=False, desc="bool, whether interpret `encode_col` as anonymous column names"
        ),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
    train(
        ctx,
        train_data,
        train_output_data,
        output_model,
        drop,
        encode_col,
        encode_idx,
        handle_unknown,
        use_anonymous,
    )


@one_hot_encoder.predict()
def one_hot_encoder_predict(
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
        drop,
        encode_col,
        encode_idx,
        handle_unknown,
        use_anonymous,
):
    logger.info(f"start scale train")
    from fate.ml.preprocessing import OneHotEncoder

    train_data = train_data.read()

    sub_ctx = ctx.sub_ctx("train")
    columns = train_data.schema.columns.to_list()
    anonymous_columns = None
    if use_anonymous:
        anonymous_columns = train_data.schema.anonymous_columns.to_list()

    encode_col = get_to_encode_cols(columns, anonymous_columns, encode_col, encode_idx)

    encoder = OneHotEncoder(drop, handle_unknown, encode_col)
    encoder.fit(sub_ctx, train_data)

    sub_ctx = ctx.sub_ctx("predict")
    output_data = encoder.transform(sub_ctx, train_data)
    train_output_data.write(output_data)

    model = encoder.get_model()
    output_model.write(model, metadata={})


def predict(ctx, input_model, test_data, test_output_data):
    logger.info(f"start scale transform")

    from fate.ml.preprocessing import OneHotEncoder

    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    encoder = OneHotEncoder.from_model(model)
    test_data = test_data.read()
    output_data = encoder.transform(sub_ctx, test_data)
    test_output_data.write(output_data)


def get_to_encode_cols(columns, anonymous_columns, encode_col, encode_idx):
    if anonymous_columns is not None:
        encode_col = [columns[anonymous_columns.index(col)] for col in encode_col]

    if encode_col is not None:
        if encode_idx is not None:
            raise ValueError(f"`encode_col` and `encode_idx` cannot be specified simultaneously, please check.")
        select_col = encode_col
    elif encode_idx is not None:
        select_col = [columns[i] for i in encode_idx]
    else:
        select_col = columns
    col_set = set(columns)
    if not all(col in col_set for col in select_col):
        raise ValueError(f"Given scale columns not found in data schema, please check.")

    if len(select_col) == 0:
        logger.warning(f"No cols provided. "
                       f"To scale all columns, please set `encode_col` to None.")
    return select_col
