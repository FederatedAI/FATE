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

from typing import Union

from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params
from fate.ml.model_selection.data_split import DataSplitModuleGuest, DataSplitModuleHost


@cpn.component(roles=[GUEST, HOST], provider="fate")
def data_split(
        ctx: Context,
        role: Role,
        input_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        train_size: cpn.parameter(type=Union[params.conint(ge=0), params.confloat(ge=0.0)], default=None,
                                  desc="size of output training data, should be either int for exact sample size or float for fraction"),
        validate_size: cpn.parameter(type=Union[params.conint(ge=0), params.confloat(ge=0.0)], default=None,
                                     desc="size of output validation data, should be either int for exact sample size or float for fraction"),
        test_size: cpn.parameter(type=Union[params.conint(ge=0), params.confloat(ge=0.0)], default=None,
                                 desc="size of output test data, should be either int for exact sample size or float for fraction"),
        stratified: cpn.parameter(type=bool, default=False,
                                  desc="whether sample with stratification, "
                                       "should not use this for data with continuous label values"),
        random_state: cpn.parameter(type=params.conint(ge=0), default=None, desc="random state"),
        ctx_mode: cpn.parameter(type=params.string_choice(["hetero", "homo", "local"]), default="hetero",
                                desc="sampling mode, 'homo' & 'local' will both sample locally"),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
        validate_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
):
    if isinstance(train_size, float) or isinstance(validate_size, float) or isinstance(test_size, float):
        if train_size + validate_size + test_size > 1:
            raise ValueError("(train_size + validate_size + test_size) should be less than or equal to 1.0")
    if train_size is None and validate_size is None and test_size is None:
        train_size = 0.8
        validate_size = 0.2
        test_size = 0.0

    sub_ctx = ctx.sub_ctx("train")
    if role.is_guest:
        module = DataSplitModuleGuest(train_size, validate_size, test_size, stratified, random_state, ctx_mode)
    elif role.is_host:
        module = DataSplitModuleHost(train_size, validate_size, test_size, stratified, random_state, ctx_mode)
    input_data = input_data.read()

    train_data_set, validate_data_set, test_data_set = module.fit(sub_ctx, input_data)
    # train_data_set, validate_data_set, test_data_set = module.split_data(train_data)
    if train_data_set:
        train_output_data.write(train_data_set)
    if validate_data_set:
        validate_output_data.write(validate_data_set)
    if test_data_set:
        test_output_data.write(test_data_set)
