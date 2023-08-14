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

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.components.components.utils import consts, tools
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params
from fate.ml.ensemble import HeteroSecureBoostGuest, HeteroSecureBoostHost, BINARY_BCE, MULTI_CE, REGRESSION_L2


logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST], provider="fate")
def hetero_sbt(ctx, role):
    ...


@hetero_sbt.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    num_trees: cpn.parameter(type=params.conint(gt=0), default=3,
                          desc="max tree num"),
    learning_rate: cpn.parameter(type=params.confloat(gt=0), default=0.3, desc='decay factor of each tree'),
    max_depth: cpn.parameter(type=params.conint(gt=0), default=3, desc='max depth of a tree'),
    max_bin: cpn.parameter(type=params.conint(gt=0), default=32, desc='max bin number of feature binning'),
    objective: cpn.parameter(type=params.string_choice(choice=[BINARY_BCE, MULTI_CE, REGRESSION_L2]), default=BINARY_BCE, \
                                       desc='objective function, available: {}'.format([BINARY_BCE, MULTI_CE, REGRESSION_L2])),
    num_class: cpn.parameter(type=params.conint(gt=0), default=2, desc='class number of multi classification, active when objective is {}'.format(MULTI_CE)),
    encrypt_key_length: cpn.parameter(type=params.conint(gt=0), default=2048, desc='paillier encrypt key length'),
    l2: cpn.parameter(type=params.confloat(gt=0), default=0.1, desc='L2 regularization'),
    min_impurity_split: cpn.parameter(type=params.confloat(gt=0), default=1e-2, desc='min impurity when splitting a tree node'),
    min_sample_split: cpn.parameter(type=params.conint(gt=0), default=2, desc='min sample to split a tree node'),
    min_leaf_node: cpn.parameter(type=params.conint(gt=0), default=1, desc='mininum sample contained in a leaf node'),
    min_child_weight: cpn.parameter(type=params.confloat(gt=0), default=1, desc='minumum hessian contained in a leaf node'),
    train_data_output: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    train_model_output: cpn.json_model_output(roles=[GUEST, HOST], optional=True),
    train_model_input: cpn.json_model_input(roles=[GUEST, HOST], optional=True)
):
    
    train_data = train_data.read()
    if validate_data is not None:
        validate_data = validate_data.read()

    if role.is_guest:
        
        booster = HeteroSecureBoostGuest(num_trees=num_trees, max_depth=max_depth, learning_rate=learning_rate, max_bin=max_bin,
                                         l2=l2, min_impurity_split=min_impurity_split, min_sample_split=min_sample_split,
                                        min_leaf_node=min_leaf_node, min_child_weight=min_child_weight, encrypt_key_length=encrypt_key_length,
                                        objective=objective, num_class=num_class)
        booster.fit(ctx, train_data, validate_data)
        # get cached train data score
        train_scores = booster.get_train_predict()
        train_data_output.write(train_scores)
        # get tree param
        tree_dict = booster.get_model()
        train_model_output.write(tree_dict, metadata={})

    elif role.is_host:
        
        booster = HeteroSecureBoostHost(num_trees=num_trees, max_depth=max_depth, learning_rate=learning_rate, max_bin=max_bin)
        booster.fit(ctx, train_data, validate_data)
        tree_dict = booster.get_model()
        train_model_output.write(tree_dict, metadata={})

    else:
        raise RuntimeError(f"Unknown role: {role}")



@hetero_sbt.predict()
def predict(
    ctx,
    role: Role,
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    predict_model_input: cpn.json_model_input(roles=[GUEST, HOST]),
    test_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    
    if role.is_guest:
        
        test_data = test_data.read()
        model_input = predict_model_input.read()

    elif role.is_host:
        pass

    else:
        raise RuntimeError(f"Unknown role: {role}")


