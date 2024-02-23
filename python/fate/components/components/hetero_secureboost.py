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
from fate.arch.dataframe import DataFrame
from fate.arch import Context
from fate.components.core import GUEST, HOST, Role, cpn, params
from fate.ml.ensemble import HeteroSecureBoostGuest, HeteroSecureBoostHost, BINARY_BCE, MULTI_CE, REGRESSION_L2
from fate.components.components.utils.tools import add_dataset_type
from fate.components.components.utils import consts


logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST], provider="fate")
def hetero_secureboost(ctx, role):
    ...


@hetero_secureboost.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    num_trees: cpn.parameter(type=params.conint(gt=0), default=3, desc="max tree num"),
    learning_rate: cpn.parameter(type=params.confloat(gt=0), default=0.3, desc="decay factor of each tree"),
    max_depth: cpn.parameter(type=params.conint(gt=0), default=3, desc="max depth of a tree"),
    complete_secure: cpn.parameter(
        type=params.conint(ge=0),
        default=0,
        desc="number of trees to use guest features only in the complete secure mode, " "0 means no complete secure",
    ),
    max_bin: cpn.parameter(type=params.conint(gt=0), default=32, desc="max bin number of feature binning"),
    objective: cpn.parameter(
        type=params.string_choice(choice=[BINARY_BCE, MULTI_CE, REGRESSION_L2]),
        default=BINARY_BCE,
        desc="objective function, available: {}".format([BINARY_BCE, MULTI_CE, REGRESSION_L2]),
    ),
    num_class: cpn.parameter(
        type=params.conint(gt=0),
        default=2,
        desc="class number of multi classification, active when objective is {}".format(MULTI_CE),
    ),
    goss: cpn.parameter(type=bool, default=False, desc="whether to use goss subsample"),
    goss_start_iter: cpn.parameter(type=params.conint(ge=0), default=5, desc="start iteration of goss subsample"),
    top_rate: cpn.parameter(type=params.confloat(gt=0, lt=1), default=0.2, desc="top rate of goss subsample"),
    other_rate: cpn.parameter(type=params.confloat(gt=0, lt=1), default=0.1, desc="other rate of goss subsample"),
    l1: cpn.parameter(type=params.confloat(ge=0), default=0, desc="L1 regularization"),
    l2: cpn.parameter(type=params.confloat(ge=0), default=0.1, desc="L2 regularization"),
    min_impurity_split: cpn.parameter(
        type=params.confloat(gt=0), default=1e-2, desc="min impurity when splitting a tree node"
    ),
    min_sample_split: cpn.parameter(type=params.conint(gt=0), default=2, desc="min sample to split a tree node"),
    min_leaf_node: cpn.parameter(type=params.conint(gt=0), default=1, desc="mininum sample contained in a leaf node"),
    min_child_weight: cpn.parameter(
        type=params.confloat(gt=0), default=1, desc="minumum hessian contained in a leaf node"
    ),
    gh_pack: cpn.parameter(type=bool, default=True, desc="whether to pack gradient and hessian together"),
    split_info_pack: cpn.parameter(type=bool, default=True, desc="for host side, whether to pack split info together"),
    hist_sub: cpn.parameter(type=bool, default=True, desc="whether to use histogram subtraction"),
    he_param: cpn.parameter(
        type=params.he_param(),
        default=params.HEParam(kind="paillier", key_length=1024),
        desc="homomorphic encryption param, support paillier, ou and mock in current version",
    ),
    train_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    output_model: cpn.json_model_output(roles=[GUEST, HOST], optional=True),
    warm_start_model: cpn.json_model_input(roles=[GUEST, HOST], optional=True),
):
    train_data = train_data.read()
    if validate_data is not None:
        validate_data = validate_data.read()

    if warm_start_model is not None:
        warm_start_model = warm_start_model.read()

    if role.is_guest:
        # initialize encrypt kit
        ctx.cipher.set_phe(ctx.device, he_param.dict())

        booster = HeteroSecureBoostGuest(
            num_trees=num_trees,
            max_depth=max_depth,
            complete_secure=complete_secure,
            learning_rate=learning_rate,
            max_bin=max_bin,
            l1=l1,
            l2=l2,
            min_impurity_split=min_impurity_split,
            min_sample_split=min_sample_split,
            min_leaf_node=min_leaf_node,
            min_child_weight=min_child_weight,
            objective=objective,
            num_class=num_class,
            gh_pack=gh_pack,
            split_info_pack=split_info_pack,
            hist_sub=hist_sub,
            top_rate=top_rate,
            other_rate=other_rate,
            goss_start_iter=goss_start_iter,
            goss=goss,
        )
        if warm_start_model:
            booster.from_model(warm_start_model)
            logger.info("sbt input model loaded, will start warmstarting")
        booster.fit(ctx, train_data, validate_data)
        # get cached train data score
        train_scores = booster.get_train_predict()
        train_scores = add_dataset_type(train_scores, consts.TRAIN_SET)
        train_output_data.write(train_scores)
        # get tree param
        tree_dict = booster.get_model()
        output_model.write(tree_dict, metadata={})

    elif role.is_host:
        booster = HeteroSecureBoostHost(
            num_trees=num_trees,
            max_depth=max_depth,
            complete_secure=complete_secure,
            max_bin=max_bin,
            hist_sub=hist_sub,
        )
        if warm_start_model is not None:
            booster.from_model(warm_start_model)
            logger.info("sbt input model loaded, will start warmstarting")
        booster.fit(ctx, train_data, validate_data)
        tree_dict = booster.get_model()
        output_model.write(tree_dict, metadata={})

    else:
        raise RuntimeError(f"Unknown role: {role}")


@hetero_secureboost.predict()
def predict(
    ctx,
    role: Role,
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    input_model: cpn.json_model_input(roles=[GUEST, HOST]),
    test_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    model_input = input_model.read()
    test_data = test_data.read()
    if role.is_guest:
        booster = HeteroSecureBoostGuest()
        booster.from_model(model_input)
        pred_table_rs = booster.predict(ctx, test_data)
        pred_table_rs = add_dataset_type(pred_table_rs, consts.TEST_SET)
        test_output_data.write(pred_table_rs)

    elif role.is_host:
        booster = HeteroSecureBoostHost()
        booster.from_model(model_input)
        booster.predict(ctx, test_data)

    else:
        raise RuntimeError(f"Unknown role: {role}")


@hetero_secureboost.cross_validation()
def cross_validation(
    ctx: Context,
    role: Role,
    cv_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    num_trees: cpn.parameter(type=params.conint(gt=0), default=3, desc="max tree num"),
    learning_rate: cpn.parameter(type=params.confloat(gt=0), default=0.3, desc="decay factor of each tree"),
    max_depth: cpn.parameter(type=params.conint(gt=0), default=3, desc="max depth of a tree"),
    complete_secure: cpn.parameter(
        type=params.conint(ge=0),
        default=0,
        desc="number of trees to use guest features only in the complete secure mode, " "0 means no complete secure",
    ),
    max_bin: cpn.parameter(type=params.conint(gt=0), default=32, desc="max bin number of feature binning"),
    objective: cpn.parameter(
        type=params.string_choice(choice=[BINARY_BCE, MULTI_CE, REGRESSION_L2]),
        default=BINARY_BCE,
        desc="objective function, available: {}".format([BINARY_BCE, MULTI_CE, REGRESSION_L2]),
    ),
    num_class: cpn.parameter(
        type=params.conint(gt=0),
        default=2,
        desc="class number of multi classification, active when objective is {}".format(MULTI_CE),
    ),
    l1: cpn.parameter(type=params.confloat(ge=0), default=0, desc="L1 regularization"),
    l2: cpn.parameter(type=params.confloat(ge=0), default=0.1, desc="L2 regularization"),
    goss: cpn.parameter(type=bool, default=False, desc="whether to use goss subsample"),
    goss_start_iter: cpn.parameter(type=params.conint(ge=0), default=5, desc="start iteration of goss subsample"),
    top_rate: cpn.parameter(type=params.confloat(gt=0, lt=1), default=0.2, desc="top rate of goss subsample"),
    other_rate: cpn.parameter(type=params.confloat(gt=0, lt=1), default=0.1, desc="other rate of goss subsample"),
    min_impurity_split: cpn.parameter(
        type=params.confloat(gt=0), default=1e-2, desc="min impurity when splitting a tree node"
    ),
    min_sample_split: cpn.parameter(type=params.conint(gt=0), default=2, desc="min sample to split a tree node"),
    min_leaf_node: cpn.parameter(type=params.conint(gt=0), default=1, desc="mininum sample contained in a leaf node"),
    min_child_weight: cpn.parameter(
        type=params.confloat(gt=0), default=1, desc="minumum hessian contained in a leaf node"
    ),
    gh_pack: cpn.parameter(type=bool, default=True, desc="whether to pack gradient and hessian together"),
    split_info_pack: cpn.parameter(type=bool, default=True, desc="for host side, whether to pack split info together"),
    hist_sub: cpn.parameter(type=bool, default=True, desc="whether to use histogram subtraction"),
    he_param: cpn.parameter(
        type=params.he_param(),
        default=params.HEParam(kind="paillier", key_length=1024),
        desc="homomorphic encryption param, support paillier, ou and mock in current version",
    ),
    cv_param: cpn.parameter(
        type=params.cv_param(),
        default=params.CVParam(n_splits=5, shuffle=False, random_state=None),
        desc="cross validation param",
    ),
    output_cv_data: cpn.parameter(type=bool, default=True, desc="whether output prediction result per cv fold"),
    cv_output_datas: cpn.dataframe_outputs(roles=[GUEST, HOST], optional=True),
):
    from fate.arch.dataframe import KFold

    kf = KFold(
        ctx, role=role, n_splits=cv_param.n_splits, shuffle=cv_param.shuffle, random_state=cv_param.random_state
    )
    i = 0
    for fold_ctx, (train_data, validate_data) in ctx.on_cross_validations.ctxs_zip(kf.split(cv_data.read())):
        logger.info(f"enter fold {i}")
        i += 1
        if role.is_guest:
            # initialize encrypt kit
            fold_ctx.cipher.set_phe(fold_ctx.device, he_param.dict())

            booster = HeteroSecureBoostGuest(
                num_trees=num_trees,
                max_depth=max_depth,
                complete_secure=complete_secure,
                learning_rate=learning_rate,
                max_bin=max_bin,
                l1=l1,
                l2=l2,
                min_impurity_split=min_impurity_split,
                min_sample_split=min_sample_split,
                min_leaf_node=min_leaf_node,
                min_child_weight=min_child_weight,
                objective=objective,
                num_class=num_class,
                gh_pack=gh_pack,
                split_info_pack=split_info_pack,
                hist_sub=hist_sub,
                goss_start_iter=goss_start_iter,
                goss=goss,
                top_rate=top_rate,
                other_rate=other_rate,
            )
            booster.fit(fold_ctx, train_data, validate_data)
            if output_cv_data:
                # train predict
                train_scores = booster.get_train_predict()
                train_scores = add_dataset_type(train_scores, consts.TRAIN_SET)
                # validate predict
                sub_ctx = fold_ctx.sub_ctx("predict_validate")
                validate_scores = booster.predict(sub_ctx, validate_data)
                validate_scores = add_dataset_type(validate_scores, consts.VALIDATE_SET)
                predict_result = DataFrame.vstack([train_scores, validate_scores])
                next(cv_output_datas).write(predict_result)

        elif role.is_host:
            booster = HeteroSecureBoostHost(
                num_trees=num_trees,
                max_depth=max_depth,
                complete_secure=complete_secure,
                max_bin=max_bin,
                hist_sub=hist_sub,
            )
            booster.fit(fold_ctx, train_data, validate_data)

            if output_cv_data:
                # validate predict
                sub_ctx = fold_ctx.sub_ctx("predict_validate")
                booster.predict(sub_ctx, validate_data)

        else:
            raise RuntimeError(f"Unknown role: {role}")
