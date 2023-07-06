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

import json
import logging

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST, ARBITER])
def coordinated_lr(ctx, role):
    ...


@coordinated_lr.train()
def train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
        learning_rate_scheduler: cpn.parameter(type=params.lr_scheduler_param(),
                                               default=params.LRSchedulerParam(method="linear",
                                                                               scheduler_params={"start_factor": 1.0}),
                                               desc="learning rate scheduler, "
                                                    "select method from {'step', 'linear', 'constant'}"
                                                    "for list of configurable arguments, "
                                                    "refer to torch.optim.lr_scheduler"),
        max_iter: cpn.parameter(type=params.conint(gt=0), default=20,
                                desc="max iteration num"),
        batch_size: cpn.parameter(type=params.conint(ge=-1), default=100,
                                  desc="batch size, "
                                       "value less or equals to 0 means full batch"),
        optimizer: cpn.parameter(type=params.optimizer_param(),
                                 default=params.OptimizerParam(method="sgd", penalty='l2', alpha=1.0,
                                                               optimizer_params={"lr": 1e-2, "weight_decay": 0})),
        tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
        early_stop: cpn.parameter(type=params.string_choice(["weight_diff", "diff", "abs"]), default="diff",
                                  desc="early stopping criterion, choose from {weight_diff, diff, abs, val_metrics}"),
        init_param: cpn.parameter(type=params.init_param(),
                                  default=params.InitParam(method='zeros', fit_intercept=True),
                                  desc="Model param init setting."),
        threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5,
                                 desc="predict threshold for binary data"),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
    logger.info(f"enter coordinated lr train")
    # temp code start
    optimizer = optimizer.dict()
    learning_rate_scheduler = learning_rate_scheduler.dict()
    init_param = init_param.dict()
    # temp code end
    if role.is_guest:
        train_guest(
            ctx, train_data, validate_data, train_output_data, output_model, max_iter,
            batch_size, optimizer, learning_rate_scheduler, init_param, threshold
        )
    elif role.is_host:
        train_host(
            ctx, train_data, validate_data, train_output_data, output_model, max_iter,
            batch_size, optimizer, learning_rate_scheduler, init_param
        )
    elif role.is_arbiter:
        train_arbiter(ctx, max_iter, early_stop, tol, batch_size, optimizer, learning_rate_scheduler)


@coordinated_lr.predict()
def predict(
        ctx,
        role: Role,
        # threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5),
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST])
):
    if role.is_guest:
        predict_guest(ctx, input_model, test_data, test_output_data)
    if role.is_host:
        predict_host(ctx, input_model, test_data, test_output_data)


"""@coordinated_lr.cross_validation()
def cross_validation(
    ctx: Context,
    role: Role,
    data: cpn.dataframe_input(roles=[GUEST, HOST]),
    num_fold: cpn.parameter(type=params.conint(ge=2), desc="num cross validation fold"),
    learning_rate: cpn.parameter(type=params.learning_rate_param(), default=0.1, desc="learning rate"),
    max_iter: cpn.parameter(type=params.conint(gt=0), default=100, desc="max iteration num"),
    batch_size: cpn.parameter(
        type=params.conint(gt=0), default=100, desc="batch size, value less or equals to 0 means full batch"
    ),
):
    cv_ctx = ctx.on_cross_validations
    data = ctx.reader(data).read_dataframe()
    # TODO: split data
    for i, fold_ctx in cv_ctx.ctxs_range(num_fold):
        if role.is_guest:
            from fate.ml.glm.coordinated_lr import CoordinatedLRModuleGuest

            module = CoordinatedLRModuleGuest(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
            train_data, validate_data = split_dataframe(data, num_fold, i)
            module.fit(fold_ctx, train_data)
            predicted = module.predict(fold_ctx, validate_data)
            evaluation = evaluate(predicted)
        elif role.is_host:
            ...
        elif role.is_arbiter:
            ...
"""


def train_guest(ctx, train_data, validate_data, train_output_data, output_model, max_iter,
                batch_size, optimizer_param, learning_rate_param, init_param, threshold):
    # optimizer = optimizer_factory(optimizer_param)
    logger.info(f"coordinated lr guest start train")
    from fate.ml.glm import CoordinatedLRModuleGuest
    sub_ctx = ctx.sub_ctx("train")
    module = CoordinatedLRModuleGuest(max_iter=max_iter, batch_size=batch_size,
                                      optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                      init_param=init_param, threshold=threshold)
    train_data = train_data.read()

    if validate_data is not None:
        logger.info(f"validate data provided")
        validate_data = validate_data.read()

    module.fit(sub_ctx, train_data, validate_data)
    model = module.get_model()
    output_model.write(model, metadata={"threshold": threshold})

    sub_ctx = ctx.sub_ctx("predict")

    predict_score = module.predict(sub_ctx, train_data)
    predict_result = transform_to_predict_result(train_data, predict_score, module.labels,
                                                 threshold=module.threshold, is_ovr=module.ovr,
                                                 data_type="train")
    if validate_data is not None:
        predict_score = module.predict(sub_ctx, validate_data)
        validate_predict_result = transform_to_predict_result(validate_data, predict_score, module.labels,
                                                              threshold=module.threshold, is_ovr=module.ovr,
                                                              data_type="validate")
        predict_result = DataFrame.vstack([predict_result, validate_predict_result])
    train_output_data.write(predict_result)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, max_iter, batch_size,
               optimizer_param, learning_rate_param, init_param):
    logger.info(f"coordinated lr host start train")
    from fate.ml.glm import CoordinatedLRModuleHost
    sub_ctx = ctx.sub_ctx("train")
    module = CoordinatedLRModuleHost(max_iter=max_iter, batch_size=batch_size,
                                     optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                     init_param=init_param)
    train_data = train_data.read()

    if validate_data is not None:
        logger.info(f"validate data provided")
        validate_data = validate_data.read()

    module.fit(sub_ctx, train_data, validate_data)
    model = module.get_model()
    output_model.write(model, metadata={})
    sub_ctx = ctx.sub_ctx("predict")
    module.predict(sub_ctx, train_data)
    if validate_data is not None:
        module.predict(sub_ctx, validate_data)


def train_arbiter(ctx, max_iter, early_stop, tol, batch_size, optimizer_param, learning_rate_scheduler):
    logger.info(f"coordinated lr arbiter start train")
    from fate.ml.glm import CoordinatedLRModuleArbiter
    sub_ctx = ctx.sub_ctx("train")
    module = CoordinatedLRModuleArbiter(max_iter=max_iter, early_stop=early_stop, tol=tol, batch_size=batch_size,
                                        optimizer_param=optimizer_param,
                                        learning_rate_param=learning_rate_scheduler)
    module.fit(sub_ctx)


def predict_guest(ctx, input_model, test_data, test_output_data):
    logger.info(f"coordinated lr guest start predict")
    from fate.ml.glm import CoordinatedLRModuleGuest

    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    module = CoordinatedLRModuleGuest.from_model(model)
    # if module.threshold != 0.5:
    #    module.threshold = threshold
    test_data = test_data.read()
    predict_score = module.predict(sub_ctx, test_data)
    predict_result = transform_to_predict_result(test_data, predict_score, module.labels,
                                                 threshold=module.threshold, is_ovr=module.ovr,
                                                 data_type="test")
    test_output_data.write(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    logger.info(f"coordinated lr host start predict")
    from fate.ml.glm import CoordinatedLRModuleHost

    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    module = CoordinatedLRModuleHost.from_model(model)
    test_data = test_data.read()
    module.predict(sub_ctx, test_data)


def transform_to_predict_result(test_data, predict_score, labels, threshold=0.5, is_ovr=False, data_type="test"):
    if is_ovr:
        df = test_data.create_dataframe(with_label=True, with_weight=False)
        df[["predict_result", "predict_score", "predict_detail"]] = predict_score.apply_row(
            lambda v: [v.argmax(),
                       v[v.argmax()],
                       json.dumps({label: v[label] for label in labels}),
                       data_type],
            enable_type_align_checking=False)
    else:
        df = test_data.create_frame(with_label=True, with_weight=False)
        pred_res = test_data.create_frame(with_label=False, with_weight=False)
        pred_res["predict_result"] = predict_score
        # logger.info(f"predict score: {list(predict_score.shardings._data.collect())}")
        df[["predict_result",
            "predict_score",
            "predict_detail",
            "type"]] = pred_res.apply_row(lambda v: [int(v[0] > threshold),
                                                     v[0],
                                                     json.dumps({1: v[0],
                                                                 0: 1 - v[0]}),
                                                     data_type],
                                          enable_type_align_checking=False)
    # temp code start
    df.rename(label_name="label")
    # temp code end
    return df
