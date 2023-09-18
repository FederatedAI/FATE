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
from fate.ml.glm import CoordinatedLRModuleGuest, CoordinatedLRModuleHost, CoordinatedLRModuleArbiter

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST, ARBITER], provider="fate")
def coordinated_lr(ctx, role):
    ...


@coordinated_lr.train()
def train(
    ctx: Context,
    role: Role,
    train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
    learning_rate_scheduler: cpn.parameter(
        type=params.lr_scheduler_param(),
        default=params.LRSchedulerParam(method="linear", scheduler_params={"start_factor": 1.0}),
        desc="learning rate scheduler, "
        "select method from {'step', 'linear', 'constant'}"
        "for list of configurable arguments, "
        "refer to torch.optim.lr_scheduler",
    ),
    epochs: cpn.parameter(type=params.conint(gt=0), default=20, desc="max iteration num"),
        batch_size: cpn.parameter(
            type=params.conint(ge=10),
            default=None, desc="batch size, None means full batch, otherwise should be no less than 10, default None"
        ),
        optimizer: cpn.parameter(
            type=params.optimizer_param(),
            default=params.OptimizerParam(
                method="sgd", penalty="l2", alpha=1.0, optimizer_params={"lr": 1e-2, "weight_decay": 0}
            ),
        ),
        floating_point_precision: cpn.parameter(
            type=params.conint(ge=0),
            default=23,
            desc="floating point precision, "),
        tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
        early_stop: cpn.parameter(
            type=params.string_choice(["weight_diff", "diff", "abs"]),
            default="diff",
            desc="early stopping criterion, choose from {weight_diff, diff, abs, val_metrics}",
        ),
        he_param: cpn.parameter(type=params.he_param(), default=params.HEParam(kind="paillier", key_length=1024),
                                desc="homomorphic encryption param"),
        init_param: cpn.parameter(
            type=params.init_param(),
            default=params.InitParam(method="random_uniform", fit_intercept=True, random_state=None),
            desc="Model param init setting.",
        ),
        threshold: cpn.parameter(
            type=params.confloat(ge=0.0, le=1.0), default=0.5, desc="predict threshold for binary data"
        ),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST, ARBITER]),
        warm_start_model: cpn.json_model_input(roles=[GUEST, HOST, ARBITER], optional=True),
):
    logger.info(f"enter coordinated lr train")
    # temp code start
    optimizer = optimizer.dict()
    learning_rate_scheduler = learning_rate_scheduler.dict()
    init_param = init_param.dict()
    ctx.cipher.set_phe(ctx.device, he_param.dict())

    if role.is_guest:
        train_guest(
            ctx,
            train_data,
            validate_data,
            train_output_data,
            output_model,
            epochs,
            batch_size,
            optimizer,
            learning_rate_scheduler,
            init_param,
            threshold,
            floating_point_precision,
            warm_start_model
        )
    elif role.is_host:
        train_host(
            ctx,
            train_data,
            validate_data,
            train_output_data,
            output_model,
            epochs,
            batch_size,
            optimizer,
            learning_rate_scheduler,
            init_param,
            floating_point_precision,
            warm_start_model
        )
    elif role.is_arbiter:
        train_arbiter(ctx,
                      epochs,
                      early_stop,
                      tol, batch_size,
                      optimizer,
                      learning_rate_scheduler,
                      output_model,
                      warm_start_model)


@coordinated_lr.predict()
def predict(
    ctx,
    role: Role,
    # threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5),
    test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
    input_model: cpn.json_model_input(roles=[GUEST, HOST]),
    test_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
):
    if role.is_guest:
        predict_guest(ctx, input_model, test_data, test_output_data)
    if role.is_host:
        predict_host(ctx, input_model, test_data, test_output_data)


@coordinated_lr.cross_validation()
def cross_validation(
        ctx: Context,
        role: Role,
        cv_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        learning_rate_scheduler: cpn.parameter(
            type=params.lr_scheduler_param(),
            default=params.LRSchedulerParam(method="linear", scheduler_params={"start_factor": 1.0}),
            desc="learning rate scheduler, "
                 "select method from {'step', 'linear', 'constant'}"
                 "for list of configurable arguments, "
                 "refer to torch.optim.lr_scheduler",
        ),
        epochs: cpn.parameter(type=params.conint(gt=0), default=20, desc="max iteration num"),
        batch_size: cpn.parameter(
            type=params.conint(ge=10),
            default=None, desc="batch size, None means full batch, otherwise should be no less than 10, default None"
        ),
        optimizer: cpn.parameter(
            type=params.optimizer_param(),
            default=params.OptimizerParam(
                method="sgd", penalty="l2", alpha=1.0, optimizer_params={"lr": 1e-2, "weight_decay": 0}
            ),
        ),
        tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
        early_stop: cpn.parameter(
            type=params.string_choice(["weight_diff", "diff", "abs"]),
            default="diff",
            desc="early stopping criterion, choose from {weight_diff, diff, abs, val_metrics}",
        ),
        init_param: cpn.parameter(
            type=params.init_param(),
            default=params.InitParam(method="random_uniform", fit_intercept=True, random_state=None),
            desc="Model param init setting.",
        ),
        threshold: cpn.parameter(
            type=params.confloat(ge=0.0, le=1.0), default=0.5, desc="predict threshold for binary data"
        ),
        he_param: cpn.parameter(type=params.he_param(), default=params.HEParam(kind="paillier", key_length=1024),
                                desc="homomorphic encryption param"),
        floating_point_precision: cpn.parameter(
            type=params.conint(ge=0),
            default=23,
            desc="floating point precision, "),
        cv_param: cpn.parameter(type=params.cv_param(),
                                default=params.CVParam(n_splits=5, shuffle=False, random_state=None),
                                desc="cross validation param"),
        metrics: cpn.parameter(type=params.metrics_param(), default=["auc"]),
        output_cv_data: cpn.parameter(type=bool, default=True, desc="whether output prediction result per cv fold"),
        cv_output_datas: cpn.dataframe_outputs(roles=[GUEST, HOST], optional=True),
):
    optimizer = optimizer.dict()
    learning_rate_scheduler = learning_rate_scheduler.dict()
    init_param = init_param.dict()
    ctx.cipher.set_phe(ctx.device, he_param.dict())

    if role.is_arbiter:
        i = 0
        for fold_ctx, _ in ctx.on_cross_validations.ctxs_zip(zip(range(cv_param.n_splits))):
            logger.info(f"enter fold {i}")
            module = CoordinatedLRModuleArbiter(
                epochs=epochs,
                early_stop=early_stop,
                tol=tol,
                batch_size=batch_size,
                optimizer_param=optimizer,
                learning_rate_param=learning_rate_scheduler,
            )
            module.fit(fold_ctx)
            i += 1
        return

    from fate.arch.dataframe import KFold
    kf = KFold(ctx, role=role, n_splits=cv_param.n_splits, shuffle=cv_param.shuffle, random_state=cv_param.random_state)
    i = 0
    for fold_ctx, (train_data, validate_data) in ctx.on_cross_validations.ctxs_zip(kf.split(cv_data.read())):
        logger.info(f"enter fold {i}")
        if role.is_guest:
            module = CoordinatedLRModuleGuest(
                epochs=epochs,
                batch_size=batch_size,
                optimizer_param=optimizer,
                learning_rate_param=learning_rate_scheduler,
                init_param=init_param,
                threshold=threshold,
                floating_point_precision=floating_point_precision,
            )
            module.fit(fold_ctx, train_data, validate_data)
            if output_cv_data:
                sub_ctx = fold_ctx.sub_ctx("predict_train")
                predict_df = module.predict(sub_ctx, train_data)
                """train_predict_result = transform_to_predict_result(
                    train_data, predict_score, module.labels, threshold=module.threshold, is_ovr=module.ovr,
                    data_type="train"
                )"""
                train_predict_result = tools.add_dataset_type(predict_df, consts.TRAIN_SET)
                sub_ctx = fold_ctx.sub_ctx("predict_validate")
                predict_df = module.predict(sub_ctx, validate_data)
                """validate_predict_result = transform_to_predict_result(
                    validate_data, predict_score, module.labels, threshold=module.threshold, is_ovr=module.ovr,
                    data_type="predict"
                )"""
                validate_predict_result = tools.add_dataset_type(predict_df, consts.VALIDATE_SET)
                predict_result = DataFrame.vstack([train_predict_result, validate_predict_result])
                next(cv_output_datas).write(df=predict_result)

            # evaluation = evaluate(predicted)
        elif role.is_host:
            module = CoordinatedLRModuleHost(
                epochs=epochs,
                batch_size=batch_size,
                optimizer_param=optimizer,
                learning_rate_param=learning_rate_scheduler,
                init_param=init_param,
                floating_point_precision=floating_point_precision
            )
            module.fit(fold_ctx, train_data, validate_data)
            if output_cv_data:
                sub_ctx = fold_ctx.sub_ctx("predict_train")
                module.predict(sub_ctx, train_data)
                sub_ctx = fold_ctx.sub_ctx("predict_validate")
                module.predict(sub_ctx, validate_data)
        i += 1


def train_guest(
    ctx,
        train_data,
        validate_data,
        train_output_data,
        output_model,
        epochs,
        batch_size,
        optimizer_param,
        learning_rate_param,
        init_param,
        threshold,
        floating_point_precision,
        input_model
):
    if input_model is not None:
        logger.info(f"warm start model provided")
        model = input_model.read()
        module = CoordinatedLRModuleGuest.from_model(model)
        module.set_epochs(epochs)
        module.set_batch_size(batch_size)

    else:
        module = CoordinatedLRModuleGuest(
            epochs=epochs,
            batch_size=batch_size,
            optimizer_param=optimizer_param,
            learning_rate_param=learning_rate_param,
            init_param=init_param,
            threshold=threshold,
            floating_point_precision=floating_point_precision
        )
    # optimizer = optimizer_factory(optimizer_param)
    logger.info(f"coordinated lr guest start train")
    sub_ctx = ctx.sub_ctx("train")
    train_data = train_data.read()

    if validate_data is not None:
        logger.info(f"validate data provided")
        validate_data = validate_data.read()

    module.fit(sub_ctx, train_data, validate_data)
    model = module.get_model()
    output_model.write(model, metadata={})

    sub_ctx = ctx.sub_ctx("predict")

    predict_df = module.predict(sub_ctx, train_data)
    """predict_result = transform_to_predict_result(
        train_data, predict_score, module.labels, threshold=module.threshold, is_ovr=module.ovr, data_type="train"
    )"""
    predict_result = tools.add_dataset_type(predict_df, consts.TRAIN_SET)
    if validate_data is not None:
        sub_ctx = ctx.sub_ctx("validate_predict")
        predict_df = module.predict(sub_ctx, validate_data)
        """validate_predict_result = transform_to_predict_result(
            validate_data,
            predict_score,
            module.labels,
            threshold=module.threshold,
            is_ovr=module.ovr,
            data_type="validate",
        )"""
        validate_predict_result = tools.add_dataset_type(predict_df, consts.VALIDATE_SET)
        predict_result = DataFrame.vstack([predict_result, validate_predict_result])
    train_output_data.write(predict_result)


def train_host(
        ctx,
        train_data,
        validate_data,
        train_output_data,
        output_model,
        epochs,
        batch_size,
        optimizer_param,
        learning_rate_param,
        init_param,
        floating_point_precision,
        input_model
):
    if input_model is not None:
        logger.info(f"warm start model provided")
        model = input_model.read()
        module = CoordinatedLRModuleHost.from_model(model)
        module.set_epochs(epochs)
        module.set_batch_size(batch_size)
    else:
        module = CoordinatedLRModuleHost(
            epochs=epochs,
            batch_size=batch_size,
            optimizer_param=optimizer_param,
            learning_rate_param=learning_rate_param,
            init_param=init_param,
            floating_point_precision=floating_point_precision
        )
    logger.info(f"coordinated lr host start train")
    sub_ctx = ctx.sub_ctx("train")
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
        sub_ctx = ctx.sub_ctx("validate_predict")
        module.predict(sub_ctx, validate_data)


def train_arbiter(ctx, epochs, early_stop, tol, batch_size, optimizer_param, learning_rate_scheduler,
                  output_model, input_model):
    if input_model is not None:
        logger.info(f"warm start model provided")
        model = input_model.read()
        module = CoordinatedLRModuleArbiter.from_model(model)
        module.set_epochs(epochs)
        module.set_batch_size(batch_size)
    else:
        module = CoordinatedLRModuleArbiter(
            epochs=epochs,
            early_stop=early_stop,
            tol=tol,
            batch_size=batch_size,
            optimizer_param=optimizer_param,
            learning_rate_param=learning_rate_scheduler,
        )
    logger.info(f"coordinated lr arbiter start train")
    sub_ctx = ctx.sub_ctx("train")
    module.fit(sub_ctx)
    model = module.get_model()
    output_model.write(model, metadata={})


def predict_guest(ctx, input_model, test_data, test_output_data):
    logger.info(f"coordinated lr guest start predict")
    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    module = CoordinatedLRModuleGuest.from_model(model)
    # if module.threshold != 0.5:
    #    module.threshold = threshold
    test_data = test_data.read()
    predict_df = module.predict(sub_ctx, test_data)
    """predict_result = transform_to_predict_result(
        test_data, predict_score, module.labels, threshold=module.threshold, is_ovr=module.ovr, data_type="test"
    )"""
    predict_result = tools.add_dataset_type(predict_df, consts.TEST_SET)
    test_output_data.write(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    logger.info(f"coordinated lr host start predict")
    sub_ctx = ctx.sub_ctx("predict")
    model = input_model.read()
    module = CoordinatedLRModuleHost.from_model(model)
    test_data = test_data.read()
    module.predict(sub_ctx, test_data)
