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

from fate.arch import Context
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params


@cpn.component(roles=[GUEST, HOST, ARBITER])
def hetero_lr(ctx, role):
    ...


@hetero_lr.train()
def train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        validate_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        learning_rate_scheduler: cpn.parameter(type=params.lr_scheduler_param(),
                                               default=params.LRSchedulerParam(method="constant",
                                                                               scheduler_params={"gamma": 0.1}),
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
        train_output_metric: cpn.json_metric_output(roles=[ARBITER]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
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
        train_arbiter(ctx, max_iter, early_stop, tol, batch_size, optimizer, learning_rate_scheduler,
                      train_output_metric)


@hetero_lr.predict()
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


"""@hetero_lr.cross_validation()
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
            from fate.ml.glm.hetero_lr import HeteroLrModuleGuest

            module = HeteroLrModuleGuest(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
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
    from fate.ml.glm.hetero_lr import HeteroLrModuleGuest
    # optimizer = optimizer_factory(optimizer_param)

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLrModuleGuest(max_iter=max_iter, batch_size=batch_size,
                                     optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                     init_param=init_param, threshold=threshold)
        train_data = sub_ctx.reader(train_data).read_dataframe().data

        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe().data

        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_lr_guest", model, metadata={"threshold": threshold})

    with ctx.sub_ctx("predict") as sub_ctx:
        predict_score = module.predict(sub_ctx, validate_data)
        predict_result = transform_to_predict_result(validate_data, predict_score, module.labels,
                                                     threshold=module.threshold, is_ovr=module.ovr,
                                                     data_type="test")
        sub_ctx.writer(train_output_data).write_dataframe(predict_result)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, max_iter, batch_size,
               optimizer_param, learning_rate_param, init_param):
    from fate.ml.glm.hetero_lr import HeteroLrModuleHost

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLrModuleHost(max_iter=max_iter, batch_size=batch_size,
                                    optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                    init_param=init_param)
        train_data = sub_ctx.reader(train_data).read_dataframe().data

        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe().data

        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_lr_host", model, metadata={})
    with ctx.sub_ctx("predict") as sub_ctx:
        module.predict(sub_ctx, validate_data)


def train_arbiter(ctx, max_iter, early_stop, tol, batch_size, optimizer_param, learning_rate_scheduler,
                  train_output_metric):
    from fate.ml.glm.hetero_lr import HeteroLrModuleArbiter

    ctx.metrics.handler.register_metrics(lr_loss=ctx.writer(train_output_metric))

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLrModuleArbiter(max_iter=max_iter, early_stop=early_stop, tol=tol, batch_size=batch_size,
                                       optimizer_param=optimizer_param, learning_rate_param=learning_rate_scheduler)
        module.fit(sub_ctx)


def predict_guest(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.hetero_lr import HeteroLrModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()

        module = HeteroLrModuleGuest.from_model(model)
        # if module.threshold != 0.5:
        #    module.threshold = threshold
        test_data = sub_ctx.reader(test_data).read_dataframe()
        predict_score = module.predict(sub_ctx, test_data)
        predict_result = transform_to_predict_result(test_data, predict_score, module.labels,
                                                     threshold=module.threshold, is_ovr=module.ovr,
                                                     data_type="test")
        sub_ctx.writer(test_output_data).write_dataframe(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.hetero_lr import HeteroLrModuleHost

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        module = HeteroLrModuleHost.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
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
        df = test_data.create_dataframe(with_label=True, with_weight=False)
        pred_res = test_data.create_dataframe(with_label=False, with_weight=False)
        pred_res["predict_result"] = predict_score
        df[["predict_result",
            "predict_score",
            "predict_detail",
            "type"]] = pred_res.apply_row(lambda v: [int(v[0] > threshold),
                                                     v[0],
                                                     json.dumps({1: v[0],
                                                                 0: 1 - v[0]}),
                                                     data_type],
                                          enable_type_align_checking=False)
    return df
