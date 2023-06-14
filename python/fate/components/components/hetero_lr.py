#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
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

from fate.arch import Context
from fate.components import ARBITER, GUEST, HOST, Role, params
from fate.components.core import artifacts, component, parameter


@component(roles=[GUEST, HOST, ARBITER])
def hetero_lr(ctx, role):
    ...


@hetero_lr.train()
@artifacts.dataframe_input("train_data", roles=[GUEST, HOST], desc="training data")
@artifacts.dataframe_input("validate_data", optional=True, roles=[GUEST, HOST], desc="validation data")
@parameter("learning_rate", type=params.learning_rate_param(), default=0.1, desc="learning rate")
@parameter("max_iter", type=params.conint(gt=0), default=100, desc="max iteration num")
@parameter(
    "batch_size", type=params.conint(gt=0), default=100, desc="batch size, value less or equals to 0 means full batch"
)
@artifacts.dataframe_output("train_output_data", roles=[GUEST, HOST])
@artifacts.json_metric_output("train_output_metric", roles=[ARBITER])
@artifacts.json_model_output("output_model", roles=[GUEST, HOST])
def train(
    ctx,
    role: Role,
    train_data,
    validate_data,
    learning_rate,
    max_iter,
    batch_size,
    train_output_data,
    train_output_metric,
    output_model,
):
    if role.is_guest:
        train_guest(
            ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size
        )
    elif role.is_host:
        train_host(
            ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size
        )
    elif role.is_arbiter:
        train_arbiter(ctx, max_iter, batch_size, train_output_metric)


@hetero_lr.predict()
@artifacts.json_model_input("input_model", roles=[GUEST, HOST])
@artifacts.dataframe_input("test_data", optional=False, roles=[GUEST, HOST])
@artifacts.dataframe_output("test_output_data", roles=[GUEST, HOST])
def predict(
    ctx,
    role: Role,
    test_data,
    input_model,
    test_output_data,
):
    if role.is_guest:
        predict_guest(ctx, input_model, test_data, test_output_data)
    if role.is_host:
        predict_host(ctx, input_model, test_data, test_output_data)


@hetero_lr.cross_validation()
@artifacts.dataframe_input("data", optional=False, roles=[GUEST, HOST])
@parameter("num_fold", type=params.conint(ge=2), desc="num cross validation fold")
@parameter("learning_rate", type=params.learning_rate_param(), default=0.1, desc="learning rate")
@parameter("max_iter", type=params.conint(gt=0), default=100, desc="max iteration num")
@parameter(
    "batch_size", type=params.conint(gt=0), default=100, desc="batch size, value less or equals to 0 means full batch"
)
def cross_validation(
    ctx: Context,
    role: Role,
    data,
    num_fold,
    learning_rate,
    max_iter,
    batch_size,
):
    cv_ctx = ctx.on_cross_validations
    data = ctx.reader(data).read_dataframe()
    # TODO: split data
    for i, fold_ctx in cv_ctx.ctxs_range(num_fold):
        if role.is_guest:
            from fate.ml.lr.guest import LrModuleGuest

            module = LrModuleGuest(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
            train_data, validate_data = split_dataframe(data, num_fold, i)
            module.fit(fold_ctx, train_data)
            predicted = module.predict(fold_ctx, validate_data)
            evaluation = evaluate(predicted)
        elif role.is_host:
            ...
        elif role.is_arbiter:
            ...


#
#
def train_guest(ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size):
    from fate.ml.lr.guest import LrModuleGuest

    with ctx.sub_ctx("train") as sub_ctx:
        module = LrModuleGuest(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        train_data = sub_ctx.reader(train_data).read_dataframe()
        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_lr_guest", model, metadata={})

    with ctx.sub_ctx("predict") as sub_ctx:
        predict_score = module.predict(sub_ctx, validate_data)
        predict_result = validate_data.data.transform_to_predict_result(predict_score)
        sub_ctx.writer(train_output_data).write_dataframe(predict_result)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size):
    from fate.ml.lr.host import LrModuleHost

    with ctx.sub_ctx("train") as sub_ctx:
        module = LrModuleHost(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        train_data = sub_ctx.reader(train_data).read_dataframe()
        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_lr_host", model, metadata={})
    with ctx.sub_ctx("predict") as sub_ctx:
        module.predict(sub_ctx, validate_data)


def train_arbiter(ctx, max_iter, batch_size, train_output_metric):
    from fate.ml.lr.arbiter import LrModuleArbiter

    ctx.metrics.handler.register_metrics(lr_loss=ctx.writer(train_output_metric))

    with ctx.sub_ctx("train") as sub_ctx:
        module = LrModuleArbiter(max_iter=max_iter, batch_size=batch_size)
        module.fit(sub_ctx)


def predict_guest(ctx, input_model, test_data, test_output_data):
    from fate.ml.lr.guest import LrModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        module = LrModuleGuest.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        predict_score = module.predict(sub_ctx, test_data)
        predict_result = test_data.data.transform_to_predict_result(predict_score, data_type="predict")
        sub_ctx.writer(test_output_data).write_dataframe(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    from fate.ml.lr.host import LrModuleHost

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        module = LrModuleHost.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        module.predict(sub_ctx, test_data)
