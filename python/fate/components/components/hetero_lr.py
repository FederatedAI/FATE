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
from fate.components import (
    ARBITER,
    GUEST,
    HOST,
    DatasetArtifact,
    Input,
    LossMetrics,
    ModelArtifact,
    Output,
    Role,
    cpn,
    params,
)


@cpn.component(roles=[GUEST, HOST, ARBITER])
def hetero_lr(ctx, role):
    ...


@hetero_lr.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST], desc="training data")
@cpn.artifact("validate_data", type=Input[DatasetArtifact], optional=True, roles=[GUEST, HOST], desc="validation data")
@cpn.parameter("max_iter", type=params.conint(gt=0), default=20, desc="max iteration num")
@cpn.parameter("early_stop", type=params.string_choice(["weight_diff", "diff", "abs"]), default="diff",
               desc="early stopping criterion, choose from {weight_diff, diff, abs, val_metrics}")
@cpn.parameter("tol", type=params.confloat(ge=0), default=1e-4)
@cpn.parameter(
    "batch_size", type=params.conint(ge=-1), default=-1, desc="batch size, value less or equals to 0 means full batch"
)
@cpn.parameter(
    "optimizer", type=params.optimizer_param(),
    default=params.OptimizerParam(method="sgd", penalty='l2', alpha=1.0,
                                  optimizer_params={"lr": 1e-2, "weight_decay": 0}),
    desc="optimizer, select method from {'sgd', 'nesterov_momentum_sgd', 'adam', 'rmsprop', 'adagrad', 'sqn'} "
         "for list of configurable arguments, refer to torch.optim"
)
@cpn.parameter(
    "learning_rate_scheduler", type=params.lr_scheduler_param(),
    default=params.LRSchedulerParam(method="constant", scheduler_params={"gamma": 0.1}),
    desc="learning rate scheduler, select method from {'step', 'linear', 'constant'}"
         "for list of configurable arguments, refer to torch.optim.lr_scheduler"
)
@cpn.parameter("init_param", type=params.init_param(),
               default=params.InitParam(method='zeros', fit_intercept=True),
               desc="Model param init setting.")
@cpn.parameter("threshold", type=params.confloat(ge=0.0, le=1.0), default=0.5,
               desc="predict threshold for binary data")
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("train_output_metric", type=Output[LossMetrics], roles=[ARBITER])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def train(
        ctx,
        role: Role,
        train_data,
        validate_data,
        max_iter,
        early_stop,
        tol,
        batch_size,
        optimizer,
        learning_rate_scheduler,
        init_param,
        threshold,
        train_output_data,
        train_output_metric,
        output_model,
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
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.parameter("threshold", type=params.confloat(ge=0.0, le=1.0), default=0.5,
               desc="predict threshold for binary data")
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def predict(
        ctx,
        role: Role,
        test_data,
        input_model,
        threshold,
        test_output_data,
):
    if role.is_guest:
        predict_guest(ctx, input_model, test_data, test_output_data, threshold)
    if role.is_host:
        predict_host(ctx, input_model, test_data, test_output_data)


def train_guest(ctx, train_data, validate_data, train_output_data, output_model, max_iter,
                batch_size, optimizer_param, learning_rate_param, init_param, threshold):
    from fate.ml.glm.hetero_lr import HeteroLrModuleGuest
    # optimizer = optimizer_factory(optimizer_param)

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLrModuleGuest(max_iter=max_iter, batch_size=batch_size,
                                     optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                     init_param=init_param, threshold=threshold)
        train_data = sub_ctx.reader(train_data).read_dataframe()

        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        # temp code start
        train_data = train_data.data
        # temp code end
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_lr_guest", model, metadata={"threshold": threshold})

    with ctx.sub_ctx("predict") as sub_ctx:
        predict_score = module.predict(sub_ctx, validate_data)
        predict_result = validate_data.data.transform_to_predict_result(predict_score)
        sub_ctx.writer(train_output_data).write_dataframe(predict_result)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, max_iter, batch_size,
               optimizer_param, learning_rate_param, init_param):
    from fate.ml.glm.hetero_lr import HeteroLrModuleHost

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLrModuleHost(max_iter=max_iter, batch_size=batch_size,
                                    optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                    init_param=init_param)
        train_data = sub_ctx.reader(train_data).read_dataframe()

        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        # temp code start
        train_data = train_data.data
        # temp code end
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


def predict_guest(ctx, input_model, test_data, test_output_data, threshold):
    from fate.ml.glm.hetero_lr import HeteroLrModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()

        module = HeteroLrModuleGuest.from_model(model)
        if threshold != 0.5:
            module.threshold = threshold
        test_data = sub_ctx.reader(test_data).read_dataframe()
        predict_score = module.predict(sub_ctx, test_data)
        predict_result = test_data.data.transform_to_predict_result(predict_score, data_type="predict")
        sub_ctx.writer(test_output_data).write_dataframe(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.hetero_lr import HeteroLrModuleHost

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        module = HeteroLrModuleHost.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        module.predict(sub_ctx, test_data)
