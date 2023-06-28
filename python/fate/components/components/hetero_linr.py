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

from fate.arch import Context
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params


@cpn.component(roles=[GUEST, HOST, ARBITER])
def hetero_linr(ctx, role):
    ...


@hetero_linr.train()
def train(
        ctx: Context,
        role: Role,
        train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        validate_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        learning_rate_scheduler: cpn.parameter(type=params.lr_scheduler_param(),
                                               default=params.LRSchedulerParam(method="constant"),
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
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        train_output_metric: cpn.json_metric_output(roles=[ARBITER]),
        output_model: cpn.json_model_output(roles=[GUEST, HOST]),
):
    if role.is_guest:
        train_guest(
            ctx, train_data, validate_data, train_output_data, output_model, max_iter,
            batch_size, optimizer, learning_rate_scheduler, init_param
        )
    elif role.is_host:
        train_host(
            ctx, train_data, validate_data, train_output_data, output_model, max_iter,
            batch_size, optimizer, learning_rate_scheduler, init_param
        )
    elif role.is_arbiter:
        train_arbiter(ctx, max_iter, early_stop, tol, batch_size, optimizer, learning_rate_scheduler,
                      train_output_metric)


@hetero_linr.predict()
def predict(
        ctx,
        role: Role,
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST])
):
    if role.is_guest:
        predict_guest(ctx, input_model, test_data, test_output_data)
    if role.is_host:
        predict_host(ctx, input_model, test_data, test_output_data)


def train_guest(ctx, train_data, validate_data, train_output_data, output_model, max_iter,
                batch_size, optimizer_param, learning_rate_param, init_param):
    from fate.ml.glm.hetero_linr import HeteroLinRModuleGuest
    # optimizer = optimizer_factory(optimizer_param)

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLinRModuleGuest(max_iter=max_iter, batch_size=batch_size,
                                       optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                       init_param=init_param)
        train_data = sub_ctx.reader(train_data).read_dataframe()
        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_linr_guest", model, metadata={})

    with ctx.sub_ctx("predict") as sub_ctx:
        predict_score = module.predict(sub_ctx, validate_data)
        predict_result = validate_data.data.transform_to_predict_result(predict_score)
        sub_ctx.writer(train_output_data).write_dataframe(predict_result)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, max_iter, batch_size,
               optimizer_param, learning_rate_param, init_param):
    from fate.ml.glm.hetero_linr import HeteroLinRModuleHost
    # optimizer = optimizer_factory(optimizer_param)

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLinRModuleHost(max_iter=max_iter, batch_size=batch_size,
                                      optimizer_param=optimizer_param, learning_rate_param=learning_rate_param,
                                      init_param=init_param)
        train_data = sub_ctx.reader(train_data).read_dataframe()
        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        with output_model as model_writer:
            model_writer.write_model("hetero_linr_host", model, metadata={})
    with ctx.sub_ctx("predict") as sub_ctx:
        module.predict(sub_ctx, validate_data)


def train_arbiter(ctx, max_iter, early_stop, tol, batch_size, optimizer_param,
                  learning_rate_param, train_output_metric):
    from fate.ml.glm.hetero_linr import HeteroLinRModuleArbiter

    ctx.metrics.handler.register_metrics(linr_loss=ctx.writer(train_output_metric))

    with ctx.sub_ctx("train") as sub_ctx:
        module = HeteroLinRModuleArbiter(max_iter=max_iter, early_stop=early_stop, tol=tol, batch_size=batch_size,
                                         optimizer_param=optimizer_param, learning_rate_param=learning_rate_param)
        module.fit(sub_ctx)


def predict_guest(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.hetero_linr import HeteroLinRModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()

        module = HeteroLinRModuleGuest.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        predict_score = module.predict(sub_ctx, test_data)
        predict_result = test_data.data.transform_to_predict_result(predict_score, data_type="predict")
        sub_ctx.writer(test_output_data).write_dataframe(predict_result)


def predict_host(ctx, input_model, test_data, test_output_data):
    from fate.ml.glm.hetero_linr import HeteroLinRModuleHost

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        module = HeteroLinRModuleHost.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        module.predict(sub_ctx, test_data)
