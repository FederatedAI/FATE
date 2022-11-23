from fate.components import cpn
from fate.components.spec import (
    DatasetArtifact,
    DatasetArtifacts,
    Input,
    MetricArtifact,
    ModelArtifact,
    Output,
    roles,
    stages,
)

# @cpn.component(...)
# def lr():
#     ...

# @lr.stage(...)
# def train():
#     ...

# @lr.stage(...)
# def predict():
#     ...


@cpn.component(roles=roles.get_all(), provider="fate", version="2.0.0.alpha")
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "validate_data", type=Input[DatasetArtifact], optional=True, roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN]
)
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.PREDICT])
@cpn.artifact(
    "test_data", type=Input[DatasetArtifact], optional=False, roles=[roles.GUEST, roles.HOST], stages=[stages.PREDICT]
)
@cpn.parameter("learning_rate", type=float, default=0.1)
@cpn.parameter("max_iter", type=int, default=100)
@cpn.parameter("batch_size", type=int, default=100)
@cpn.artifact(
    "train_output_data", type=Output[DatasetArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN]
)
@cpn.artifact("train_output_metric", type=Output[MetricArtifact], roles=[roles.ARBITER], stages=[stages.TRAIN])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "test_output_data", type=Output[DatasetArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.PREDICT]
)
def hetero_lr(
    ctx,
    role,
    stage,
    train_data,
    validate_data,
    test_data,
    input_model,
    learning_rate,
    max_iter,
    batch_size,
    train_output_data,
    train_output_metric,
    output_model,
    test_output_data,
):
    if stage == "train":
        if role == "guest":
            train_guest(
                ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size
            )
        elif role == "host":
            train_host(
                ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size
            )
        elif role == "arbiter":
            train_arbiter(ctx, max_iter, train_output_metric)
    elif stage == "predict":
        if role == "guest":
            predict_guest(ctx, input_model, test_data, test_output_data)
        if role == "host":
            predict_host(ctx, input_model, test_data, test_output_data)


def train_guest(ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size):
    from fate.ml.lr.guest import LrModuleGuest

    with ctx.sub_ctx("train") as sub_ctx:
        module = LrModuleGuest(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        train_data = sub_ctx.reader(train_data).read_dataframe()
        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        sub_ctx.writer(output_model).write_model(model)
    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = module.predict(sub_ctx, validate_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def train_host(ctx, train_data, validate_data, train_output_data, output_model, max_iter, learning_rate, batch_size):
    from fate.ml.lr.host import LrModuleHost

    with ctx.sub_ctx("train") as sub_ctx:
        module = LrModuleHost(max_iter=max_iter, learning_rate=learning_rate, batch_size=batch_size)
        train_data = sub_ctx.reader(train_data).read_dataframe()
        if validate_data is not None:
            validate_data = sub_ctx.reader(validate_data).read_dataframe()
        module.fit(sub_ctx, train_data, validate_data)
        model = module.get_model()
        sub_ctx.writer(output_model).write_model(model)
    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = module.predict(sub_ctx, validate_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def train_arbiter(ctx, max_iter, train_output_metric):
    from fate.ml.lr.arbiter import LrModuleArbiter

    with ctx.sub_ctx("train") as sub_ctx:
        module = LrModuleArbiter(max_iter=max_iter)
        module.fit(sub_ctx)
        for metric in module.get_metrics():
            sub_ctx.writer(train_output_metric).write_metric(metric)


def predict_guest(ctx, input_model, test_data, test_output_data):
    from fate.ml.lr.guest import LrModuleGuest

    with ctx.sub_ctx("predict") as sub_ctx:
        model = sub_ctx.reader(input_model).read_model()
        module = LrModuleGuest.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        output_data = module.predict(sub_ctx, test_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)


def predict_host(ctx, input_model, test_data, test_output_data):
    from fate.ml.lr.host import LrModuleHost

    with ctx.sub_ctx("predict") as sub_ctx:
        model = sub_ctx.reader(input_model).read_model()
        module = LrModuleHost.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe()
        output_data = module.predict(sub_ctx, test_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)
