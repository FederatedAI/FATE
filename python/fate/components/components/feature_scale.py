from fate.components import cpn
from fate.components.spec import (
    DatasetArtifact,
    Input,
    MetricArtifact,
    ModelArtifact,
    Output,
    roles,
)
from fate.ml.feature_scale import FeatureScale


@cpn.component(roles=[roles.GUEST, roles.HOST], provider="fate", version="2.0.0.alpha")
def feature_scale(ctx, role):
    ...


@feature_scale.stage("train")
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[roles.GUEST, roles.HOST])
@cpn.parameter("method", type=str, default="standard", optional=False)
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[roles.GUEST, roles.HOST])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[roles.GUEST, roles.HOST])
def feature_scale_train(
    ctx,
    role,
    train_data,
    method,
    train_output_data,
    output_model,
):
    train(
        ctx, train_data, train_output_data, output_model, method
    )


@feature_scale.stage("predict")
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[roles.GUEST, roles.HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[roles.GUEST, roles.HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[roles.GUEST, roles.HOST])
def feature_scale_predict(
    ctx,
    role,
    test_data,
    input_model,
    test_output_data,
):
    predict(ctx, input_model, test_data, test_output_data)


def train(ctx, train_data, train_output_data, output_model, method):
    scaler = FeatureScale(method)
    with ctx.sub_ctx("train") as sub_ctx:
        train_data = sub_ctx.reader(train_data).read_dataframe().data.to_local()
        scaler.fit(sub_ctx, train_data)

        model = scaler.to_model()
        sub_ctx.writer(output_model).write_model(model)

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = scaler.transform(sub_ctx, train_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def predict(ctx, input_model, test_data, test_output_data):
    with ctx.sub_ctx("predict") as sub_ctx:
        model = sub_ctx.reader(input_model).read_model()
        scaler = FeatureScale.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        output_data = scaler.transform(sub_ctx, test_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)
