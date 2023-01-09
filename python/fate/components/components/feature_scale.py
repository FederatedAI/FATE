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
    GUEST,
    HOST,
    DatasetArtifact,
    Input,
    ModelArtifact,
    Output,
    Role,
    cpn,
)


@cpn.component(roles=[GUEST, HOST])
def feature_scale(ctx, role):
    ...


@feature_scale.train()
@cpn.artifact("train_data", type=Input[DatasetArtifact], roles=[GUEST, HOST])
@cpn.parameter("method", type=str, default="standard", optional=False)
@cpn.artifact("train_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[GUEST, HOST])
def feature_scale_train(
    ctx,
    role: Role,
    train_data,
    method,
    train_output_data,
    output_model,
):
    train(ctx, train_data, train_output_data, output_model, method)


@feature_scale.predict()
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[GUEST, HOST])
@cpn.artifact("test_data", type=Input[DatasetArtifact], optional=False, roles=[GUEST, HOST])
@cpn.artifact("test_output_data", type=Output[DatasetArtifact], roles=[GUEST, HOST])
def feature_scale_predict(
    ctx,
    role: Role,
    test_data,
    input_model,
    test_output_data,
):
    predict(ctx, input_model, test_data, test_output_data)


def train(ctx, train_data, train_output_data, output_model, method):
    from fate.ml.feature_scale import FeatureScale

    scaler = FeatureScale(method)
    with ctx.sub_ctx("train") as sub_ctx:
        train_data = sub_ctx.reader(train_data).read_dataframe().data
        scaler.fit(sub_ctx, train_data)

        model = scaler.to_model()
        with output_model as model_writer:
            model_writer.write_model("feature_scale", model, metadata={})

    with ctx.sub_ctx("predict") as sub_ctx:
        output_data = scaler.transform(sub_ctx, train_data)
        sub_ctx.writer(train_output_data).write_dataframe(output_data)


def predict(ctx, input_model, test_data, test_output_data):
    from fate.ml.feature_scale import FeatureScale

    with ctx.sub_ctx("predict") as sub_ctx:
        with input_model as model_reader:
            model = model_reader.read_model()
        scaler = FeatureScale.from_model(model)
        test_data = sub_ctx.reader(test_data).read_dataframe().data
        output_data = scaler.transform(sub_ctx, test_data)
        sub_ctx.writer(test_output_data).write_dataframe(output_data)
