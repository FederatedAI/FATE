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
    ClassificationMetrics,
    DatasetArtifact,
    Input,
    Output,
    Role,
    cpn,
)
from fate.ml.evaluation import BinaryEvaluator


@cpn.component(roles=[GUEST, HOST, ARBITER])
@cpn.artifact("input_data", type=Input[DatasetArtifact], roles=[GUEST, HOST, ARBITER])
@cpn.parameter("eval_type", type=str, default="binary", optional=True)
@cpn.artifact("output_metric", type=Output[ClassificationMetrics], roles=[GUEST, HOST, ARBITER])
def evaluation(ctx, role: Role, input_data, eval_type, output_metric):
    evaluate(ctx, input_data, eval_type, output_metric)


def evaluate(ctx, input_data, eval_type, output_metric):
    data = ctx.reader(input_data).read_dataframe().data
    y_true = data.label.tolist()
    y_pred = data.predict_score.values.tolist()

    if eval_type == "binary":
        ctx.metrics.handler.register_metrics(auc=ctx.writer(output_metric))
        evaluator = BinaryEvaluator()
        evaluator.fit(ctx, y_true, y_pred)
