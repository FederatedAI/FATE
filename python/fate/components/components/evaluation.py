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
from fate.components import ARBITER, GUEST, HOST, Role
from fate.components.core import artifacts, component, parameter
from fate.ml.evaluation import BinaryEvaluator


@component(roles=[GUEST, HOST, ARBITER])
@artifacts.dataframe_input("input_data", roles=[GUEST, HOST, ARBITER])
@parameter("eval_type", type=str, default="binary", optional=True)
@artifacts.json_metric_output("output_metric", roles=[GUEST, HOST, ARBITER])
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
