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

from fate.ml.evaluation import classification as classi
from fate.ml.evaluation import regression as reg
from fate.ml.evaluation.metric_base import Metric, MetricEnsemble
from fate.components.params import string_choice


def get_binary_metrics():

    binary_ensembles = MetricEnsemble()
    binary_ensembles.add_metric(classi.AUC()).add_metric(classi.KS()).add_metric(classi.ConfusionMatrix())
    binary_ensembles.add_metric(classi.Gain()).add_metric(classi.Lift())
    binary_ensembles.add_metric(classi.BiClassPrecisionTable()).add_metric(classi.BiClassRecallTable())
    binary_ensembles.add_metric(classi.BiClassAccuracyTable()).add_metric(classi.FScoreTable())
    return binary_ensembles


def get_multi_metrics():
    
    multi_ensembles = MetricEnsemble()
    multi_ensembles.add_metric(classi.MultiAccuracy()).add_metric(classi.MultiPrecision).add_metric(classi.MultiRecall())
    
    return multi_ensembles


def get_regression_metrics():
    
    regression_ensembles = MetricEnsemble()
    regression_ensembles.add(reg.RMSE()).add(reg.MAE()).add(reg.MSE()).add(reg.R2Score())
    return regression_ensembles


def get_special_metrics():
    # metrics that need special input format like PSI
    ensembles = MetricEnsemble()
    ensembles.add_metric(classi.PSI())
    return ensembles


@cpn.component(roles=[GUEST, HOST, ARBITER])
@cpn.artifact("input_data", type=Input[DatasetArtifact], roles=[GUEST, HOST, ARBITER])
@cpn.parameter("eval_type", type=string_choice(choice=['binary', 'multi', 'regression']), default="binary", optional=True)
@cpn.artifact("output_metric", type=Output[ClassificationMetrics], roles=[GUEST, HOST, ARBITER])
def evaluation(ctx, role: Role, input_data, eval_type, output_metric):
    evaluate(ctx, input_data, eval_type, output_metric)


def evaluate(ctx, input_data, eval_type, output_metric):

    data = ctx.reader(input_data).read_dataframe().data
    y_true = data.label.tolist()
    y_pred = data.predict_score.values.tolist()

    if eval_type == "binary":
        ctx.metrics.handler.register_metrics(auc=ctx.writer(output_metric))
        metrics = get_binary_metrics()

    elif eval_type == 'regression':
        metrics = get_regression_metrics()

    elif eval_type == "multi":
        metrics = get_multi_metrics()

    rs = metrics.compute(y_true, y_pred)