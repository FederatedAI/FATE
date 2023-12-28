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

import inspect
from fate.ml.evaluation import classification as classi
from fate.ml.evaluation import regression as reg
from fate.ml.evaluation.metric_base import Metric, MetricEnsemble


def get_metric_names(modules):
    result = {}

    for module in modules:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if hasattr(obj, "metric_name") and issubclass(obj, Metric):
                    metric_name = getattr(obj, "metric_name")
                    if metric_name is not None:
                        result[metric_name] = obj

    return result


def all_available_metrics():
    return get_metric_names([classi, reg])


def get_single_val_binary_metrics(threshold=0.5):
    binary_ensembles = MetricEnsemble()
    binary_ensembles.add_metric(classi.AUC()).add_metric(classi.BinaryAccuracy(threshold=threshold)).add_metric(
        classi.BinaryF1Score(threshold=threshold)
    )
    binary_ensembles.add_metric(classi.BinaryPrecision(threshold=threshold)).add_metric(
        classi.BinaryRecall(threshold=threshold)
    )
    return binary_ensembles


def get_binary_metrics():
    binary_ensembles = MetricEnsemble()
    binary_ensembles.add_metric(classi.AUC()).add_metric(classi.KS()).add_metric(classi.ConfusionMatrix())
    binary_ensembles.add_metric(classi.Gain()).add_metric(classi.Lift())
    binary_ensembles.add_metric(classi.BiClassPrecisionTable()).add_metric(classi.BiClassRecallTable())
    binary_ensembles.add_metric(classi.BiClassAccuracyTable()).add_metric(classi.FScoreTable())
    return binary_ensembles


def get_multi_metrics():
    multi_ensembles = MetricEnsemble()
    multi_ensembles.add_metric(classi.MultiAccuracy()).add_metric(classi.MultiPrecision()).add_metric(
        classi.MultiRecall()
    )

    return multi_ensembles


def get_regression_metrics():
    regression_ensembles = MetricEnsemble()
    regression_ensembles.add_metric(reg.RMSE()).add_metric(reg.MAE()).add_metric(reg.MSE()).add_metric(reg.R2Score())
    return regression_ensembles


def get_special_metrics():
    # metrics that need special input format like PSI
    ensembles = MetricEnsemble()
    ensembles.add_metric(classi.PSI())
    return ensembles


def get_specified_metrics(metric_names: list):
    ensembles = MetricEnsemble()
    available_metrics = get_metric_names([classi, reg])
    for metric_name in metric_names:
        if metric_name in available_metrics:
            ensembles.add_metric(get_metric_names([classi, reg])[metric_name]())
        else:
            raise ValueError(
                f"metric {metric_name} is not supported yet, supported metrics are \n {list(available_metrics.keys())}"
            )
    return ensembles
