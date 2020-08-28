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
#

import math
import json
import typing
from pathlib import Path

from prettytable import PrettyTable
from ruamel import yaml


def _get_common_metrics(**results):
    common_metrics = None
    for result in results.values():
        if common_metrics is None:
            common_metrics = result.keys()
        else:
            common_metrics = common_metrics & result.keys()
    return list(common_metrics)


def _filter_results(metrics, **results):
    filtered_results = {}
    for model_name, result in results.items():
        model_result = [result[metric] for metric in metrics]
        filtered_results[model_name] = model_result
    return filtered_results


def evaluate_almost_equal(metrics, results, abs_tol=None):
    """
    Evaluate for each given metric if values in results are almost equal
    Parameters
    ----------
    metrics: List[str], metrics names
    results: dict, results to be evaluated
    abs_tol: float, absolute error tolerance

    Returns
    -------
    bool, return True if all metrics in results are almost equal

    """
    # return False if empty
    if len(metrics) == 0:
        return False
    eval_summary = {}
    for i, metric in enumerate(metrics):
        v_eval = [res[i] for res in results]
        first_v = v_eval[0]
        if abs_tol is not None:
            eval_summary[metric] = all(math.isclose(v, first_v, abs_tol=abs_tol) for v in v_eval)
        else:
            eval_summary[metric] = all(math.isclose(v, first_v) for v in v_eval)
    return eval_summary


def match_metrics(evaluate, **results):
    """
    Get metrics
    Parameters
    ----------
    evaluate: bool, whether to evaluate metrics are almost equal, and include compare results in output report
    results: dict of model name: metrics

    Returns
    -------
    match result

    """
    common_metrics = _get_common_metrics(**results)
    filtered_results = _filter_results(common_metrics, **results)
    table = PrettyTable()
    model_names = list(filtered_results.keys())
    table.field_names = ["Model Name"] + common_metrics
    for model_name in model_names:
        row = [model_name] + filtered_results[model_name]
        table.add_row(row)
    print(table.get_string(title="Match Results"))

    if evaluate:
        eval_summary = evaluate_almost_equal(common_metrics, filtered_results)
        eval_table = PrettyTable()
        eval_table.field_names = ["Metric", "All Match"]
        for metric, v in eval_summary.items():
            row = [metric, v]
            eval_table.add_row(row)
        print(eval_table)


def load_conf(path: typing.Union[str, Path]):
    """
    Loads conf content from json or yaml file. Used by match to read in parameter configuration
    Parameters
    ----------
    path: str, path to conf file, should be absolute path

    Returns
    -------
    dict, parameter configuration in dictionary format

    """
    if isinstance(path, str):
        path = Path(path)
    config = {}
    if path is not None:
        file_type = path.suffix
        with path.open("r") as f:
            if file_type == "yaml":
                config.update(yaml.safe_load(f))
            elif file_type == "json":
                config.update(json.load(f))
            else:
                raise ValueError(f"Cannot load conf from file type {file_type}")
    return config
