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

import typing
import json
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
        model_result = {metric: result[metric] for metric in metrics}
        filtered_results[model_name] = model_result
    return filtered_results


def match_metrics(eval, **results):
    """
        Get metrics
        Parameters
        ----------
        eval: bool, whether to evaluate metrics of models are almost equal, and include compare results in output report
        results: dict of model name: metrics

        Returns
        -------

        """
    common_metrics = _get_common_metrics(**results)
    filtered_results = _filter_results(common_metrics, **results)
    table = PrettyTable()
    table.field_names = ["Model Name"] + common_metrics
    for model_name, result in filtered_results.items():
        table.add_row([model_name] + results)
    print(table)


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
