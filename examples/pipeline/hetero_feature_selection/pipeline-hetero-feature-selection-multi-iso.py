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

import argparse
import os
import sys

cur_path = os.path.realpath(__file__)
for i in range(4):
    cur_path = os.path.dirname(cur_path)
print(f'fate_path: {cur_path}')
sys.path.append(cur_path)

from examples.pipeline.hetero_feature_selection import common_tools
from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    backend = config.backend
    work_mode = config.work_mode

    binning_param = {
        "name": 'hetero_feature_binning_0',
        "method": "quantile",
        "compress_thres": 10000,
        "head_size": 10000,
        "error": 0.001,
        "bin_num": 10,
        "bin_indexes": -1,
        "bin_names": None,
        "category_indexes": None,
        "category_names": None,
        "adjustment_factor": 0.5,
        "local_only": False,
        "transform_param": {
            "transform_cols": -1,
            "transform_names": None,
            "transform_type": "bin_num"
        }
    }

    statistic_param = {
        "name": "statistic_0",
        "statistics": ["95%", "coefficient_of_variance", "stddev"],
        "column_indexes": -1,
        "column_names": []
    }
    psi_param = {
        "name": "psi_0",
        "max_bin_num": 20
    }

    secureboost_param = {
        "name": "secureboost_0",
        "task_type": "classification",
        "learning_rate": 0.1,
        "num_trees": 5,
        "subsample_feature_rate": 1,
        "n_iter_no_change": False,
        "tol": 0.0001,
        "bin_num": 50,
        "objective_param": {
            "objective": "cross_entropy"
        },
        "encrypt_param": {
            "method": "paillier"
        },
        "predict_param": {
            "threshold": 0.5
        },
        "validation_freqs": 1
    }

    selection_param = {
        "name": "hetero_feature_selection_0",
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": [
            "iv_filter",
            "statistic_filter",
            "psi_filter",
            "hetero_sbt_filter"
        ],
        "iv_param": {
            "metrics": ["iv", "iv", "iv"],
            "filter_type": ["threshold", "top_k", "top_percentile"],
            "take_high": True,
            "threshold": [0.03, 15, 0.7],
            "host_thresholds": [[0.15], None, None],
            "select_federated": True
        },
        "statistic_param": {
            "metrics": ["skewness", "skewness", "kurtosis", "median"],
            "filter_type": "threshold",
            "take_high": [True, False, True, True],
            "threshold": [-10, 10, -1.5, -1.5]
        },
        "psi_param": {
            "metrics": "psi",
            "filter_type": "threshold",
            "take_high": False,
            "threshold": -0.1
        },
        "sbt_param": {
            "metrics": "feature_importance",
            "filter_type": "threshold",
            "take_high": True,
            "threshold": 0.03
        }}
    pipeline = common_tools.make_normal_dsl(config, namespace, selection_param,
                                            binning_param=binning_param,
                                            statistic_param=statistic_param,
                                            psi_param=psi_param,
                                            sbt_param=secureboost_param)
    pipeline.fit(backend=backend, work_mode=work_mode)
    common_tools.prettify(pipeline.get_component("hetero_feature_selection_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
