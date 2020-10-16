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
        config = Config.load(config)
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

    selection_param = {
        "name": "hetero_feature_selection_0",
        "select_col_indexes": -1,
        "select_names": [],
        "filter_methods": [
            "unique_value",
            "iv_value_thres",
            "coefficient_of_variation_value_thres",
            "iv_percentile",
            "outlier_cols"
        ],
        "unique_param": {
            "eps": 1e-06
        },
        "iv_value_param": {
            "value_threshold": 0.1
        },
        "iv_percentile_param": {
            "percentile_threshold": 0.9
        },
        "variance_coe_param": {
            "value_threshold": 0.3
        },
        "outlier_param": {
            "percentile": 0.95,
            "upper_threshold": 2.0
        }
    }
    pipeline = common_tools.make_normal_dsl(config, namespace, selection_param,
                                            binning_param=binning_param,
                                            statistic_param=statistic_param,
                                            is_multi_host=True)
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
