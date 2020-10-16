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

    fast_sbt_param = {
        "name": "fast_secureboost_0",
        "task_type": "classification",
        "learning_rate": 0.1,
        "num_trees": 4,
        "subsample_feature_rate": 1,
        "n_iter_no_change": False,
        "work_mode": "layered",
        "guest_depth": 2,
        "host_depth": 3,
        "tol": 0.0001,
        "bin_num": 50,
        "metrics": ["Recall", "ks", "auc", "roc"],
        "objective_param": {
            "objective": "cross_entropy"
        },
        "encrypt_param": {
            "method": "iterativeAffine"
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
            "hetero_fast_sbt_filter"
        ],
        "sbt_param": {
            "metrics": "feature_importance",
            "filter_type": "threshold",
            "take_high": True,
            "threshold": 0.03
        }}
    pipeline = common_tools.make_normal_dsl(config, namespace, selection_param,
                                            fast_sbt_param=fast_sbt_param)
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
