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

from examples.pipeline.hetero_logistic_regression import common_tools

from examples.util.config import Config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = Config.load(config)
    backend = config.backend
    work_mode = config.work_mode

    lr_param = {
        "name": "hetero_lr_0",
        "penalty": "L2",
        "optimizer": "rmsprop",
        "tol": 1e-05,
        "alpha": 0.01,
        "max_iter": 10,
        "early_stop": "diff",
        "batch_size": -1,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "random_uniform"
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False
        }
    }

    pipeline = common_tools.make_feature_engineering_dsl(config, namespace, lr_param)
    # fit model
    pipeline.fit(backend=backend, work_mode=work_mode)
    # query component summary
    common_tools.prettify(pipeline.get_component("hetero_lr_0").get_summary())
    common_tools.prettify(pipeline.get_component("evaluation_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
