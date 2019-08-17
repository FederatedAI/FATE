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

import json

from arch.api import eggroll
from arch.api import federation

from workflow import ArbiterWorkFlow

job_id = "112233"
LOGGER_path = '/data/projects/fate/python/src/test_homo_arbiter.log'
config_path = "/data/projects/fate/python/src/federatedml/workflow/homo_lr_workflow/test/arbiter_runtime_conf.json"
runtime_json = """
{

    "local": {
        "role": "arbiter",
        "party_id": 99999
    },

    "role": {
        "host": [
            10001
        ],
        "arbiter": [
            99999
        ],
        "guest": [
            100000
        ]
    },

    "DataIOParam": {
        "with_label": "True",
        "label_idx": "0",
        "label_type": "int",
        "output_format": "sparse"
    },
    "WorkFlowParam": {
        "method": "train",
        "train_input_table": "some_lr_input_table_name",
        "train_input_namespace": "some_lr_input_path",
        "predict_input_table": "some_predict_data_table_name",
        "predict_input_namespace": "some_predict_data_path",
        "predict_result_partition": 2,
        "predict_output_table": "some_predict_output_table_name",
        "predict_output_namespace": "some_predict_output_path",
        "evaluation_output_table": "some_evaluate_output_table_name",
        "evaluation_output_namespace": "some_evaluate_output_path",
        "data_input_table": "some_train_data_input_table_name",
        "data_input_namespace": "some_train_data_input_path",
        "intersect_data_output_table": null,
        "intersect_data_output_namespace": null,
        "do_cross_validation": false,
        "n_split": 5
    },
    "EncryptParam": {
        "method": "paillier"
    },
    "InitParam": {
        "init_method": "random_normal"
    },
    "EvaluateParam":{
        "metrics": ["auc", "precision"],
        "classi_type": "binary",
        "pos_label": 1,
        "predict_threshold": [0.5]
    },
    "LogisticParam": {
        "penalty": "L2",
        "optimizer": "sgd",
        "eps": 1e-5,
        "alpha": 0.01,
        "max_iter": 100,
        "converge_func": "diff",
        "re_encrypt_batches": 2,
        "party_weight": 1,
        "batch_size": 320,
        "learning_rate": 0.01
    },
    "DecisionTreeParam": {
        "criterion_method": "xgboost",
        "criterion_params": [0.3],
        "max_depth": 5,
        "min_sample_split": 2,
        "min_impurity_split": 1e-3,
        "n_iter_no_change": true,
        "tol": 0.0001
    },
    "BoostingTreeParam": {
        "loss_type": "cross_entropy",
        "learning_rate": 0.15,
        "num_trees": 10,
        "subsample_feature_rate": 0.8,
        "n_iter_no_change": true,
        "tol": 0.0001,
        "metrics": ["auc", "precision", "recall"]
    },
    "IntersectParam": {
        "intersect_method": "rsa",
        "random_bit": 128,
        "is_send_intersect_ids": true,
        "is_get_intersect_ids": true
    }
}
"""


class TestHomoLRArbiter(ArbiterWorkFlow):
    def _init_argument(self):
        self._init_LOGGER(LOGGER_path)
        self._initialize(config_path)
        with open(config_path) as conf_f:
            runtime_json = json.load(conf_f)
        eggroll.init(job_id)
        federation.init(job_id, runtime_json)


if __name__ == '__main__':
    workflow = TestHomoLRArbiter()
    workflow.run()
