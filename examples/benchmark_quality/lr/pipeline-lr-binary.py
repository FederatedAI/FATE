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

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import CoordinatedLR, Intersection
from fate_client.pipeline.components.fate import Evaluation
from fate_client.pipeline.interface import DataWarehouseChannel
from fate_client.pipeline.utils import test_utils
from fate_test.utils import extract_data, parse_summary_result


def main(config="../../config.yaml", param="./lr_config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    if isinstance(param, str):
        param = test_utils.JobConfig.load_from_file(param)

    assert isinstance(param, dict)

    data_set = param.get("data_guest").split('/')[-1]
    if data_set == "default_credit_hetero_guest.csv":
        guest_data_table = 'default_credit_hetero_guest'
        host_data_table = 'default_credit_hetero_host'
    elif data_set == 'breast_hetero_guest.csv':
        guest_data_table = 'breast_hetero_guest'
        host_data_table = 'breast_hetero_host'
    elif data_set == 'give_credit_hetero_guest.csv':
        guest_data_table = 'give_credit_hetero_guest'
        host_data_table = 'give_credit_hetero_host'
    elif data_set == 'epsilon_5k_hetero_guest.csv':
        guest_data_table = 'epsilon_5k_hetero_guest'
        host_data_table = 'epsilon_5k_hetero_host'
    else:
        raise ValueError(f"Cannot recognized data_set: {data_set}")

    guest_train_data = {"name": guest_data_table, "namespace": f"experiment{namespace}"}
    host_train_data = {"name": host_data_table, "namespace": f"experiment{namespace}"}
    pipeline = FateFlowPipeline().set_roles(guest=guest, host=host, arbiter=arbiter)

    intersect_0 = Intersection("intersect_0", method="raw")
    intersect_0.guest.component_setting(input_data=DataWarehouseChannel(name=guest_train_data["name"],
                                                                        namespace=guest_train_data["namespace"]))
    intersect_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name=host_train_data["name"],
                                                                           namespace=host_train_data["namespace"]))

    lr_param = {
    }

    config_param = {
        "epochs": param["epochs"],
        "learning_rate_scheduler": param["learning_rate_scheduler"],
        "optimizer": param["optimizer"],
        "batch_size": param["batch_size"],
        "early_stop": param["early_stop"],
        "init_param": param["init_param"],
        "tol": 1e-5
    }
    lr_param.update(config_param)
    lr_0 = CoordinatedLR("lr_0",
                         train_data=intersect_0.outputs["output_data"],
                         **config_param)
    lr_1 = CoordinatedLR("lr_1",
                         test_data=intersect_0.outputs["output_data"],
                         input_model=lr_0.outputs["output_model"])

    evaluation_0 = Evaluation("evaluation_0",
                              label_column_name="y",
                              runtime_roles=["guest"],
                              default_eval_setting="binary",
                              input_data=lr_0.outputs["train_output_data"])

    pipeline.add_task(intersect_0)
    pipeline.add_task(lr_0)
    pipeline.add_task(lr_1)
    pipeline.add_task(evaluation_0)

    pipeline.compile()
    print(pipeline.get_dag())
    pipeline.fit()

    lr_0_data = pipeline.get_task_info("lr_0").get_output_data()["train_output_data"]
    lr_1_data = pipeline.get_task_info("lr_1").get_output_data()["test_output_data"]
    lr_0_score = extract_data(lr_0_data, "predict_result")
    lr_0_label = extract_data(lr_0_data, "y")
    lr_1_score = extract_data(lr_1_data, "predict_result")
    lr_1_label = extract_data(lr_1_data, "y")
    lr_0_score_label = extract_data(lr_0_data, "predict_result", keep_id=True)
    lr_1_score_label = extract_data(lr_1_data, "predict_result", keep_id=True)
    """print(f"evaluation result: {pipeline.get_task_info('evaluation_0').get_output_metric()};"
          f"result type: {type(pipeline.get_task_info('evaluation_0').get_output_metric())}")
    """
    result_summary = parse_summary_result(pipeline.get_task_info("evaluation_0").get_output_metric()[0]["data"])
    print(f"result_summary")

    data_summary = {"train": {"guest": guest_train_data["name"], "host": host_train_data["name"]},
                    "test": {"guest": guest_train_data["name"], "host": host_train_data["name"]}
                    }

    return data_summary, result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./breast_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
