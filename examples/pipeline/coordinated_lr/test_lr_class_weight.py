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

import argparse

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import CoordinatedLR, PSI
from fate_client.pipeline.components.fate import Evaluation
from fate_client.pipeline.interface import DataWarehouseChannel
from fate_client.pipeline.utils import test_utils


def main(config="../config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]

    pipeline = FateFlowPipeline().set_roles(guest=guest, host=host, arbiter=arbiter)
    if config.task_cores:
        pipeline.conf.set("task_cores", config.task_cores)
    if config.timeout:
        pipeline.conf.set("timeout", config.timeout)

    psi_0 = PSI("psi_0")
    psi_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                  namespace=f"experiment{namespace}"))
    psi_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                     namespace=f"experiment{namespace}"))
    lr_0 = CoordinatedLR("lr_0",
                         epochs=10,
                         batch_size=300,
                         optimizer={"method": "SGD", "optimizer_params": {"lr": 0.1}, "penalty": "l2", "alpha": 0.001},
                         init_param={"fit_intercept": True, "method": "zeros"},
                         train_data=psi_0.outputs["output_data"],
                         learning_rate_scheduler={"method": "linear", "scheduler_params": {"start_factor": 0.7,
                                                                                           "total_iters": 100}},
                         class_weight='balanced')

    evaluation_0 = Evaluation("evaluation_0",
                              runtime_roles=["guest"],
                              default_eval_setting="binary",
                              input_data=lr_0.outputs["train_output_data"])

    pipeline.add_task(psi_0)
    pipeline.add_task(lr_0)
    pipeline.add_task(evaluation_0)

    pipeline.compile()
    pipeline.fit()
    print(f"predict lr_0 model: {pipeline.get_task_info('lr_0').get_output_model()}")

    pipeline.deploy([psi_0, lr_0])

    predict_pipeline = FateFlowPipeline()

    deployed_pipeline = pipeline.get_deployed_pipeline()
    deployed_pipeline.psi_0.guest.component_setting(
        input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                        namespace=f"experiment{namespace}"))
    deployed_pipeline.psi_0.hosts[0].component_setting(
        input_data=DataWarehouseChannel(name="breast_hetero_host",
                                        namespace=f"experiment{namespace}"))

    predict_pipeline.add_task(deployed_pipeline)
    predict_pipeline.compile()
    # print("\n\n\n")
    # print(predict_pipeline.compile().get_dag())
    predict_pipeline.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
