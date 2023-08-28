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
from fate_client.pipeline.components.fate import CoordinatedPoisson, PSI
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
    psi_0.guest.component_setting(input_data=DataWarehouseChannel(name="dvisits_hetero_guest",
                                                                  namespace=f"experiment{namespace}"))
    psi_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="dvisits_hetero_host",
                                                                     namespace=f"experiment{namespace}"))
    poisson_0 = CoordinatedPoisson("poisson_0",
                                   epochs=4,
                                   batch_size=None,
                                   optimizer={"method": "SGD", "optimizer_params": {"lr": 0.01},
                                              "alpha": 0.001},
                                   init_param={"fit_intercept": True, "method": "zeros"},
                                   train_data=psi_0.outputs["output_data"],
                                   learning_rate_scheduler={"method": "constant", "scheduler_params": {"factor": 1.0,
                                                                                                       "total_iters": 100}})
    poisson_1 = CoordinatedPoisson("poisson_1", train_data=psi_0.outputs["output_data"],
                                   warm_start_model=poisson_0.outputs["output_model"],
                                   epochs=2,
                                   batch_size=None,
                                   optimizer={"method": "SGD", "optimizer_params": {"lr": 0.01},
                                              "alpha": 0.001},
                                   )

    poisson_2 = CoordinatedPoisson("poisson_2", epochs=6,
                                   batch_size=None,
                                   optimizer={"method": "SGD", "optimizer_params": {"lr": 0.01},
                                              "alpha": 0.001},
                                   init_param={"fit_intercept": True, "method": "zeros"},
                                   train_data=psi_0.outputs["output_data"],
                                   learning_rate_scheduler={"method": "constant", "scheduler_params": {"factor": 1.0,
                                                                                                       "total_iters": 100}})

    evaluation_0 = Evaluation("evaluation_0",
                              label_column_name="doctorco",
                              runtime_roles=["guest"],
                              default_eval_setting="regression",
                              input_data=[poisson_1.outputs["train_output_data"],
                                          poisson_2.outputs["train_output_data"]])

    pipeline.add_task(psi_0)
    pipeline.add_task(poisson_0)
    pipeline.add_task(poisson_1)
    pipeline.add_task(poisson_2)
    pipeline.add_task(evaluation_0)

    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()
    # print(f"poisson_1 model: {pipeline.get_task_info('poisson_1').get_output_model()}")
    # print(f"train poisson_1 data: {pipeline.get_task_info('poisson_1').get_output_data()}")

    # print(f"poisson_2 model: {pipeline.get_task_info('poisson_2').get_output_model()}")
    # print(f"train poisson_2 data: {pipeline.get_task_info('poisson_2').get_output_data()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
