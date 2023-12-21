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
from fate_client.pipeline.components.fate import Evaluation, Reader
from fate_client.pipeline.components.fate import SSHELR, PSI
from fate_client.pipeline.utils import test_utils


def main(config="../config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host)
    if config.task_cores:
        pipeline.conf.set("task_cores", config.task_cores)
    if config.timeout:
        pipeline.conf.set("timeout", config.timeout)

    reader_0 = Reader("reader_0")
    reader_0.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_guest"
    )
    reader_0.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_host"
    )
    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])
    lr_0 = SSHELR("lr_0",
                  epochs=4,
                  batch_size=300,
                  learning_rate=0.05,
                  init_param={"fit_intercept": True, "method": "zeros"},
                  train_data=psi_0.outputs["output_data"],
                  reveal_every_epoch=False,
                  early_stop="diff",
                  reveal_loss_freq=1,
                  )
    lr_1 = SSHELR("lr_1", train_data=psi_0.outputs["output_data"],
                  warm_start_model=lr_0.outputs["output_model"],
                  epochs=2,
                  batch_size=300,
                  learning_rate=0.05,
                  reveal_every_epoch=False,
                  early_stop="diff",
                  reveal_loss_freq=1,
                  )

    lr_2 = SSHELR("lr_2", epochs=6,
                  batch_size=300,
                  learning_rate=0.05,
                  init_param={"fit_intercept": True, "method": "zeros"},
                  train_data=psi_0.outputs["output_data"],
                  reveal_every_epoch=False,
                  early_stop="diff",
                  reveal_loss_freq=1,
                  )

    evaluation_0 = Evaluation("evaluation_0",
                              runtime_parties=dict(guest=guest),
                              default_eval_setting="binary",
                              input_data=[lr_1.outputs["train_output_data"], lr_2.outputs["train_output_data"]])

    pipeline.add_tasks([reader_0, psi_0, lr_0, lr_1, lr_2, evaluation_0])

    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()
    # print(f"lr_1 model: {pipeline.get_task_info('lr_1').get_output_model()}")
    # print(f"train lr_1 data: {pipeline.get_task_info('lr_1').get_output_data()}")

    # print(f"lr_2 model: {pipeline.get_task_info('lr_2').get_output_model()}")
    # print(f"train lr_2 data: {pipeline.get_task_info('lr_2').get_output_data()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
