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
from fate_client.pipeline.components.fate import SSHELR, PSI
from fate_client.pipeline.interface import DataWarehouseChannel
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

    psi_0 = PSI("psi_0")
    psi_0.guest.task_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                             namespace=f"experiment{namespace}"))
    psi_0.hosts[0].task_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                namespace=f"experiment{namespace}"))
    lr_0 = SSHELR("lr_0",
                  learning_rate=0.15,
                  epochs=2,
                  batch_size=None,
                  init_param={"fit_intercept": True},
                  cv_data=psi_0.outputs["output_data"],
                  cv_param={"n_splits": 3},
                  reveal_every_epoch=False,
                  early_stop="diff",
                  reveal_loss_freq=1,
                  )

    pipeline.add_task(psi_0)
    pipeline.add_task(lr_0)
    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)