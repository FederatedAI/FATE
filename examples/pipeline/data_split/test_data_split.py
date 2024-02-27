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
from fate_client.pipeline.components.fate import DataSplit, PSI, Reader
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
    if config.engine_run:
        # print(f"config engine run: {config.engine_run}")
        pipeline.conf.set("task", dict(engine_run=config.engine_run))

    reader_0 = Reader("reader_0")
    reader_0.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_guest"
    )
    reader_0.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_host"
    )
    reader_1 = Reader("reader_1")
    reader_1.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_guest"
    )
    reader_1.hosts[0].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_hetero_host"
    )

    psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])
    psi_1 = PSI("psi_1", input_data=reader_1.outputs["output_data"])

    data_split_0 = DataSplit("data_split_0",
                             train_size=0.6,
                             validate_size=0.1,
                             test_size=None,
                             stratified=True,
                             input_data=psi_0.outputs["output_data"])

    data_split_1 = DataSplit("data_split_1",
                             train_size=200,
                             test_size=50,
                             input_data=psi_1.outputs["output_data"]
                             )

    pipeline.add_tasks([reader_0, reader_1, psi_0, psi_1, data_split_0, data_split_1])

    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()

    # print(pipeline.get_task_info("data_split_0").get_output_data())
    """output_data = pipeline.get_task_info("data_split_0").get_output_data()
    import pandas as pd
    
    print(f"data split 0 train size: {pd.DataFrame(output_data['train_output_data']).shape};"
          f"validate size: {pd.DataFrame(output_data['validate_output_data']).shape}"
          f"test size: {pd.DataFrame(output_data['test_output_data']).shape}")
    output_data = pipeline.get_task_info("data_split_1").get_output_data()
    print(f"data split 1train size: {pd.DataFrame(output_data['train_output_data']).shape};"
          f"validate size: {pd.DataFrame(output_data['validate_output_data']).shape}"
          f"test size: {pd.DataFrame(output_data['test_output_data']).shape}")"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
