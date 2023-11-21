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
from fate_client.pipeline.components.fate import PSI, FeatureScale, Statistics
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

    psi_1 = PSI("psi_1")
    psi_1.guest.task_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                             namespace=f"experiment{namespace}"))
    psi_1.hosts[0].task_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                namespace=f"experiment{namespace}"))

    feature_scale_0 = FeatureScale("feature_scale_0",
                                   method="min_max",
                                   feature_range={"x0": [-1, 1]},
                                   scale_col=["x0", "x1", "x3"],
                                   train_data=psi_0.outputs["output_data"])

    feature_scale_1 = FeatureScale("feature_scale_1",
                                   test_data=psi_1.outputs["output_data"],
                                   input_model=feature_scale_0.outputs["output_model"])

    statistics_0 = Statistics("statistics_0",
                              metrics=["max", "min", "mean", "std"],
                              input_data=feature_scale_1.outputs["test_output_data"])

    pipeline.add_task(psi_0)
    pipeline.add_task(psi_1)
    pipeline.add_task(feature_scale_0)
    pipeline.add_task(feature_scale_1)
    pipeline.add_task(statistics_0)

    # pipeline.add_task(hetero_feature_binning_0)
    pipeline.compile()
    print(pipeline.get_dag())
    pipeline.fit()

    print(pipeline.get_task_info("statistics_0").get_output_model())

    pipeline.deploy([psi_0, feature_scale_0])

    predict_pipeline = FateFlowPipeline()

    deployed_pipeline = pipeline.get_deployed_pipeline()
    deployed_pipeline.psi_0.guest.task_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                               namespace=f"experiment{namespace}"))
    deployed_pipeline.psi_0.hosts[0].task_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
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
