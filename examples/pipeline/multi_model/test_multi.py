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
from fate_client.pipeline.components.fate import PSI, HeteroFeatureSelection, HeteroFeatureBinning, \
    FeatureScale, Union, DataSplit, CoordinatedLR, Statistics, Sample, Evaluation
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

    data_split_0 = DataSplit("data_split_0", input_data=psi_0.outputs["output_data"],
                             train_size=0.8, test_size=0.2, random_state=42)
    union_0 = Union("union_0", input_data_list=[data_split_0.outputs["train_output_data"],
                                                data_split_0.outputs["test_output_data"]])
    sample_0 = Sample("sample_0", input_data=data_split_0.outputs["train_output_data"],
                      n=800, replace=True, hetero_sync=True)

    binning_0 = HeteroFeatureBinning("binning_0",
                                     method="quantile",
                                     n_bins=10,
                                     train_data=sample_0.outputs["output_data"]
                                     )
    statistics_0 = Statistics("statistics_0",
                              input_data=psi_0.outputs["output_data"])
    selection_0 = HeteroFeatureSelection("selection_0",
                                         method=["iv", "statistics"],
                                         train_data=sample_0.outputs["output_data"],
                                         input_models=[binning_0.outputs["output_model"],
                                                       statistics_0.outputs["output_model"]],
                                         iv_param={"metrics": "iv", "filter_type": "threshold", "value": 0.1},
                                         statistic_param={"metrics": ["max", "min"], "filter_type": "top_k",
                                                          "threshold": 5})

    selection_1 = HeteroFeatureSelection("selection_1",
                                         input_model=selection_0.outputs["train_output_model"],
                                         test_data=data_split_0.outputs["test_output_data"])

    scale_0 = FeatureScale("scale_0", method="min_max",
                           train_data=selection_0.outputs["train_output_data"], )

    lr_0 = CoordinatedLR("lr_0", train_data=selection_0.outputs["train_output_data"],
                         validate_data=selection_1.outputs["test_output_data"], epochs=3)
    linr_0 = CoordinatedLR("linr_0", train_data=selection_0.outputs["train_output_data"],
                           validate_data=selection_1.outputs["test_output_data"], epochs=3)

    evaluation_0 = Evaluation("evaluation_0", input_data=lr_0.outputs["train_output_data"],
                              label_column_name="y",
                              runtime_roles=["guest"])
    evaluation_1 = Evaluation("evaluation_1", input_data=linr_0.outputs["train_output_data"],
                              default_eval_setting="regression",
                              label_column_name="y",
                              runtime_roles=["guest"])
    pipeline.add_task(psi_0)
    pipeline.add_task(data_split_0)
    pipeline.add_task(union_0)
    pipeline.add_task(sample_0)
    pipeline.add_task(binning_0)
    pipeline.add_task(statistics_0)
    pipeline.add_task(selection_0)
    pipeline.add_task(scale_0)
    pipeline.add_task(selection_1)
    pipeline.add_task(lr_0)
    pipeline.add_task(linr_0)
    pipeline.add_task(evaluation_0)
    pipeline.add_task(evaluation_1)

    # pipeline.add_task(hetero_feature_binning_0)
    pipeline.compile()
    # print(pipeline.get_dag())
    pipeline.fit()

    # print(pipeline.get_task_info("feature_scale_1").get_output_model())

    pipeline.deploy([psi_0, selection_0])

    predict_pipeline = FateFlowPipeline()

    deployed_pipeline = pipeline.get_deployed_pipeline()
    psi_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                  namespace=f"experiment{namespace}"))
    psi_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                     namespace=f"experiment{namespace}"))

    predict_pipeline.add_task(deployed_pipeline)
    predict_pipeline.compile()
    predict_pipeline.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)
