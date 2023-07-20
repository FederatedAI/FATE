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
import json

from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import Intersection, HeteroFeatureBinning
from fate_client.pipeline.components.fate import Statistics, HeteroFeatureSelection
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

intersection_0 = Intersection("intersection_0",
                              method="raw")
intersection_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                                       namespace="experiment"))
intersection_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host_sid",
                                                                          namespace="experiment"))

"""
intersection_1 = Intersection("intersection_1",
                              method="raw")
intersection_1.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                       namespace="experiment_64"))
intersection_1.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                          namespace="experiment_64"))
"""

statistics_0 = Statistics("statistics_0", input_data=intersection_0.outputs["output_data"],
                          metrics=["mean", "max", "std", "var", "kurtosis", "skewness"])
binning_0 = HeteroFeatureBinning("binning_0", train_data=intersection_0.outputs["output_data"],
                                 method="quantile", n_bins=5, bin_col=None, category_col=None,
                                 skip_metrics=False, transform_method="woe")

selection_0 = HeteroFeatureSelection("selection_0",
                                     train_data=intersection_0.outputs["output_data"],
                                     method=["iv", "statistics"],
                                     input_models=[binning_0.outputs["output_model"],
                                                   statistics_0.outputs["output_model"]
                                                   ],
                                     iv_param={"select_federated": True},
                                     statistic_param={"metrics": ["mean", "max", "kurtosis", "skewness"]},
                                     manual_param={"filter_out_col": ["x0", "x3"]})

pipeline.add_task(intersection_0)
pipeline.add_task(binning_0)
pipeline.add_task(statistics_0)
# pipeline.add_task(selection_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
# print(json.dumps(pipeline.get_task_info("statistics_0").get_output_model(), indent=4))
print(json.dumps(pipeline.get_task_info("binning_0").get_output_model(), indent=4))
print(pipeline.get_task_info("binning_0").get_output_data())

predict_pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")
pipeline.deploy([intersection_0, binning_0, selection_0])

deployed_pipeline = pipeline.get_deployed_pipeline()

deployed_pipeline.interesction_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                                                         namespace="experiment"))
deployed_pipeline.intersection_0.hosts[0].component_setting(
    input_data=DataWarehouseChannel(name="breast_hetero_host_sid",
                                    namespace="experiment"))

predict_pipeline.add_task(deployed_pipeline)

print("\n\n\n")
print(predict_pipeline.compile().get_dag())
predict_pipeline.predict()

print(predict_pipeline.get_task_info("selection_0").get_output_data())
