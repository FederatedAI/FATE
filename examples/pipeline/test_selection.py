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
from fate_client.pipeline.components.fate import FeatureScale
from fate_client.pipeline.components.fate import Intersection
from fate_client.pipeline.components.fate import Statistics, HeteroFeatureSelection
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

intersection_0 = Intersection("intersection_0",
                              method="raw")
intersection_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                       namespace="experiment"))
intersection_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                          namespace="experiment"))

intersection_1 = Intersection("intersection_1",
                              method="raw")
intersection_1.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                       namespace="experiment"))
intersection_1.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                          namespace="experiment"))

feature_scale_0 = FeatureScale("feature_scale_0",
                               method="standard",
                               train_data=intersection_0.outputs["output_data"])

feature_scale_1 = FeatureScale("feature_scale_1",
                               test_data=intersection_1.outputs["output_data"],
                               input_model=feature_scale_0.outputs["output_model"])

statistics_0 = Statistics("statistics_0", input_data=feature_scale_1.outputs["test_output_data"],
                          metrics=["mean", "max", "std", "var", "kurtosis", "skewness"])

selection_0 = HeteroFeatureSelection(name="selection_0", train_data=intersection_0.outputs["output_data"],
                                     method=["statistic"],
                                     input_statistic_model=statistics_0.outputs["output_model"],
                                     statistic_param={"metrics": ["mean", "max", "kurtosis", "skewness"]})

pipeline.add_task(intersection_0)
pipeline.add_task(intersection_1)
pipeline.add_task(feature_scale_0)
pipeline.add_task(feature_scale_1)
pipeline.add_task(statistics_0)
pipeline.add_task(selection_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
print(json.dumps(pipeline.get_task_info("statistics_0").get_output_model(), indent=4))
print(json.dumps(pipeline.get_task_info("selection_0").get_output_model(), indent=4))

predict_pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")
pipeline.deploy([intersection_0, selection_0])
deployed_pipeline = pipeline.get_deployed_pipeline()

predict_pipeline.add_task(deployed_pipeline)

print("\n\n\n")
print(predict_pipeline.compile().get_dag())
predict_pipeline.predict()

print(predict_pipeline.get_task_info("selection_0").get_output_data())
