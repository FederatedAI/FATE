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
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import CoordinatedLR, Intersection
from fate_client.pipeline.components.fate import Evaluation
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

intersect_0 = Intersection("intersect_0", method="raw")
intersect_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                    namespace="experiment_sid"))
intersect_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                       namespace="experiment_sid"))
lr_0 = CoordinatedLR("lr_0",
                     epochs=4,
                     batch_size=None,
                     optimizer={"method": "rprop", "optimizer_params": {"lr": 0.01}},
                     init_param={"fit_intercept": True, "method": "zeros"},
                     train_data=intersect_0.outputs["output_data"])
lr_1 = CoordinatedLR("lr_1", test_data=intersect_0.outputs["output_data"],
                     input_model=lr_0.outputs["output_model"])

"""lr_0.guest.component_setting(train_data=DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                             namespace="experiment"))
lr_0.hosts[0].component_setting(train_data=DataWarehouseChannel(name="breast_hetero_host_sid",
                                                                namespace="experiment"))"""

evaluation_0 = Evaluation("evaluation_0",
                          label_column_name="y",
                          runtime_roles=["guest"],
                          default_eval_setting="binary",
                          input_data=lr_0.outputs["train_output_data"])

# pipeline.add_task(feature_scale_0)
# pipeline.add_task(feature_scale_1)
pipeline.add_task(intersect_0)
pipeline.add_task(lr_0)
# pipeline.add_task(evaluation_0)
# pipeline.add_task(hetero_feature_binning_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
print(f"lr_0 model: {pipeline.get_task_info('lr_0').get_output_model()}")
print(f"train lr_0 data: {pipeline.get_task_info('lr_0').get_output_data()}")

# print(pipeline.get_task_info("statistics_0").get_output_model())
# print(f"evaluation metrics: ")
# print(pipeline.get_task_info("evaluation_0").get_output_metric())

pipeline.deploy([intersect_0, lr_0])

predict_pipeline = FateFlowPipeline()

deployed_pipeline = pipeline.get_deployed_pipeline()
deployed_pipeline.intersect_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                                      namespace="experiment_sid"))
deployed_pipeline.intersect_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                                         namespace="experiment_sid"))

predict_pipeline.add_task(deployed_pipeline)
predict_pipeline.compile()
# print("\n\n\n")
# print(predict_pipeline.compile().get_dag())
predict_pipeline.predict()
print(f"predict lr_0 data: {pipeline.get_task_info('lr_0').get_output_data()}")
