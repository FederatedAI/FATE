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
from fate_client.pipeline.components.fate import CoordinatedLR
from fate_client.pipeline.components.fate import Evaluation
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

lr_0 = CoordinatedLR("lr_0",
                     epochs=10,
                     batch_size=100,
                     optimizer={"method": "sgd", "optimizer_params": {"lr": 0.1}, "alpha": 0.5},
                     init_param={"fit_intercept": True})
lr_1 = CoordinatedLR("lr_1", input_model=lr_0.outputs["output_model"],
                     test_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                    namespace="experiment_64")
                     )

lr_0.guest.component_setting(train_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                             namespace="experiment_64"))
lr_0.hosts[0].component_setting(train_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                namespace="experiment_64"))

evaluation_0 = Evaluation("evaluation_0",
                          runtime_roles=["guest"],
                          input_data=lr_0.outputs["train_output_data"])

# pipeline.add_task(feature_scale_0)
# pipeline.add_task(feature_scale_1)
pipeline.add_task(lr_0)
pipeline.add_task(lr_1)
pipeline.add_task(evaluation_0)
# pipeline.add_task(hetero_feature_binning_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()

# print(pipeline.get_task_info("statistics_0").get_output_model())
print(pipeline.get_task_info("lr_0").get_output_model())
print(pipeline.get_task_info("lr_0").get_output_metric())
print(f"evaluation metric: ")
print(pipeline.get_task_info("evaluation_0").get_output_metric())

pipeline.deploy([lr_0])

predict_pipeline = FateFlowPipeline()

deployed_pipeline = pipeline.get_deployed_pipeline()
deployed_pipeline.lr_0.guest.component_setting(test_data=DataWarehouseChannel(name="breast_hetero_guest_data",
                                                                              namespace="experiment"))
deployed_pipeline.lr_0.hosts[0].component_setting(test_data=DataWarehouseChannel(name="breast_hetero_guest_data",
                                                                                 namespace="experiment"))

predict_pipeline.add_task(deployed_pipeline)
predict_pipeline.compile()
# print("\n\n\n")
# print(predict_pipeline.compile().get_dag())
predict_pipeline.predict()
