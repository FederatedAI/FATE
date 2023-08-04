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
from fate_client.pipeline.components.fate import CoordinatedLinR, Intersection
from fate_client.pipeline.components.fate import Evaluation
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

intersect_0 = Intersection("intersect_0", method="raw")
intersect_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                    namespace="experiment"))
intersect_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                       namespace="experiment"))
linr_0 = CoordinatedLinR("linr_0",
                         epochs=3,
                         batch_size=None,
                         optimizer={"method": "sgd", "optimizer_params": {"lr": 0.15}, "alpha": 0.1},
                         init_param={"fit_intercept": True, "method": "zeros"},
                         train_data=intersect_0.outputs["output_data"],
                         shuffle=False)
linr_1 = CoordinatedLinR("linr_1", train_data=intersect_0.outputs["output_data"],
                         warm_start_model=linr_0.outputs["output_model"],
                         epochs=2,
                         batch_size=None)
linr_2 = CoordinatedLinR("linr_2",
                         epochs=5,
                         batch_size=None,
                         optimizer={"method": "sgd", "optimizer_params": {"lr": 0.15}, "alpha": 0.1},
                         init_param={"fit_intercept": True, "method": "zeros"},
                         train_data=intersect_0.outputs["output_data"],
                         shuffle=False)

"""linr_0.guest.component_setting(train_data=DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                             namespace="experiment"))
linr_0.hosts[0].component_setting(train_data=DataWarehouseChannel(name="breast_hetero_host_sid",
                                                                namespace="experiment"))"""

evaluation_0 = Evaluation("evaluation_0",
                          runtime_roles=["guest"],
                          metrics=["r2_score", "mse"],
                          label_column_name="y",
                          input_data=[linr_1.outputs["train_output_data"], linr_2.outputs["train_output_data"]])

# pipeline.add_task(feature_scale_0)
# pipeline.add_task(feature_scale_1)
pipeline.add_task(intersect_0)
pipeline.add_task(linr_0)
pipeline.add_task(linr_1)
pipeline.add_task(linr_2)
pipeline.add_task(evaluation_0)
# pipeline.add_task(hetero_feature_binning_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
import numpy as np

linr_0_coef = np.array(
    pipeline.get_task_info('linr_0').get_output_model()["output_model"]["data"]['estimator']["param"]["coef_"])
linr_0_intercept = np.array(
    pipeline.get_task_info('linr_0').get_output_model()["output_model"]["data"]['estimator']["param"]["intercept_"])

linr_1_coef = np.array(
    pipeline.get_task_info('linr_1').get_output_model()["output_model"]["data"]['estimator']["param"]["coef_"])
linr_1_intercept = np.array(
    pipeline.get_task_info('linr_1').get_output_model()["output_model"]["data"]['estimator']["param"]["intercept_"])
# print(f"linr_1 data: {pipeline.get_task_info('linr_0').get_output_data()}")
linr_2_coef = np.array(
    pipeline.get_task_info('linr_2').get_output_model()["output_model"]["data"]['estimator']["param"]["coef_"])
linr_2_intercept = np.array(
    pipeline.get_task_info('linr_2').get_output_model()["output_model"]["data"]['estimator']["param"]["intercept_"])

print(f"linr_1 coef: {linr_1_coef}, intercept: {linr_1_intercept}")
print(f"linr_2 coef: {linr_2_coef}, intercept: {linr_2_intercept}")
print(f"linr_1 vs l2_1 coef diff: {linr_1_coef - linr_2_coef}, intercept diff: {linr_1_intercept - linr_2_intercept}")

print(f"\n evaluation result: {pipeline.get_task_info('evaluation_0').get_output_metric()[0]['data']}")
