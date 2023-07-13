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
                                                                    namespace="experiment_sid_0713"))
intersect_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                       namespace="experiment_sid_0713"))
lr_0 = CoordinatedLR("lr_0",
                     epochs=2,
                     batch_size=100,
                     optimizer={"method": "sgd", "optimizer_params": {"lr": 0.01}},
                     init_param={"fit_intercept": True},
                     cv_data=intersect_0.outputs["output_data"],
                     cv_param={"n_splits": 3})

"""lr_0.guest.component_setting(train_data=DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                             namespace="experiment"))
lr_0.hosts[0].component_setting(train_data=DataWarehouseChannel(name="breast_hetero_host_sid",
                                                                namespace="experiment"))"""

evaluation_0 = Evaluation("evaluation_0",
                          runtime_roles=["guest"],
                          input_data=lr_0.outputs["train_output_data"])

# pipeline.add_task(feature_scale_0)
# pipeline.add_task(feature_scale_1)
pipeline.add_task(intersect_0)
pipeline.add_task(lr_0)
pipeline.add_task(evaluation_0)
# pipeline.add_task(hetero_feature_binning_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
