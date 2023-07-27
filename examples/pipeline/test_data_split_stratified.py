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
from fate_client.pipeline.components.fate import DataSplit
from fate_client.pipeline.components.fate import Intersection
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

intersection_0 = Intersection("intersection_0",
                              method="raw")
intersection_0.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                       namespace="experiment_sid"))
intersection_0.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                          namespace="experiment_sid"))

intersection_1 = Intersection("intersection_1",
                              method="raw")
intersection_1.guest.component_setting(input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                                       namespace="experiment_sid"))
intersection_1.hosts[0].component_setting(input_data=DataWarehouseChannel(name="breast_hetero_host",
                                                                          namespace="experiment_sid"))

data_split_0 = DataSplit("data_split_0",
                         train_size=0.6,
                         validate_size=0.0,
                         test_size=0.4,
                         stratified=True,
                         input_data=intersection_0.outputs["output_data"])

data_split_1 = DataSplit("data_split_1",
                         train_size=200,
                         test_size=50,
                         input_data=intersection_0.outputs["output_data"]
                         )

pipeline.add_task(intersection_0)
pipeline.add_task(intersection_1)
pipeline.add_task(data_split_0)
pipeline.add_task(data_split_1)

# pipeline.add_task(hetero_feature_binning_0)
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()

# print(pipeline.get_task_info("data_split_0").get_output_data())
output_data = pipeline.get_task_info("data_split_0").get_output_data()
import pandas as pd

print(f"data split 0 train size: {pd.DataFrame(output_data['train_output_data']).shape};"
      f"validate size: {pd.DataFrame(output_data['validate_output_data']).shape}"
      f"test size: {pd.DataFrame(output_data['test_output_data']).shape}")
output_data = pipeline.get_task_info("data_split_1").get_output_data()
print(f"data split 1train size: {pd.DataFrame(output_data['train_output_data']).shape};"
      f"validate size: {pd.DataFrame(output_data['validate_output_data']).shape}"
      f"test size: {pd.DataFrame(output_data['test_output_data']).shape}")
