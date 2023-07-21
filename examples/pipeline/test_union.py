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
from fate_client.pipeline.components.fate import FeatureUnion
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999")

feature_union_0 = FeatureUnion("feature_union_0",
                               runtime_roles=["guest"],
                               input_data_list=[DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                                     namespace="experiment"),
                                                DataWarehouseChannel(name="breast_hetero_guest_sid",
                                                                     namespace="experiment")],
                               axis=0)

pipeline.add_task(feature_union_0)

pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()

print(pipeline.get_task_info("feature_union_0").get_output_data())
