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
from fate_client.pipeline.components.fate import Statistics
from fate_client.pipeline.interface import DataWarehouseChannel

pipeline = FateFlowPipeline().set_roles(guest="9999", host="9998", arbiter="9998")

"""intersection_0 = Intersection("intersection_0",input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                            namespace="experiment"))"""
statistics_0 = Statistics("statistics_0",
                          input_data=DataWarehouseChannel(name="breast_hetero_guest",
                                                          namespace="experiment"),
                          metrics=["mean", "max", "std", "var", "kurtosis", "skewness", "median", "count", "min"])

pipeline.add_task(statistics_0)
pipeline.compile()
pipeline.fit()
print(json.dumps(pipeline.get_task_info("statistics_0").get_output_model(), indent=4))
