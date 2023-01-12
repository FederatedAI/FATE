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
from fate_client.pipeline.pipeline import FateFlowPipeline

pipeline = FateFlowPipeline()
pipeline.upload(file="/Users/maguoqiang/mgq/FATE-2.0-alpha-with-flow/FATE/examples/data/breast_hetero_guest.csv",
                head=1,
                partitions=4,
                namespace="experiment",
                name="breast_hetero_guest",
                storage_engine="standalone",
                meta={
                    "label_name": "y",
                    "label_type": "float32",
                    "dtype": "float32"
                })
