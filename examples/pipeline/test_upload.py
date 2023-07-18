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

pipeline = FateFlowPipeline().set_roles(
    local="0")
pipeline.set_site_role("local")
pipeline.set_site_party_id("0")
meta = {'delimiter': ',',
        'dtype': 'float32',
        'input_format': 'dense',
        'label_type': 'int32',
        'label_name': 'y',
        'match_id_name': 'id',
        'match_id_range': 0,
        'sample_id_name': 'id',
        'tag_value_delimiter': ':',
        'tag_with_value': False,
        'weight_type': 'float32'}

pipeline.transform_local_file_to_dataframe(  # file="${abs_path_of_data_guest}",
    file="/Users/yuwu/PycharmProjects/FATE/examples/data/breast_hetero_guest.csv",
    meta=meta, head=True,
    namespace="experiment",
    name="breast_hetero_guest")

meta = {'delimiter': ',',
        'dtype': 'float32',
        'input_format': 'dense',
        'label_type': 'int',
        'match_id_name': 'id',
        'match_id_range': 0,
        'sample_id_name': 'id',
        'tag_value_delimiter': ':',
        'tag_with_value': False,
        'weight_type': 'float32'}

pipeline = FateFlowPipeline().set_roles(
    local="0")
pipeline.set_site_role("local")
pipeline.set_site_party_id("0")

pipeline.transform_local_file_to_dataframe(  # file="${abs_path_of_data_host}",
    file="/Users/yuwu/PycharmProjects/FATE/examples/data/breast_hetero_host.csv",
    meta=meta, head=True,
    namespace="experiment",
    name="breast_hetero_host")
