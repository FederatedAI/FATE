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
#

from pipeline.component.hetero_pearson import HeteroPearson
from pipeline.demo.hetero_pearson._common_component import run_pipeline, get_config

if __name__ == "__main__":
    config = get_config()
    hetero_pearson = HeteroPearson(name="hetero_pearson_0", column_indexes=-1, cross_parties=False)
    hetero_pearson.get_party_instance("guest", config.guest).algorithm_param(need_run=False)
    pipeline = run_pipeline(config=config,
                            guest_data={"name": "breast_hetero_guest", "namespace": "experiment"},
                            host_data={"name": "breast_hetero_host", "namespace": "experiment"},
                            hetero_pearson=hetero_pearson)
