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
import os
import sys

additional_path = os.path.realpath('../')
if additional_path not in sys.path:
    sys.path.append(additional_path)

from hetero_pearson._common_component import run_pearson_pipeline, dataset


def main(config="../../config.yaml", namespace=""):
    common_param = dict(column_indexes=-1, use_mix_rand=True)
    pipeline = run_pearson_pipeline(config=config, namespace=namespace, data=dataset.breast, common_param=common_param)
    print(pipeline.get_component("hetero_pearson_0").get_model_param())
    print(pipeline.get_component("hetero_pearson_0").get_summary())
