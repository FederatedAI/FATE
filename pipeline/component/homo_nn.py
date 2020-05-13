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

from pipeline.component.component_base import Component
from pipeline.interface.output import Output
from federatedml.param.homo_nn_param import HomoNNParam


class HomoNN(Component, HomoNNParam):
    def __init__(self, **kwargs):
        Component.__init__(self, **kwargs)

        print (self.name)
        new_kwargs = self.erase_component_base_param(**kwargs)

        HomoNNParam.__init__(self, **new_kwargs)
        self.output = Output(self.name)
        self._module_name = "HomoNN"

    def summary(self, data, metric_keyword):
        if data is None:
            return
        # meta info
        metrics = {}
        for namespace in data:
            for name in data[namespace]:
                metric_data = data[namespace][name]["meta"]
                print(f"metric_data: {metric_data}")
                for metric_name, metric_val in metric_data.items():
                    if not metric_keyword or metric_name in metric_keyword:
                        metrics[metric_name] = metric_val

        for metric_name in metric_keyword:
            if metric_name not in metrics:
                metrics[metric_name] = None
        return metrics
