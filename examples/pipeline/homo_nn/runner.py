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

import sys
import os
import argparse
from enum import Enum

additional_path = os.path.realpath('../')
if additional_path not in sys.path:
    sys.path.append(additional_path)


class HomoNNExample(Enum):
    SINGLE_LAYER = "single_layer"
    MULTI_LAYER = "multi_layer"
    MULTI_LABEL = "multi_label"

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return HomoNNExample[s.upper()]
        except KeyError:
            raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    parser.add_argument("-example", type=HomoNNExample.from_string, required=True,
                        choices=list(HomoNNExample.__iter__()),
                        help="example to run")
    args = parser.parse_args()
    kwargs = {}
    if args.config is not None:
        kwargs["config"] = args.config
    example: HomoNNExample = args.example

    if example == HomoNNExample.SINGLE_LAYER:
        from homo_nn.pipeline_homo_nn_single_layer import main

        main(**kwargs)

    elif example == HomoNNExample.MULTI_LAYER:
        from homo_nn.pipeline_homo_nn_multy_layer import main

        main(**kwargs)

    elif example == HomoNNExample.MULTI_LABEL:
        from homo_nn.pipeline_homo_nn_multy_label import main

        main(**kwargs)

    else:
        raise NotImplementedError(example)
