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

import argparse
import os
import sys
from enum import Enum

additional_path = os.path.realpath('../')
if additional_path not in sys.path:
    sys.path.append(additional_path)


class PearsonExample(Enum):
    DEFAULT = "default"
    HOST_ONLY = "host_only"
    SOLE = "sole"
    MIX_RAND = "mix_rand"

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return PearsonExample[s.upper()]
        except KeyError:
            raise ValueError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str, help="config file")
    parser.add_argument("-example", type=PearsonExample.from_string, required=True,
                        choices=list(PearsonExample.__iter__()),
                        help="example to run")
    args = parser.parse_args()
    kwargs = {}
    if args.config is not None:
        kwargs["config"] = args.config
    example: PearsonExample = args.example

    if example == PearsonExample.DEFAULT:
        from hetero_pearson.pipeline_hetero_pearson import main

        main(**kwargs)

    elif example == PearsonExample.SOLE:
        from hetero_pearson.pipeline_hetero_pearson_sole import main

        main(**kwargs)

    elif example == PearsonExample.HOST_ONLY:
        from hetero_pearson.pipeline_hetero_pearson_host_only import main

        main(**kwargs)

    elif example == PearsonExample.MIX_RAND:
        from hetero_pearson.pipeline_hetero_pearson_mix_rand import main

        main(**kwargs)

    else:
        raise NotImplementedError(example)
