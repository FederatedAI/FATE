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

import numpy as np
import time
import random
import string
import unittest
from arch.api import eggroll
from arch.api import federation
from fate_flow.manager.tracking import Tracking 
from federatedml.feature.sampler import Sampler

class TestSampler(unittest.TestCase):
    def setUp(self):
        self.data = [(i * 10 + 5, i * i) for i in range(100)]
        self.table = eggroll.parallelize(self.data, include_key=True)
        self.args = {"data": 
                      {"sample_0": {
                       "data": self.table
                        }
                      } 
                    }

    def test_hetero_sample(self):
        component_param = {"SampleParam":
                            {"mode": "random",
                             "method": "downsample",
                             "fractions": 0.6,
                             "task_type": "hetero"
                            },
                            "local": 
                            {"role": "guest",
                             "party_id": 9999
                            }
                          }

        sampler = Sampler()
        tracker = Tracking("jobid", "host", 10000, "abc", "123")
        sampler.set_tracker(tracker)
        sampler.run(component_param, self.args)

        self.assertTrue(np.abs(len(list(sampler.save_data().collect())) - 60) < 10)


if __name__ == '__main__':
    eggroll.init("jobid")
    federation.init("jobid", 
                    {"local": {
                       "role": "guest",
                       "party_id": 9999
                    },
                     "role": {
                       "host": [
                           10000
                       ],
                       "guest": [
                           9999
                       ]
                     }
                    })
    unittest.main()
