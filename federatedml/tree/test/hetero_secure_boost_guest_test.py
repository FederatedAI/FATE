#
#  Copyright 2019 Toe FATE Authors. All Rights Reserved.
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
import sys
import unittest
from arch.api import eggroll
from arch.api import federation
from fate_flow.manager.tracking import Tracking 
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest


class TestHeteroSecureBoostGuest(unittest.TestCase):
    def setUp(self):
        self.data = []
        for i in range(100):
            dict = {}
            indices = []
            data = []
            for j in range(40):
                idx = random.randint(0, 49)
                if idx in dict:
                    continue
                dict[idx] = 1
                val = random.random()
                indices.append(idx)
                data.append(val)

            sparse_vec = SparseVector(indices, data, 50)
            self.data.append((i, Instance(features=sparse_vec, label=i % 2)))

        self.table = eggroll.parallelize(self.data, include_key=True)
        self.table.schema = {"header": ["fid" + str(i) for i in range(50)]}
        self.args = {"data": 
                      {"hetero_secure_boost_0": {
                       "train_data": self.table,
                       "eval_data": self.table
                        }
                      } 
                    }

    def test_hetero_sample(self):
        component_param = {"BoostingTreeParam": 
                            {"tree_param": 
                              {"max_depth": 3,
                               "min_leaf_node": 10
                              },
                             "task_type": "classification",
                             "objective_param":
                              {"objective": "cross_entropy"
                              },
                             "num_trees": 3,
                             "cv_param":{
                                 "n_splits": 3,
                                 "mode": "hetero",
                                 "role": "guest",
                                 "need_cv": True
                             }
                            }
                          }
       
        tree_guest = HeteroSecureBoostingTreeGuest()
        tracker = Tracking("jobid", "host", 10000, "abc", "123")
        tree_guest.set_tracker(tracker)
        tree_guest.run(component_param, self.args)
        tree_guest.save_model()


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
