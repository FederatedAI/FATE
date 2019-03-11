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

import unittest

from federatedml.tree import Node, SplitInfo


class TestNode(unittest.TestCase):
    def setUp(self):
        pass

    def test_node(self):
        param_dict = {"id": 5, "sitename": "test", "fid": 55, "bid": 555,
                      "weight": -1, "is_leaf": True, "sum_grad": 2, "sum_hess": 3,
                      "left_nodeid": 6, "right_nodeid": 7}
        node = Node(id=5, sitename="test", fid=55, bid=555, weight=-1, is_leaf=True,
                    sum_grad=2, sum_hess=3, left_nodeid=6, right_nodeid=7)
        for key in param_dict:
            self.assertTrue(param_dict[key] == getattr(node, key))


class TestSplitInfo(unittest.TestCase):
    def setUp(self):
        pass

    def test_splitinfo(self):
        pass
        param_dict = {"sitename": "testsplitinfo",
                      "best_fid": 23, "best_bid": 233,
                      "sum_grad": 2333, "sum_hess": 23333, "gain": 233333}
        splitinfo = SplitInfo(sitename="testsplitinfo", best_fid=23, best_bid=233,
                              sum_grad=2333, sum_hess=23333, gain=233333)
        for key in param_dict:
            self.assertTrue(param_dict[key] == getattr(splitinfo, key))


if __name__ == '__main__':
    unittest.main()
