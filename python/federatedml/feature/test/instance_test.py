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

from federatedml.feature.instance import Instance


class TestInstance(unittest.TestCase):
    def setUp(self):
        pass

    def test_instance(self):
        inst = Instance(inst_id=5, weight=2.0, features=[1, 2, 3], label=-5)
        self.assertTrue(inst.inst_id == 5 and abs(inst.weight - 2.0) < 1e-8 \
                        and inst.features == [1, 2, 3] and inst.label == -5)

        inst.set_weight(3)
        inst.set_label(5)
        inst.set_feature(["yes", "no"])
        self.assertTrue(inst.weight == 3 and inst.label == 5 and inst.features == ["yes", "no"])


if __name__ == '__main__':
    unittest.main()
