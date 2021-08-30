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
import uuid

from fate_arch.session import computing_session as session

from federatedml.param.intersect_param import IntersectParam


class TestRsaIntersectHost(unittest.TestCase):
    def setUp(self):
        self.jobid = str(uuid.uuid1())
        session.init(self.jobid)

        from federatedml.statistic.intersect import RsaIntersectionHost
        from federatedml.statistic.intersect import RawIntersectionHost
        intersect_param = IntersectParam()
        self.rsa_operator = RsaIntersectionHost()
        self.rsa_operator.load_params(intersect_param)
        self.raw_operator = RawIntersectionHost()
        self.raw_operator.load_params(intersect_param)

    def data_to_table(self, data):
        return session.parallelize(data, include_key=True, partition=2)

    def test_func_generate_rsa_key(self):
        res = self.rsa_operator.generate_rsa_key(1024)
        self.assertEqual(65537, res[0])

    def test_get_common_intersection(self):
        d1 = [(1, "a"), (2, "b"), (4, "c")]
        d2 = [(4, "a"), (5, "b"), (6, "c")]
        d3 = [(4, "a"), (5, "b"), (7, "c")]
        D1 = self.data_to_table(d1)
        D2 = self.data_to_table(d2)
        D3 = self.data_to_table(d3)

        res = self.raw_operator.get_common_intersection([D1, D2, D3])
        gt = [(4, "id")]
        self.assertListEqual(list(res.collect()), gt)

    def tearDown(self):
        session.stop()


if __name__ == "__main__":
    unittest.main()
