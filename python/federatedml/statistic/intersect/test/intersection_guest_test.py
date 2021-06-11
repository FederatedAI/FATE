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
from federatedml.secureprotol.hash.hash_factory import Hash


class TestRsaIntersectGuest(unittest.TestCase):
    def setUp(self):
        self.jobid = str(uuid.uuid1())
        session.init(self.jobid)

        from federatedml.statistic.intersect import RsaIntersectionGuest
        from federatedml.statistic.intersect import RsaIntersect
        intersect_param = IntersectParam()
        self.rsa_operator = RsaIntersectionGuest()
        self.rsa_operator.load_params(intersect_param)
        self.rsa_op2 = RsaIntersect()
        self.rsa_op2.load_params(intersect_param)

    def data_to_table(self, data):
        return session.parallelize(data, include_key=True, partition=2)

    def test_func_map_raw_id_to_encrypt_id(self):
        d1 = [("a", 1), ("b", 2), ("c", 3)]
        d2 = [(4, "a"), (5, "b"), (6, "c")]
        D1 = self.data_to_table(d1)
        D2 = self.data_to_table(d2)

        res = self.rsa_operator.map_raw_id_to_encrypt_id(D1, D2)

        gt = [(4, "id"), (5, "id"), (6, "id")]
        self.assertListEqual(list(res.collect()), gt)

    def test_hash(self):
        hash_operator = Hash("sha256")
        res = str(self.rsa_op2.hash("1", hash_operator))
        self.assertEqual(res, "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b")

    def tearDown(self):
        session.stop()


if __name__ == "__main__":
    unittest.main()
