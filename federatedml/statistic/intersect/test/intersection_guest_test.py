import unittest
import numpy as np

import time
import uuid

from arch.api import session
from federatedml.param.intersect_param import IntersectParam

class TestRsaIntersectGuest(unittest.TestCase):
    def setUp(self):
        self.jobid = str(uuid.uuid1())
        session.init(self.jobid)
        
        from federatedml.statistic.intersect.intersect_guest import RsaIntersectionGuest
        from federatedml.statistic.intersect.intersect import RsaIntersect
        intersect_param = IntersectParam()
        self.rsa_operator = RsaIntersectionGuest(intersect_param)
        self.rsa_op2 = RsaIntersect(intersect_param)

    def data_to_eggroll_table(self, data):
        return session.parallelize(data, include_key=True, partition=2)

    def test_func_map_raw_id_to_encrypt_id(self):
        d1 = [("a", 1), ("b", 2), ("c", 3)]
        d2 = [(4, "a"), (5, "b"), (6, "c")]
        D1 = self.data_to_eggroll_table(d1)
        D2 = self.data_to_eggroll_table(d2)

        res = self.rsa_operator.map_raw_id_to_encrypt_id(D1, D2)

        gt = [(4,"id"),(5,"id"),(6,"id")]
        self.assertListEqual(list(res.collect()), gt)

    def test_hash(self):
        res = str(self.rsa_op2.hash("1"))
        self.assertEqual(res, "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b")
        

    
    def tearDown(self):
        session.stop()
        try:
            session.cleanup("*", self.jobid, True)
        except EnvironmentError:
            pass
        try:
            session.cleanup("*", self.jobid, False)
        except EnvironmentError:
            pass

if __name__ == "__main__":
    unittest.main()
