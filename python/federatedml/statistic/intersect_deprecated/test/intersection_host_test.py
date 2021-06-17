import unittest
import uuid

from fate_arch.session import computing_session as session

from federatedml.param.intersect_param import IntersectParam


class TestRsaIntersectHost(unittest.TestCase):
    def setUp(self):
        self.jobid = str(uuid.uuid1())
        session.init(self.jobid)

        from federatedml.statistic.intersect_deprecated.intersect_host import RsaIntersectionHost
        from federatedml.statistic.intersect_deprecated.intersect_host import RawIntersectionHost
        intersect_param = IntersectParam()
        self.rsa_operator = RsaIntersectionHost(intersect_param)
        self.raw_operator = RawIntersectionHost(intersect_param)

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
