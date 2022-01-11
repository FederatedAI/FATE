import unittest
import uuid

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.param.statistics_param import StatisticsParam
from federatedml.statistic.data_statistics import DataStatistics
from federatedml.feature.instance import Instance


class TestStatisticCpn(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def gen_data(self, data_num, partition):
        data = []
        header = [str(i) for i in range(2)]
        col_1 = np.random.randn(data_num)
        col_2 = np.random.rand(data_num)
        for key in range(data_num):
            data.append((key, Instance(features=np.array([col_1[key], col_2[key]]))))

        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.header = header
        self.col_1 = col_1
        self.col_2 = col_2
        return result

    def test_something(self):
        statistics_param = StatisticsParam(statistics="summary")
        statistics_param.check()
        print(statistics_param.statistics)
        test_data = self.gen_data(1000, 16)
        test_obj = DataStatistics()
        test_obj.model_param = statistics_param
        test_obj._init_model(statistics_param)
        test_obj.fit(test_data)
        static_result = test_obj.summary()
        stat_res_1 = static_result[self.header[0]]
        self.assertTrue(self._float_equal(stat_res_1['sum'], np.sum(self.col_1)))
        self.assertTrue(self._float_equal(stat_res_1['max'], np.max(self.col_1)))
        self.assertTrue(self._float_equal(stat_res_1['mean'], np.mean(self.col_1)))
        self.assertTrue(self._float_equal(stat_res_1['stddev'], np.std(self.col_1)))
        self.assertTrue(self._float_equal(stat_res_1['min'], np.min(self.col_1)))

        # self.assertEqual(True, False)

    def _float_equal(self, x, y, error=1e-6):
        if np.fabs(x - y) < error:
            return True
        print(f"x: {x}, y: {y}")
        return False

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
