import unittest

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.statistic.data_statistics import DataStatistics
from federatedml.param.statistics_param import StatisticsParam
import uuid

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
            data.append((key, np.array([col_1[key], col_2[key]])))

        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.header = header
        return result

    def test_something(self):
        statistics_param = StatisticsParam(statistics="summary")
        statistics_param.check()
        print(statistics_param.statistics)
        test_data = self.gen_data(1000, 16)
        test_obj = DataStatistics()
        test_obj.model_param = statistics_param
        test_obj.fit(test_data)
        self.assertEqual(True, False)

    def tearDown(self):
        session.stop()
        try:
            session.cleanup("*", self.job_id, True)
        except EnvironmentError:
            pass
        try:
            session.cleanup("*", self.job_id, False)
        except EnvironmentError:
            pass


if __name__ == '__main__':
    unittest.main()
