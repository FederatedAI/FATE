from federatedml.one_vs_rest.one_vs_rest import OneVsRest

import unittest
import random
from arch.api import eggroll
from federatedml.feature.instance import Instance
from federatedml.util import consts
import time


class TestOneVsRest(unittest.TestCase):
    def setUp(self):
        label_num = [10, 15, 20, 25, 30]
        label = [0, 1, 2, 3, 4]

        label_set = []
        for i, value in enumerate(label):
            for _ in range(label_num[i]):
                label_set.append(value)

        random.shuffle(label_set)

        self.data = []
        for label in label_set:
            self.data.append(Instance(label=label))

        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.data_instances = self.data_to_eggroll_table(self.data, str_time)

        self.one_vs_rest_obj = OneVsRest("test", role=consts.GUEST, mode=consts.HETERO)

    def data_to_eggroll_table(self, data, jobid, partition=10, work_mode=1):
        eggroll.init(jobid, mode=work_mode)
        data_table = eggroll.parallelize(data, include_key=False, partition=partition)
        return data_table

    def test_get_data_classes(self):
        self.one_vs_rest_obj.fit(self.data_instances)

if __name__ == "__main__":
    unittest.main()