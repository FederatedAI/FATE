import unittest
import numpy as np
import time
import copy
import uuid
from fate_arch.session import computing_session as session
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic.psi.psi import PSI
from federatedml.param.psi_param import PSIParam


class TestPSI(unittest.TestCase):

    def setUp(self):

        session.init('test', 0)
        print('generating dense tables')
        l1, l2 = [], []
        col = [i for i in range(20)]
        for i in range(100):
            inst = Instance()
            inst.features = np.random.random(20)
            l1.append(inst)
        for i in range(1000):
            inst = Instance()
            inst.features = np.random.random(20)
            l2.append(inst)
        self.dense_table1, self.dense_table2 = session.parallelize(l1, partition=4, include_key=False), \
            session.parallelize(l2, partition=4, include_key=False)
        self.dense_table1.schema['header'] = copy.deepcopy(col)
        self.dense_table2.schema['header'] = copy.deepcopy(col)
        print('generating done')

        print('generating sparse tables')
        l1, l2 = [], []
        col = [i for i in range(20)]
        for i in range(100):
            inst = Instance()
            inst.features = SparseVector(indices=copy.deepcopy(col), data=list(np.random.random(20)))
            l1.append(inst)
        for i in range(1000):
            inst = Instance()
            inst.features = SparseVector(indices=copy.deepcopy(col), data=list(np.random.random(20)))
            l2.append(inst)
        self.sp_table1, self.sp_table2 = session.parallelize(l1, partition=4, include_key=False), \
            session.parallelize(l2, partition=4, include_key=False)
        self.sp_table1.schema['header'] = copy.deepcopy(col)
        self.sp_table2.schema['header'] = copy.deepcopy(col)
        print('generating done')

    def test_dense_psi(self):

        param = PSIParam()
        psi = PSI()
        psi._init_model(param)
        psi.fit(self.dense_table1, self.dense_table2)
        print('dense testing done')

    def test_sparse_psi(self):

        param = PSIParam()
        psi = PSI()
        psi._init_model(param)
        psi.fit(self.sp_table1, self.sp_table2)
        print('dense testing done')


if __name__ == "__main__":
    unittest.main()
