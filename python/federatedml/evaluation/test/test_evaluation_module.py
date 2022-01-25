import unittest

import numpy as np
from federatedml.util import consts
from federatedml.evaluation.metrics import classification_metric, clustering_metric, regression_metric
from federatedml.evaluation.metric_interface import MetricInterface


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.bin_score = np.random.random(100)
        self.bin_label = (self.bin_score > 0.5) + 0

        self.reg_score = np.random.random(100) * 10
        self.reg_label = np.random.random(100) * 10

        self.multi_score = np.random.randint([4 for i in range(50)])
        self.multi_label = np.random.randint([4 for i in range(50)])

        self.clustering_score = np.random.randint([4 for i in range(50)])
        self.clustering_label = np.random.randint([3 for i in range(50)])

        self.psi_train_score = np.random.random(10000)
        self.psi_train_label = (self.psi_train_score > 0.5) + 0
        self.psi_val_score = np.random.random(1000)
        self.psi_val_label = (self.psi_val_score > 0.5) + 0

    def test_regression(self):
        print('testing regression metric')
        regression_metric.R2Score().compute(self.reg_score, self.reg_label)
        regression_metric.MSE().compute(self.reg_score, self.reg_label)
        regression_metric.RMSE().compute(self.reg_score, self.reg_label)
        regression_metric.ExplainedVariance().compute(self.reg_score, self.reg_label)
        regression_metric.Describe().compute(self.reg_score)

    def test_binary(self):
        print('testing binary')
        interface = MetricInterface(pos_label=1, eval_type=consts.BINARY)
        interface.auc(self.bin_label, self.bin_score)
        interface.confusion_mat(self.bin_label, self.bin_score)
        interface.ks(self.bin_label, self.bin_score)
        interface.accuracy(self.bin_label, self.bin_score)
        interface.f1_score(self.bin_label, self.bin_score)
        interface.gain(self.bin_label, self.bin_score)
        interface.lift(self.bin_label, self.bin_score)
        interface.quantile_pr(self.bin_label, self.bin_score)
        interface.precision(self.bin_label, self.bin_score)
        interface.recall(self.bin_label, self.bin_score)
        interface.roc(self.bin_label, self.bin_score)

    def test_psi(self):
        interface = MetricInterface(pos_label=1, eval_type=consts.BINARY)
        interface.psi(self.psi_train_score, self.psi_val_score, train_labels=self.psi_train_label,
                      validate_labels=self.psi_val_label)

    def test_multi(self):
        print('testing multi')
        interface = MetricInterface(eval_type=consts.MULTY, pos_label=1)
        interface.precision(self.multi_label, self.multi_score)
        interface.recall(self.multi_label, self.multi_score)
        interface.accuracy(self.multi_label, self.multi_score)

    def test_clustering(self):
        print('testing clustering')
        interface = MetricInterface(eval_type=consts.CLUSTERING, pos_label=1)
        interface.confusion_mat(self.clustering_label, self.clustering_score)

    def test_newly_added(self):
        print('testing newly added')
        binary_data = list(zip([i for i in range(len(self.psi_train_score))], self.psi_train_score))
        classification_metric.Distribution().compute(binary_data, binary_data)
        multi_data = list(zip([i for i in range(len(self.multi_score))], self.multi_score))
        classification_metric.Distribution().compute(multi_data, multi_data)

        classification_metric.KSTest().compute(self.multi_score, self.multi_score)
        classification_metric.KSTest().compute(self.psi_train_score, self.psi_val_score)

        classification_metric.AveragePrecisionScore().compute(self.psi_train_score, self.psi_val_score,
                                                              self.psi_train_label, self.psi_val_label)


if __name__ == '__main__':
    unittest.main()
