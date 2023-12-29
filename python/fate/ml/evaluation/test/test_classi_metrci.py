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

import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from fate.ml.evaluation.classification import *


class TestMetric(unittest.TestCase):
    def test_AUC(self):
        auc_metric = AUC()
        predict = np.random.random_sample(1000)
        label = np.random.randint(0, 2, 1000)
        result = auc_metric(predict, label)
        print(result.to_dict())
        self.assertEqual(result.metric_name, "auc")
        self.assertAlmostEqual(result.result, roc_auc_score(label, predict), places=7)

    def test_MultiAccuracy(self):
        multi_acc_metric = MultiAccuracy()
        predict = np.random.random_sample((1000, 3))
        label = np.random.randint(0, 3, 1000)
        result = multi_acc_metric(predict, label)
        print(result.to_dict())
        self.assertEqual(result.metric_name, "multi_accuracy")
        self.assertAlmostEqual(result.result, accuracy_score(label, predict.argmax(axis=-1)), places=7)

    def test_MultiRecall(self):
        multi_recall_metric = MultiRecall()
        predict = np.random.random_sample((1000, 3))
        label = np.random.randint(0, 3, 1000)
        result = multi_recall_metric(predict, label)
        print(result.to_dict())
        self.assertEqual(result.metric_name, "multi_recall")
        self.assertAlmostEqual(result.result, recall_score(label, predict.argmax(axis=-1), average="micro"), places=7)

    def test_MultiPrecision(self):
        multi_precision_metric = MultiPrecision()
        predict = np.random.random_sample((1000, 3))
        label = np.random.randint(0, 3, 1000)
        result = multi_precision_metric(predict, label)
        print(result.to_dict())
        self.assertEqual(result.metric_name, "multi_precision")
        self.assertAlmostEqual(
            result.result, precision_score(label, predict.argmax(axis=-1), average="micro"), places=7
        )

    def test_MultiF1Score(self):
        multi_f1_metric = MultiF1Score()
        predict = np.random.random_sample((1000, 3))
        label = np.random.randint(0, 3, 1000)
        result = multi_f1_metric(predict, label)
        print(result.to_dict())
        self.assertEqual(result.metric_name, "multi_f1_score")
        self.assertAlmostEqual(result.result, f1_score(label, predict.argmax(axis=-1), average="micro"), places=7)

    def test_BinaryMetrics(self):
        for metric_class, sklearn_metric in [
            (BinaryAccuracy, accuracy_score),
            (BinaryRecall, recall_score),
            (BinaryPrecision, precision_score),
            (BinaryF1Score, f1_score),
        ]:
            with self.subTest(metric_class=metric_class):
                metric = metric_class()
                predict = np.random.random_sample(1000)
                label = np.random.randint(0, 2, 1000)
                binary_predict = (predict > metric.threshold).astype(int)
                result = metric(predict, label)
                print(result.to_dict())
                self.assertEqual(result.metric_name, metric.metric_name)
                self.assertAlmostEqual(result.result, sklearn_metric(label, binary_predict), places=7)


if __name__ == "__main__":
    unittest.main()
