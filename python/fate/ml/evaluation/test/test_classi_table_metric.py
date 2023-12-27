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


def generate_predict_and_label(num):
    predict = np.random.random_sample(num)
    label = np.random.randint(0, 2, num)
    return predict, label


class TestMetric(unittest.TestCase):

    def test_KS(self):
        ks_metric = KS()
        predict, label = generate_predict_and_label(1000)
        result = ks_metric(predict, label)
        print(result)
        print(result[0].to_dict())
        print(result[1].to_dict())

    def test_confusion_matrix(self):
        confusion_matrix_metric = ConfusionMatrix()
        predict, label = generate_predict_and_label(1000)
        result = confusion_matrix_metric(predict, label)
        print(result.to_dict())

    def test_gain(self):
        gain_metric = Gain()
        predict, label = generate_predict_and_label(1000)
        result = gain_metric(predict, label)
        print(result.to_dict())

    def test_lift(self):
        lift_metric = Lift()
        predict, label = generate_predict_and_label(1000)
        result = lift_metric(predict, label)
        print(result.to_dict())

    def test_bi_acc(self):
        bi_acc_metric = BiClassAccuracyTable()
        predict, label = generate_predict_and_label(1000)
        result = bi_acc_metric(predict, label)
        print(result.to_dict())

    def test_psi(self):
        psi_metric = PSI()
        predict, label = generate_predict_and_label(1000)
        predict2, label2 = generate_predict_and_label(1000)
        result = psi_metric({'train_scores': predict, 'validate_scores': predict2}, {'train_labels': label, 'validate_labels': label2})
        print('result is {}'.format(result))
        print(result[0].to_dict())
        print(result[1].to_dict())


if __name__ == '__main__':
    unittest.main()
