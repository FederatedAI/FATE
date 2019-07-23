#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from sklearn.metrics import roc_curve, precision_score
from sklearn.metrics import recall_score

from federatedml.evaluation.evaluation import Evaluation


class TestEvaluationBinary(unittest.TestCase):
    def setUp(self):
        self.data_num = 50
        final_result = []

        # for i in range(self.data_num):
        #     tmp = [np.random.choice([0, 1]), np.random.random(), np.random.choice([0, 1]), np.random.choice([0, 1]),
        #            "train"]
        #     tmp_pair = (str(i), tmp)
        #     final_result.append(tmp_pair)
        #
        # self.table = eggroll.parallelize(final_result,
        #                                  include_key=True,
        #                                  partition=10)

        self.model_name = 'Evaluation'
        self.args = {"data": {self.model_name: {"data": None}}}

    def _make_param_dict(self):
        component_param = {
            "EvaluateParam": {
                "eval_type": "binary",
                "pos_label": 1
            }
        }

        return component_param

    def test_evaluation(self):
        self.eval_obj = Evaluation()
        component_param = self._make_param_dict()
        self.eval_obj.run(component_param, self.args)

        ### start test
        self.auc_test()
        self.ks_test()
        self.lift_test()
        self.precision_test()
        self.recall_test()
        self.accuracy_test()

    def assertFloatEqual(self, op1, op2):
        diff = np.abs(op1 - op2)
        self.assertLess(diff, 1e-6)

    def auc_test(self):
        y_true = np.array([0, 0, 1, 1])
        y_predict = np.array([0.1, 0.4, 0.35, 0.8])
        ground_true_auc = 0.75

        auc = self.eval_obj.auc(y_true, y_predict)
        auc = round(auc, 2)

        self.assertFloatEqual(auc, ground_true_auc)

    def ks_test(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        y_predict = np.array(
            [0.42, 0.73, 0.55, 0.37, 0.57, 0.70, 0.25, 0.23, 0.46, 0.62, 0.76, 0.46, 0.55, 0.56, 0.56, 0.38, 0.37, 0.73,
             0.77, 0.21, 0.39])
        ground_true_ks = 0.75

        sk_fpr, sk_tpr ,sk_threshold = roc_curve(y_true, y_predict, drop_intermediate=0)

        ks, fpr, tpr, threshold = self.eval_obj.ks(y_true, y_predict)
        ks = round(ks, 2)

        self.assertFloatEqual(ks, ground_true_ks)
        self.assertListEqual(fpr, list(sk_fpr))
        self.assertListEqual(tpr, list(sk_tpr))
        self.assertListEqual(threshold, list(sk_threshold))

    def lift_test(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.30, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        dict_score = {"0": {0: 0, 1: 1}, "0.4": {0: 2, 1: 1.43}, "0.6": {0: 1.43, 1: 2}}

        split_thresholds = [0, 0.4, 0.6]

        lifts, thresholds = self.eval_obj.lift(y_true, y_predict, thresholds=split_thresholds)
        fix_lifts = []
        for lift in lifts:
            fix_lift = [round(pos, 2) for pos in lift]
            fix_lifts.append(fix_lift)

        for i in range(len(split_thresholds)):
            score_0 = dict_score[str(split_thresholds[i])][0]
            score_1 = dict_score[str(split_thresholds[i])][1]

            pos_lift = fix_lifts[i]
            self.assertEqual(len(pos_lift), 2)
            self.assertFloatEqual(score_0, pos_lift[0])
            self.assertFloatEqual(score_1, pos_lift[1])

        self.assertListEqual(split_thresholds, thresholds)

    def precision_test(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.30, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        dict_score = {"0.4": {0: 1, 1: 0.71}, "0.6": {0: 0.71, 1: 1}}

        split_thresholds = [0.4, 0.6]

        prec_values, thresholds = self.eval_obj.precision(y_true, y_predict, thresholds=split_thresholds)
        fix_prec_values = []
        for prec_value in prec_values:
            fix_prec_value = [round(pos, 2) for pos in prec_value]
            fix_prec_values.append(fix_prec_value)

        for i in range(len(split_thresholds)):
            score_0 = dict_score[str(split_thresholds[i])][0]
            score_1 = dict_score[str(split_thresholds[i])][1]

            pos_prec_value = fix_prec_values[i]
            self.assertEqual(len(pos_prec_value), 2)
            self.assertFloatEqual(score_0, pos_prec_value[0])
            self.assertFloatEqual(score_1, pos_prec_value[1])
        self.assertListEqual(thresholds, split_thresholds)

    def recall_test(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.31, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        dict_score = {"0.3": {0: 0.2, 1: 1}, "0.4": {0: 0.6, 1: 1}}

        split_thresholds = [0.3, 0.4]

        recalls, thresholds = self.eval_obj.recall(y_true, y_predict, thresholds=split_thresholds)
        round_recalls = []
        for recall in recalls:
            round_recall = [round(pos, 2) for pos in recall]
            round_recalls.append(round_recall)

        for i in range(len(split_thresholds)):
            score_0 = dict_score[str(split_thresholds[i])][0]
            score_1 = dict_score[str(split_thresholds[i])][1]

            pos_recall = round_recalls[i]
            self.assertEqual(len(pos_recall), 2)
            self.assertFloatEqual(score_0, pos_recall[0])
            self.assertFloatEqual(score_1, pos_recall[1])

        self.assertListEqual(thresholds, split_thresholds)

    def accuracy_test(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.31, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        gt_score = {"0.3": 0.6, "0.5": 1.0, "0.7": 0.7}

        split_thresholds = [0.3, 0.5, 0.7]

        acc, thresholds = self.eval_obj.accuracy(y_true, y_predict, thresholds=split_thresholds)
        for i in range(len(split_thresholds)):
            score = gt_score[str(split_thresholds[i])]
            self.assertFloatEqual(score, acc[i])

        self.assertListEqual(thresholds, split_thresholds)

    def tearDown(self):
        # self.table.destroy()
        pass

class TestEvaluationMulti(unittest.TestCase):
    def setUp(self):
        self.data_num = 50
        # final_result = []

        # for i in range(self.data_num):
        #     tmp = [np.random.choice([0, 1]), np.random.random(), np.random.choice([0, 1]), np.random.choice([0, 1]),
        #            "train"]
        #     tmp_pair = (str(i), tmp)
        #     final_result.append(tmp_pair)
        #
        # self.table = eggroll.parallelize(final_result,
        #                                  include_key=True,
        #                                  partition=10)

        self.model_name = 'Evaluation'
        self.args = {"data": {self.model_name: {"data": None}}}

    def _make_param_dict(self):
        component_param = {
            "EvaluateParam": {
                "eval_type": "multi",
                "pos_label": 1
            }
        }

        return component_param

    def test_evaluation(self):
        self.eval_obj = Evaluation()
        component_param = self._make_param_dict()
        self.eval_obj.run(component_param, self.args)

        ### start test
        self.accuracy_test()
        self.precision_test()
        self.recall_test()

    def assertFloatEqual(self, op1, op2):
        diff = np.abs(op1 - op2)
        self.assertLess(diff, 1e-6)

    def accuracy_test(self):
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        y_predict = [1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4]
        gt_score = 0.6
        gt_number = 12

        acc = self.eval_obj.accuracy(y_true, y_predict)
        self.assertFloatEqual(gt_score, acc)
        acc_number = self.eval_obj.accuracy(y_true, y_predict, normalize=False)
        self.assertEqual(acc_number, gt_number)

    def recall_test(self):
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        y_predict = np.array([1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6])
        sk_recall = recall_score(y_true, y_predict, average=None)
        gt_labels = [1,2,3,4,5,6]

        recalls, all_labels = self.eval_obj.recall(y_true, y_predict)

        self.assertListEqual(list(recalls), list(sk_recall))
        self.assertListEqual(all_labels, gt_labels)

    def precision_test(self):
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        y_predict = np.array([1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6])
        sk_precision = precision_score(y_true, y_predict, average=None)
        gt_labels = [1, 2, 3, 4, 5, 6]

        precision, all_labels = self.eval_obj.precision(y_true, y_predict)

        self.assertListEqual(list(precision), list(sk_precision))
        self.assertListEqual(all_labels, gt_labels)

    def tearDown(self):
        # self.table.destroy()
        pass

class TestEvaluationRegression(unittest.TestCase):
    def setUp(self):
        self.data_num = 50
        # final_result = []

        # for i in range(self.data_num):
        #     tmp = [np.random.choice([0, 1]), np.random.random(), np.random.choice([0, 1]), np.random.choice([0, 1]),
        #            "train"]
        #     tmp_pair = (str(i), tmp)
        #     final_result.append(tmp_pair)
        #
        # self.table = eggroll.parallelize(final_result,
        #                                  include_key=True,
        #                                  partition=10)

        self.model_name = 'Evaluation'
        self.args = {"data": {self.model_name: {"data": None}}}

    def _make_param_dict(self):
        component_param = {
            "EvaluateParam": {
                "eval_type": "regression",
                "pos_label": 1
            }
        }

        return component_param

    def test_evaluation(self):
        self.eval_obj = Evaluation()
        component_param = self._make_param_dict()
        self.eval_obj.run(component_param, self.args)

        ### start test
        self.explained_variance_test()
        self.mean_absolute_error_test()
        self.mean_squared_error_test()
        self.mean_squared_log_error_test()
        self.median_absolute_error_test()
        self.root_mean_squared_error_test()

    def assertFloatEqual(self, op1, op2):
        diff = np.abs(op1 - op2)
        self.assertLess(diff, 1e-6)

    def explained_variance_test(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(np.around(self.eval_obj.explained_variance(y_true, y_pred), 4), 0.9572)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(np.around(self.eval_obj.explained_variance(y_true, y_pred), 4), 0.9839)

    def mean_absolute_error_test(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(self.eval_obj.mean_absolute_error(y_true, y_pred), 0.5)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(self.eval_obj.mean_absolute_error(y_true, y_pred), 0.75)

    def mean_squared_error_test(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(self.eval_obj.mean_squared_error(y_true, y_pred), 0.375)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(np.around(self.eval_obj.mean_squared_error(y_true, y_pred), 4), 0.7083)

    def mean_squared_log_error_test(self):
        y_true = [3, 5, 2.5, 7]
        y_pred = [2.5, 5, 4, 8]
        self.assertFloatEqual(np.around(self.eval_obj.mean_squared_log_error(y_true, y_pred), 4), 0.0397)

        y_true = [[0.5, 1], [1, 2], [7, 6]]
        y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
        self.assertFloatEqual(np.around(self.eval_obj.mean_squared_log_error(y_true, y_pred), 4), 0.0442)

    def median_absolute_error_test(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(self.eval_obj.median_absolute_error(y_true, y_pred), 0.5)

        y_true = [3, -0.6, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(self.eval_obj.median_absolute_error(y_true, y_pred), 0.55)

    def root_mean_squared_error_test(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(np.around(self.eval_obj.root_mean_squared_error(y_true, y_pred), 4), 0.6124)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(np.around(self.eval_obj.root_mean_squared_error(y_true, y_pred), 4), 0.8416)

    def tearDown(self):
        # self.table.destroy()
        pass



if __name__ == '__main__':
    unittest.main()
