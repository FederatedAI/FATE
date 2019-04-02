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
#

from federatedml.evaluation import Evaluation
import numpy as np
import unittest


class TestClassificationEvaluaction(unittest.TestCase):
    def assertFloatEqual(self, op1, op2):
        diff = np.abs(op1 - op2)
        self.assertLess(diff, 1e-6)

    def test_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_predict = np.array([0.1, 0.4, 0.35, 0.8])
        ground_true_auc = 0.75

        eva = Evaluation("binary")
        auc = eva.auc(y_true, y_predict)
        auc = round(auc, 2)

        self.assertFloatEqual(auc, ground_true_auc)

    def test_ks(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        y_predict = np.array(
            [0.42, 0.73, 0.55, 0.37, 0.57, 0.70, 0.25, 0.23, 0.46, 0.62, 0.76, 0.46, 0.55, 0.56, 0.56, 0.38, 0.37, 0.73,
             0.77, 0.21, 0.39])
        ground_true_ks = 0.75

        eva = Evaluation("binary")
        ks = eva.ks(y_true, y_predict)
        ks = round(ks, 2)

        self.assertFloatEqual(ks, ground_true_ks)

    def test_lift(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.30, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        dict_score = {"0": {0: 0, 1: 1}, "0.4": {0: 2, 1: 1.43}, "0.6": {0: 1.43, 1: 2}}

        eva = Evaluation("binary")
        split_thresholds = [0, 0.4, 0.6]

        lifts = eva.lift(y_true, y_predict, thresholds=split_thresholds)
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

    def test_precision(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.30, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        dict_score = {"0.4": {0: 1, 1: 0.71}, "0.6": {0: 0.71, 1: 1}}

        eva = Evaluation("binary")
        split_thresholds = [0.4, 0.6]

        prec_values = eva.precision(y_true, y_predict, thresholds=split_thresholds)
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

    def test_recall(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.31, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        dict_score = {"0.3": {0: 0.2, 1: 1}, "0.4": {0: 0.6, 1: 1}}

        eva = Evaluation("binary")
        split_thresholds = [0.3, 0.4]

        recalls = eva.recall(y_true, y_predict, thresholds=split_thresholds)
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

    def test_bin_accuracy(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.31, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])
        gt_score = {"0.3": 0.6, "0.5": 1.0, "0.7": 0.7}

        split_thresholds = [0.3, 0.5, 0.7]
        eva = Evaluation("binary")

        acc = eva.accuracy(y_true, y_predict, thresholds=split_thresholds)
        for i in range(len(split_thresholds)):
            score = gt_score[str(split_thresholds[i])]
            self.assertFloatEqual(score, acc[i])

    def test_multi_accuracy(self):
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        y_predict = [1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4]
        gt_score = 0.6
        gt_number = 12
        eva = Evaluation("multi")

        acc = eva.accuracy(y_true, y_predict)
        self.assertFloatEqual(gt_score, acc)
        acc_number = eva.accuracy(y_true, y_predict, normalize=False)
        self.assertEqual(acc_number, gt_number)

    def test_multi_recall(self):
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        y_predict = np.array([1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6])
        gt_score = {1: 0.4, 3: 0.8, 4: 1.0, 6: 0, 7: -1}

        eva = Evaluation("multi")
        result_filter = [1, 3, 4, 6, 7]
        recall_scores = eva.recall(y_true, y_predict, result_filter=result_filter)

        for i in range(len(result_filter)):
            score = gt_score[result_filter[i]]
            self.assertFloatEqual(score, recall_scores[result_filter[i]])

    def test_multi_precision(self):
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        y_predict = np.array([1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6])
        gt_score = {2: 0.25, 3: 0.8, 5: 0, 6: 0, 7: -1}

        eva = Evaluation("multi")
        result_filter = [2, 3, 5, 6, 7]
        precision_scores = eva.precision(y_true, y_predict, result_filter=result_filter)
        for i in range(len(result_filter)):
            score = gt_score[result_filter[i]]
            self.assertFloatEqual(score, precision_scores[result_filter[i]])

    def test_explained_variance(self):
        eva = Evaluation()

        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(np.around(eva.explain_variance(y_true, y_pred), 4), 0.9572)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(np.around(eva.explain_variance(y_true, y_pred), 4), 0.9839)

    def test_mean_absolute_error(self):
        eva = Evaluation()
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(eva.mean_absolute_error(y_true, y_pred), 0.5)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(eva.mean_absolute_error(y_true, y_pred), 0.75)

    def test_mean_squared_error(self):
        eva = Evaluation()
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(eva.mean_squared_error(y_true, y_pred), 0.375)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(np.around(eva.mean_squared_error(y_true, y_pred), 4), 0.7083)

    def test_mean_squared_log_error(self):
        eva = Evaluation()
        y_true = [3, 5, 2.5, 7]
        y_pred = [2.5, 5, 4, 8]
        self.assertFloatEqual(np.around(eva.mean_squared_log_error(y_true, y_pred), 4), 0.0397)

        y_true = [[0.5, 1], [1, 2], [7, 6]]
        y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
        self.assertFloatEqual(np.around(eva.mean_squared_log_error(y_true, y_pred), 4), 0.0442)

    def test_median_absolute_error(self):
        eva = Evaluation()
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(eva.median_absolute_error(y_true, y_pred), 0.5)

        y_true = [3, -0.6, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(eva.median_absolute_error(y_true, y_pred), 0.55)

    def test_root_mean_squared_error(self):
        eva = Evaluation()
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertFloatEqual(np.around(eva.root_mean_squared_error(y_true, y_pred), 4), 0.6124)

        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        self.assertFloatEqual(np.around(eva.root_mean_squared_error(y_true, y_pred), 4), 0.8416)

    def test_binary_report(self):
        eva = Evaluation("binary")
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.31, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])

        metrics = ["auc", "ks", "lift", "precision", "recall", "accuracy",
                   "explained_variance", "mean_absolute_error", "mean_squared_error",
                   "mean_squared_log_error", "median_absolute_error", "r2_score", "root_mean_squared_error"]

        eval_results = eva.report(y_true, y_predict, metrics)
        self.assertFloatEqual(eval_results['auc'], 1.0)
        self.assertFloatEqual(eval_results['ks'], 1.0)
        self.assertListEqual(eval_results['lift'], [(0.5, 2.0)])
        self.assertListEqual(eval_results['precision'], [(0.5, 1.0)])
        self.assertListEqual(eval_results['recall'], [(0.5, 1.0)])
        self.assertListEqual(eval_results['accuracy'], [(0.5, 1.0)])
        self.assertFloatEqual(eval_results['explained_variance'], 0.4501)
        self.assertFloatEqual(eval_results['mean_absolute_error'], 0.3620)
        self.assertFloatEqual(eval_results['mean_squared_error'], 0.1375)
        self.assertFloatEqual(eval_results['mean_squared_log_error'], 0.0707)
        self.assertFloatEqual(eval_results['median_absolute_error'], 0.3650)
        self.assertFloatEqual(eval_results['r2_score'], 0.4501)
        self.assertFloatEqual(eval_results['root_mean_squared_error'], 0.3708)

    def test_binary_report_with_pos_label(self):
        eva = Evaluation("binary")
        y_true = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
        y_predict = np.array([0.57, 0.70, 0.25, 0.31, 0.46, 0.62, 0.76, 0.46, 0.35, 0.56])

        metrics = ["auc", "ks", "lift", "precision", "recall", "accuracy",
                   "explained_variance", "mean_absolute_error", "mean_squared_error",
                   "mean_squared_log_error", "median_absolute_error", "r2_score", "root_mean_squared_error"]
        eval_results = eva.report(y_true, y_predict, metrics, pos_label=0)
        print(eval_results)
        self.assertFloatEqual(eval_results['auc'], 0.0)
        self.assertFloatEqual(eval_results['ks'], 0.0)
        self.assertListEqual(eval_results['lift'], [(0.5, 0.0)])
        self.assertListEqual(eval_results['precision'], [(0.5, 0.0)])
        self.assertListEqual(eval_results['recall'], [(0.5, 0.0)])
        self.assertListEqual(eval_results['accuracy'], [(0.5, 0.0)])
        self.assertFloatEqual(eval_results['explained_variance'], -0.6539)
        self.assertFloatEqual(eval_results['mean_absolute_error'], 0.6380)
        self.assertFloatEqual(eval_results['mean_squared_error'], 0.4135)
        self.assertFloatEqual(eval_results['mean_squared_log_error'], 0.1988)
        self.assertFloatEqual(eval_results['median_absolute_error'], 0.6350)
        self.assertFloatEqual(eval_results['r2_score'], -0.6539)
        self.assertFloatEqual(eval_results['root_mean_squared_error'], 0.643)
    
    def test_multi_report(self):
        eva = Evaluation("multi")
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        y_predict = np.array([1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6])

        metrics = ["auc", "ks", "lift", "precision", "recall", "accuracy",
                   "explained_variance", "mean_absolute_error", "mean_squared_error",
                   "mean_squared_log_error", "median_absolute_error", "r2_score", "root_mean_squared_error"]

        eval_results = eva.report(y_true, y_predict, metrics)
        self.assertIsNone(eval_results['auc'])
        self.assertIsNone(eval_results['ks'])
        self.assertIsNone(eval_results['lift'])
        self.assertDictEqual(eval_results['precision'], {1: 0.3333, 2: 0.25, 3: 0.8, 4: 1.0, 5: 0.0, 6: 0.0})
        self.assertDictEqual(eval_results['recall'], {1: 0.4, 2: 0.2, 3: 0.8, 4: 1.0, 5: 0.0, 6: 0.0})
        self.assertFloatEqual(eval_results['accuracy'], 0.48)
        self.assertFloatEqual(eval_results['explained_variance'], 0.6928)
        self.assertFloatEqual(eval_results['mean_absolute_error'], 0.5600)
        self.assertFloatEqual(eval_results['mean_squared_error'], 0.6400)
        self.assertFloatEqual(eval_results['mean_squared_log_error'], 0.0667)
        self.assertFloatEqual(eval_results['median_absolute_error'], 1.000)
        self.assertFloatEqual(eval_results['r2_score'], 0.6800)
    
    def test_multi_report_with_absent_value(self):
        eva = Evaluation("multi")
        y_true = np.array(   [1, 1, 1, 1, 1, None, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, None])
        y_predict = np.array([1, 1, 2, 2, 3, 3,2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6])

        metrics = ["auc", "ks", "lift", "precision", "recall", "accuracy",
                   "explained_variance", "mean_absolute_error", "mean_squared_error",
                   "mean_squared_log_error", "median_absolute_error", "r2_score", "root_mean_squared_error"]

        eval_results = eva.report(y_true, y_predict, metrics)
        self.assertIsNone(eval_results['auc'])
        self.assertIsNone(eval_results['ks'])
        self.assertIsNone(eval_results['lift'])
        self.assertDictEqual(eval_results['precision'], {1: 0.3333, 2: 0.25, 3: 0.8, 4: 1.0, 5: 0.0, 6: 0.0})
        self.assertDictEqual(eval_results['recall'], {1: 0.4, 2: 0.2, 3: 0.8, 4: 1.0, 5: 0.0, 6: 0.0})
        self.assertFloatEqual(eval_results['accuracy'], 0.48)
        self.assertFloatEqual(eval_results['explained_variance'], 0.6928)
        self.assertFloatEqual(eval_results['mean_absolute_error'], 0.5600)
        self.assertFloatEqual(eval_results['mean_squared_error'], 0.6400)
        self.assertFloatEqual(eval_results['mean_squared_log_error'], 0.0667)
        self.assertFloatEqual(eval_results['median_absolute_error'], 1.000)
        self.assertFloatEqual(eval_results['r2_score'], 0.6800)
        self.assertFloatEqual(eval_results['root_mean_squared_error'], 0.800)
        self.assertFloatEqual(eval_results['root_mean_squared_error'], 0.800)

    def test_regression_report(self):
        eva = Evaluation("regression")
        y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
        y_predict = np.array([1, 1, 2, 2, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6])

        metrics = ["auc", "ks", "lift", "precision", "recall", "accuracy",
                   "explained_variance", "mean_absolute_error", "mean_squared_error",
                   "mean_squared_log_error", "median_absolute_error", "r2_score", "root_mean_squared_error"]

        eval_results = eva.report(y_true, y_predict, metrics)
        self.assertFloatEqual(eval_results['explained_variance'], 0.6928)
        self.assertFloatEqual(eval_results['mean_absolute_error'], 0.5600)
        self.assertFloatEqual(eval_results['mean_squared_error'], 0.6400)
        self.assertFloatEqual(eval_results['mean_squared_log_error'], 0.0667)
        self.assertFloatEqual(eval_results['median_absolute_error'], 1.000)
        self.assertFloatEqual(eval_results['r2_score'], 0.6800)
        self.assertFloatEqual(eval_results['root_mean_squared_error'], 0.800)

        metrics = ["auc", "ks", "lift", "precision", "recall", "accuracy"]
        eval_results = eva.report(y_true, y_predict, metrics)
        self.assertIsNone(eval_results)


if __name__ == '__main__':
    unittest.main()
