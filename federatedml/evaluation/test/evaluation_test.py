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
from sklearn.metrics import roc_auc_score
import numpy as np
import unittest

class TestClassificationEvaluaction(unittest.TestCase):
    def assertFloatEqual(self,op1, op2):
        diff = np.abs(op1 - op2)
        self.assertLess(diff, 1e-6)

    def test_auc(self):
        y_true = np.array([0,0,1,1])
        y_predict = np.array([0.1,0.4,0.35,0.8])
        ground_true_auc = 0.75

        eva = Evaluation("binary")
        auc = eva.auc(y_true,y_predict)
        auc = round(auc,2)

        self.assertFloatEqual(auc, ground_true_auc)

    def test_ks(self):
        y_true = np.array([1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0])
        y_predict = np.array([0.42,0.73,0.55,0.37,0.57,0.70,0.25,0.23,0.46,0.62,0.76,0.46,0.55,0.56,0.56,0.38,0.37,0.73,0.77,0.21,0.39])
        ground_true_ks = 0.75

        eva = Evaluation("binary")
        ks = eva.ks(y_true,y_predict)
        ks = round(ks,2)

        self.assertFloatEqual(ks, ground_true_ks)

    def test_lift(self):
        y_true = np.array([1,1,0,0,0,1,1,0,0,1])
        y_predict = np.array([0.57,0.70,0.25,0.30,0.46,0.62,0.76,0.46,0.35,0.56])
        dict_score = { "0":{0:0,1:1},"0.4":{0:2,1:1.43},"0.6":{0:1.43,1:2} }

        eva = Evaluation("binary")
        split_thresholds = [0,0.4,0.6]

        lifts = eva.lift(y_true,y_predict,thresholds=split_thresholds)
        fix_lifts = []
        for lift in lifts:
            fix_lift = [ round(pos,2) for pos in lift ]
            fix_lifts.append(fix_lift)

        for i in range(len(split_thresholds)):
            score_0 = dict_score[str(split_thresholds[i])][0]
            score_1 = dict_score[str(split_thresholds[i])][1]
            
            pos_lift = fix_lifts[i]
            self.assertEqual(len(pos_lift), 2)
            self.assertFloatEqual(score_0, pos_lift[0])
            self.assertFloatEqual(score_1, pos_lift[1])

    def test_precision(self):
        y_true = np.array([1,1,0,0,0,1,1,0,0,1])
        y_predict = np.array([0.57,0.70,0.25,0.30,0.46,0.62,0.76,0.46,0.35,0.56])
        dict_score = { "0.4":{0:1,1:0.71},"0.6":{0:0.71,1:1} }

        eva = Evaluation("binary")
        split_thresholds = [0.4,0.6]

        prec_values = eva.precision(y_true,y_predict,thresholds=split_thresholds)
        fix_prec_values = []
        for prec_value in prec_values:
            fix_prec_value = [ round(pos,2) for pos in prec_value ]
            fix_prec_values.append(fix_prec_value)

        for i in range(len(split_thresholds)):
            score_0 = dict_score[str(split_thresholds[i])][0]
            score_1 = dict_score[str(split_thresholds[i])][1]
            
            pos_prec_value = fix_prec_values[i]
            self.assertEqual(len(pos_prec_value), 2)
            self.assertFloatEqual(score_0, pos_prec_value[0])
            self.assertFloatEqual(score_1, pos_prec_value[1])


    def test_recall(self):
        y_true = np.array([1,1,0,0,0,1,1,0,0,1])
        y_predict = np.array([0.57,0.70,0.25,0.31,0.46,0.62,0.76,0.46,0.35,0.56])
        dict_score = { "0.3":{0:0.2,1:1},"0.4":{0:0.6,1:1} }

        eva = Evaluation("binary")
        split_thresholds = [0.3,0.4]

        recalls = eva.recall(y_true,y_predict,thresholds=split_thresholds)
        round_recalls = []
        for recall in recalls:
            round_recall = [ round(pos,2) for pos in recall ]
            round_recalls.append(round_recall)

        for i in range(len(split_thresholds)):
            score_0 = dict_score[str(split_thresholds[i])][0]
            score_1 = dict_score[str(split_thresholds[i])][1]
            
            pos_recall = round_recalls[i]
            self.assertEqual(len(pos_recall), 2)
            self.assertFloatEqual(score_0, pos_recall[0])
            self.assertFloatEqual(score_1, pos_recall[1])
        
    def test_bin_accuracy(self):
        y_true = np.array([1,1,0,0,0,1,1,0,0,1])
        y_predict = np.array([0.57,0.70,0.25,0.31,0.46,0.62,0.76,0.46,0.35,0.56])
        gt_score =  {"0.3":0.6, "0.5":1.0, "0.7":0.7 }
        
        split_thresholds = [0.3,0.5,0.7]
        eva = Evaluation("binary")
        
        acc = eva.accuracy(y_true,y_predict,thresholds=split_thresholds)
        for i in range(len(split_thresholds)):
            score = gt_score[str(split_thresholds[i])]
            self.assertFloatEqual(score, acc[i])
            
        
    def test_multi_accuracy(self):
        y_true = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
        y_predict = [1,1,2,2,3,2,1,1,1,1,3,3,3,3,2,4,4,4,4,4]
        gt_score =  0.6
        gt_number = 12
        eva = Evaluation("multi")
        
        acc = eva.accuracy(y_true,y_predict)
        self.assertFloatEqual(gt_score, acc)
        acc_number = eva.accuracy(y_true,y_predict,normalize=False)
        self.assertEqual(acc_number, gt_number)
    
    def test_multi_recall(self):
        y_true = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5])
        y_predict = np.array([1,1,2,2,3,2,1,1,1,1,3,3,3,3,2,4,4,4,4,4,6,6,6,6,6])
        gt_score =  {"1":0.4, "3":0.8, "4":1.0,"6":0,"7":-1}
        
        eva = Evaluation("multi")
        result_filter = [1,3,4,6,7]
        recall_scores = eva.recall(y_true,y_predict,result_filter=result_filter)
        
        for i in range(len(result_filter)):
            score = gt_score[str(result_filter[i])]
            self.assertFloatEqual(score, recall_scores[i])

    def test_multi_precision(self):
        y_true = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5])
        y_predict = np.array([1,1,2,2,3,2,1,1,1,1,3,3,3,3,2,4,4,4,4,4,6,6,6,6,6])
        gt_score =  {"2":0.25, "3":0.8, "5":0,"6":0,"7":-1}
        
        eva = Evaluation("multi")
        result_filter = [2,3,5,6,7]
        precision_scores = eva.precision(y_true,y_predict,result_filter=result_filter)
        
        for i in range(len(result_filter)):
            score = gt_score[str(result_filter[i])]
            self.assertFloatEqual(score, precision_scores[i])

if __name__ == '__main__':
    unittest.main()
