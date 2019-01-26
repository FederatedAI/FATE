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

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from federatedml.util import consts

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()
class Evaluation(object):
    def __init__(self, classi_type='binary'):
        self.classi_type = classi_type

    def report(self, labels, pred_scores, metrics, thresholds=None, pos_label=None):
        if metrics is None:
            LOGGER.warning("Not metrics can be found in evaluation, return None")
            return None

        if thresholds is None:
            thresholds = [0.5]

        eval_res = {}
        new_labels = []
        new_pred_scores = []

        for i in range(labels.shape[0]):
            if labels[i] is not None:
                if self.classi_type == consts.BINARY and pos_label is not None:
                    if pos_label == labels[i]:
                        new_labels.append(1)
                    else:
                        new_labels.append(0)
                else:
                    new_labels.append(labels[i])
                new_pred_scores.append(pred_scores[i])

        if len(new_labels) == 0:
            LOGGER.warning("Each of labels is None, can not evaluation!")
            for metric in metrics:
                eval_res[metric] = None
            return eval_res

        labels = np.array(new_labels)
        pred_scores = np.array(new_pred_scores)

        for metric in metrics:
            if metric == consts.AUC:
                eval_res[consts.AUC] = np.around(self.auc(labels, pred_scores), 4)
            elif metric == consts.KS:
                eval_res[consts.KS] = np.around(self.ks(labels, pred_scores), 4)
            elif metric == consts.LIFT:
                lifts = self.lift(labels, pred_scores, thresholds=thresholds)
                lifts = self.__evaluation_format_translate(lifts, thresholds, self.classi_type)
                eval_res[consts.LIFT] = lifts
            elif metric == consts.RECALL:
                recalls = self.recall(labels, pred_scores, thresholds=thresholds)
                recalls = self.__evaluation_format_translate(recalls, thresholds, self.classi_type)
                eval_res[consts.RECALL] = recalls
            elif metric == consts.PRECISION:
                precisions = self.precision(labels, pred_scores, thresholds=thresholds)
                precisions = self.__evaluation_format_translate(precisions, thresholds, self.classi_type)
                eval_res[consts.PRECISION] = precisions
            elif metric == consts.ACCURACY:
                accuracys = self.accuracy(labels, pred_scores, thresholds, normalize=True)
                # binary and multi have the same format of accuracy
                accuracys = self.__evaluation_format_translate(accuracys, thresholds)
                eval_res[consts.ACCURACY] = accuracys
            else:
                LOGGER.warning("can not find evaluation of " + str(metric))
        return eval_res

    def __evaluation_format_translate(self, results, thresholds, classi_type=consts.MULTY):
        evaluation_format = []
        for i in range(len(thresholds)):
            if classi_type == consts.BINARY:
                score = results[i][-1]
                if isinstance(score, float):
                    score = np.around(score, 4)
                res = (thresholds[i], score)
            else:
                res = ("score", np.around(results))
            evaluation_format.append(res)

        return evaluation_format

    def auc(self, labels, pred_scores):
        if self.classi_type == "binary":
            return roc_auc_score(labels, pred_scores)
        else:
            LOGGER.warning("auc is just suppose Binary Classification! return None as results")
            return None

    def ks(self, labels, pred_scores):
        if self.classi_type == "binary":
            fpr, tpr, thresholds = roc_curve(np.array(labels), pred_scores, drop_intermediate=0)
            return max(tpr - fpr)
        else:
            LOGGER.warning("ks is just suppose Binary Classification! return None as results")
            return None

    def lift(self, label, pred_scores_one_hot, thresholds=None):
        if thresholds is None:
            thresholds = [0.5]

        if self.classi_type == "binary":
            lift_operator = Lift()
            return lift_operator.compute(label, pred_scores_one_hot, thresholds=thresholds)
        else:
            LOGGER.warning("lift is just suppose Binary Classification! return None as results")
            return None

    def precision(self, labels, pred_scores, thresholds=None, result_filter=None):
        if self.classi_type == "binary":
            precision_operator = BiClassPrecision()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.classi_type == "multi":
            precision_operator = MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores, result_filter)
        else:
            LOGGER.warning("error:can not find classification type:" + self.classi_type)

    def recall(self, labels, pred_scores, thresholds=None, result_filter=None):
        if self.classi_type == "binary":
            precision_operator = BiClassRecall()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.classi_type == "multi":
            precision_operator = MultiClassRecall()
            return precision_operator.compute(labels, pred_scores, result_filter)
        else:
            LOGGER.warning("error:can not find classification type:" + self.classi_type)

    def accuracy(self, labels, pred_scores, thresholds=None, normalize=True):
        if self.classi_type == "binary":
            precision_operator = BiClassAccuracy()
            return precision_operator.compute(labels, pred_scores, thresholds, normalize)
        elif self.classi_type == "multi":
            precision_operator = MultiClassAccuracy()
            return precision_operator.compute(labels, pred_scores, normalize)
        else:
            LOGGER.warning("error:can not find classification type:" + self.classi_type)


class Lift(object):
    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value >= threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def __compute_lift(self, label, pred_scores_one_hot, pos_label="1"):
        tn, fp, fn, tp = confusion_matrix(label, pred_scores_one_hot).ravel()

        if pos_label == '0':
            tp, tn = tn, tp
            fp, fn = fn, fp

        if tp + fp == 0 or tp + fn == 0 or tp + tn + fp + fn == 0:
            lift = 0
            return lift

        pv_plus = tp / (tp + fp)
        pi1 = (tp + fn) / (tp + tn + fp + fn)
        lift = pv_plus / pi1
        return lift

    def compute(self, labels, pred_scores, thresholds=None):
        lifts = []

        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            label_type = ['0', '1']
            lift_type = []
            for lt in label_type:
                lift = self.__compute_lift(labels, pred_scores_one_hot, pos_label=lt)
                lift_type.append(lift)
            lifts.append(lift_type)

        return lifts


class BiClassPrecision(object):
    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value >= threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def compute(self, labels, pred_scores, thresholds):
        if thresholds is None:
            thresholds = [0.5]
        scores = []
        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            score = precision_score(labels, pred_scores_one_hot, average=None)
            scores.append(score)

        return scores


class MultiClassPrecision(object):
    def compute(self, labels, pred_scores, result_filter):
        scores = precision_score(labels, pred_scores, average=None)
        if result_filter is None:
            return scores

        scores = scores.tolist()

        label_type = []
        for label in labels:
            if label not in label_type:
                label_type.append(label)

        for pred_score in pred_scores:
            if pred_score not in label_type:
                label_type.append(pred_score)

        label_type.sort()
        if len(label_type) != len(scores):
            LOGGER.warning("label type size != scores size, exit")
            return -1

        label_key_pair = {}
        for i in range(len(label_type)):
            label_key_pair[label_type[i]] = i

        ret_scores = []

        for rf in result_filter:
            if rf in label_key_pair:
                ret_scores.append(scores[label_key_pair[rf]])
            else:
                ret_scores.append(-1)

        return ret_scores


class BiClassRecall(object):
    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value >= threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def compute(self, labels, pred_scores, thresholds):
        if thresholds is None:
            thresholds = [0.5]
        scores = []

        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            score = recall_score(labels, pred_scores_one_hot, average=None)
            scores.append(score)

        return scores


class MultiClassRecall(object):
    def compute(self, labels, pred_scores, result_filter=None):
        scores = recall_score(labels, pred_scores, average=None)
        if result_filter is None:
            return scores

        scores = scores.tolist()
        label_type = []
        for label in labels:
            if label not in label_type:
                label_type.append(label)

        for pred_score in pred_scores:
            if pred_score not in label_type:
                label_type.append(pred_score)

        label_type.sort()

        if len(label_type) != len(scores):
            LOGGER.warning("error:label type size != scores size, return -1 as result")
            return -1

        label_key_pair = {}
        for i in range(len(label_type)):
            label_key_pair[label_type[i]] = i

        ret_scores = []
        for rf in result_filter:
            if rf in label_key_pair:
                ret_scores.append(scores[label_key_pair[rf]])
            else:
                ret_scores.append(-1)

        return ret_scores


class BiClassAccuracy(object):
    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value >= threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def compute(self, labels, pred_scores, thresholds, normalize=True):
        if thresholds is None:
            thresholds = [0.5]
        scores = []
        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            score = accuracy_score(labels, pred_scores_one_hot, normalize)
            scores.append(score)

        return scores


class MultiClassAccuracy(object):
    def compute(self, labels, pred_scores, normalize=True):
        return accuracy_score(labels, pred_scores, normalize)
