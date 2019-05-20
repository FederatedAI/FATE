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
from collections import Iterable
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from arch.api.utils import log_utils
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Evaluation(object):
    def __init__(self, eval_type='binary'):
        self.eval_type = eval_type
        self.thresholds = None
        self.normalize = False
        self.eval_func = {
            consts.AUC: self.auc,
            consts.KS: self.ks,
            consts.LIFT: self.lift,
            consts.PRECISION: self.precision,
            consts.RECALL: self.recall,
            consts.ACCURACY: self.accuracy,
            consts.EXPLAINED_VARIANCE: self.explain_variance,
            consts.MEAN_ABSOLUTE_ERROR: self.mean_absolute_error,
            consts.MEAN_SQUARED_ERROR: self.mean_squared_error,
            consts.MEAN_SQUARED_LOG_ERROR: self.mean_squared_log_error,
            consts.MEDIAN_ABSOLUTE_ERROR: self.median_absolute_error,
            consts.R2_SCORE: self.r2_score,
            consts.ROOT_MEAN_SQUARED_ERROR: self.root_mean_squared_error
        }

        self.regression_support_func = [
            consts.EXPLAINED_VARIANCE,
            consts.MEAN_ABSOLUTE_ERROR,
            consts.MEAN_SQUARED_ERROR,
            consts.MEAN_SQUARED_LOG_ERROR,
            consts.MEDIAN_ABSOLUTE_ERROR,
            consts.R2_SCORE,
            consts.ROOT_MEAN_SQUARED_ERROR,
        ]

    def report(self, labels, pred_scores, metrics, thresholds=None, pos_label=None):
        """
        Define the report of each evaluation method in metrics.

        Parameters
        ----------
        labels : value list. The labels of data set.
        pred_scores : value list. The predict results of model. It should be corresponding to labels each data.
        metrics: str list. It includes one or several evaluations methods you want. The evaluation method include "auc", "ks", "lift",
                "precision", "recall", "accuracy", "explained_variance", "mean_absolute_error", "mean_squared_log_error",
                "median_absolute_error", "r2_score", "root_mean_squared_error".
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        pos_label: The same as DataIO label type. This parameter effective only for 'binary'. If input label in parameter labels is pos_label, it will be
                    set to 1, and set to 0 if not. If pos_label is None, do nothing to labels. Default None

        Returns
        ----------
        dict
            The key of return is element in metrics and the value is evaluation result. For instance, if metrics is ["auc", "precision"], thresholds is [0.5, 0.7],
            the return is { 'auc': 0.81, 'precision': [ (0.5, 0.77), (0.7, 0.66) ] }. (0.81, 0.77, 0.66 are examples)
        """
        if metrics is None:
            LOGGER.warning("Not metrics can be found in evaluation, return None")
            return None

        if self.eval_type == consts.REGRESSION:
            new_metrics = []
            for metric in metrics:
                if metric in self.regression_support_func:
                    new_metrics.append(metric)
            metrics = new_metrics
            if len(metrics) == 0:
                LOGGER.warning("Not metrics can be found in evaluation of regression, return None")
                return None

        self.thresholds = thresholds
        if self.thresholds is None and self.eval_type == consts.BINARY:
            self.thresholds = [0.5]
        elif self.eval_type == consts.MULTY:
            self.thresholds = None

        new_labels = []
        new_pred_scores = []
        for i in range(labels.shape[0]):
            if labels[i] is not None:
                if self.eval_type == consts.BINARY and pos_label is not None:
                    if pos_label == labels[i]:
                        new_labels.append(1)
                    else:
                        new_labels.append(0)
                else:
                    new_labels.append(labels[i])
                new_pred_scores.append(pred_scores[i])

        eval_res = {}
        if len(new_labels) == 0:
            LOGGER.warning("Each of labels is None, can not evaluation!")
            for metric in metrics:
                eval_res[metric] = None
            return eval_res

        labels = np.array(new_labels)
        pred_scores = np.array(new_pred_scores)
        for metric in metrics:
            if metric in self.eval_func:
                res = self.eval_func[metric](labels, pred_scores)
                eval_res[metric] = self.__evaluation_format_translate(res, self.thresholds, self.eval_type)
            else:
                LOGGER.warning("can not find evaluation of {}".format(metric))

        self.thresholds = None
        return eval_res

    def __evaluation_format_translate(self, results, thresholds, eval_type):
        """
        Transform evaluation result's format for output
        """
        if isinstance(results, float):
            return np.around(results, 4)
        else:
            evaluation_format = []
            if eval_type == consts.BINARY:
                for i in range(len(thresholds)):
                    if isinstance(results[i], Iterable):
                        score = results[i][-1]
                    else:
                        score = results[i]

                    if isinstance(score, float):
                        score = np.around(score, 4)
                    res = (thresholds[i], score)
                    evaluation_format.append(res)
            else:
                if isinstance(results, float):
                    results = np.around(results, 4)
                elif isinstance(results, dict):
                    for key in results:
                        if isinstance(results[key], float):
                            results[key] = np.around(results[key], 4)

                evaluation_format = results

            return evaluation_format

    def auc(self, labels, pred_scores):
        """
        Compute AUC for binary classification.

        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.

        Returns
        ----------
        float
            The AUC
        """
        if self.eval_type == consts.BINARY:
            return roc_auc_score(labels, pred_scores)
        else:
            LOGGER.warning("auc is just suppose Binary Classification! return None as results")
            return None

    def explain_variance(self, labels, pred_scores):
        """
        Compute explain variance
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.

        Returns
        ----------
        float
            The explain variance
        """
        return explained_variance_score(labels, pred_scores)

    def mean_absolute_error(self, labels, pred_scores):
        """
        Compute mean absolute error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A non-negative floating point.
        """
        return mean_absolute_error(labels, pred_scores)

    def mean_squared_error(self, labels, pred_scores):
        """
        Compute mean square error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A non-negative floating point value
        """
        return mean_squared_error(labels, pred_scores)

    def mean_squared_log_error(self, labels, pred_scores):
        """
        Compute mean squared logarithmic error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A non-negative floating point value
        """
        return mean_squared_log_error(labels, pred_scores)

    def median_absolute_error(self, labels, pred_scores):
        """
        Compute median absolute error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A positive floating point value
        """
        return median_absolute_error(labels, pred_scores)

    def r2_score(self, labels, pred_scores):
        """
        Compute R^2 (coefficient of determination) score
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            The R^2 score
        """
        return r2_score(labels, pred_scores)

    def root_mean_squared_error(self, labels, pred_scores):
        """
        Compute the root of mean square error
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Return
        ----------
        float
            A positive floating point value
        """
        return np.sqrt(mean_squared_error(labels, pred_scores))

    def ks(self, labels, pred_scores):
        """
        Compute Kolmogorov-Smirnov
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        float
            A positive floating point value
        """
        if self.eval_type == consts.BINARY:
            fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(pred_scores), drop_intermediate=0)
            return max(tpr - fpr)
        else:
            LOGGER.warning("ks is just suppose Binary Classification! return None as results")
            return None

    def lift(self, labels, pred_scores, thresholds=None):
        """
        Compute lift of binary classification.
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        Returns
        ----------
        float
            The lift
        """
        if thresholds is None:
            thresholds = self.thresholds

        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = [0.5]

        if self.eval_type == consts.BINARY:
            lift_operator = Lift()
            return lift_operator.compute(labels, pred_scores, thresholds=thresholds)
        else:
            LOGGER.warning("lift is just suppose Binary Classification! return None as results")
            return None

    def precision(self, labels, pred_scores, thresholds=None, result_filter=None):
        """
        Compute the precision
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        result_filter: value list. If result_filter is not None, it will filter the label results not in result_filter.
        Returns
        ----------
        dict
            The key is threshold and the value is another dic, which key is label in parameter labels, and value is the label's precision.
        """
        if thresholds is None:
            thresholds = self.thresholds

        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = [0.5]

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassPrecision()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores, result_filter)
        else:
            LOGGER.warning("error:can not find classification type:{}".format(self.eval_type))

    def recall(self, labels, pred_scores, thresholds=None, result_filter=None):
        """
        Compute the recall
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        result_filter: value list. If result_filter is not None, it will filter the label results not in result_filter.
        Returns
        ----------
        dict
            The key is threshold and the value is another dic, which key is label in parameter labels, and value is the label's recall.
        """
        if thresholds is None:
            thresholds = self.thresholds

        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = [0.5]

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassRecall()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassRecall()
            return precision_operator.compute(labels, pred_scores, result_filter)
        else:
            LOGGER.warning("error:can not find classification type:{}".format(self.eval_type))

    def accuracy(self, labels, pred_scores, thresholds=None, normalize=True):
        """
        Compute the accuracy
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        normalize: bool. If true, return the fraction of correctly classified samples, else returns the number of correctly classified samples
        Returns
        ----------
        dict
            the key is threshold and the value is the accuracy of this threshold.
        """
        if thresholds is None:
            thresholds = self.thresholds

        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = [0.5]

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassAccuracy()
            return precision_operator.compute(labels, pred_scores, thresholds, normalize)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassAccuracy()
            return precision_operator.compute(labels, pred_scores, normalize)
        else:
            LOGGER.warning("error:can not find classification type:".format(self.eval_type))


class Lift(object):
    """
    Compute lift
    """
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
    """
    Compute binary classification precision
    """
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
    """
    Compute multi-classification precision
    """
    def compute(self, labels, pred_scores, result_filter):
        scores = precision_score(labels, pred_scores, average=None)

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

        ret_scores = {}
        if result_filter is not None:
            for rf in result_filter:
                if rf in label_key_pair:
                    ret_scores[rf] = scores[label_key_pair[rf]]
                else:
                    ret_scores[rf] = -1
        else:
            for label in label_key_pair:
                ret_scores[label] = scores[label_key_pair[label]]

        return ret_scores


class BiClassRecall(object):
    """
    Compute binary classification recall
    """
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
    """
    Compute multi-classification recall
    """
    def compute(self, labels, pred_scores, result_filter=None):
        scores = recall_score(labels, pred_scores, average=None)

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

        ret_scores = {}
        if result_filter is not None:
            for rf in result_filter:
                if rf in label_key_pair:
                    ret_scores[rf] = scores[label_key_pair[rf]]
                else:
                    ret_scores[rf] = -1
        else:
            for label in label_key_pair:
                ret_scores[label] = scores[label_key_pair[label]]

        return ret_scores


class BiClassAccuracy(object):
    """
    Compute binary classification accuracy
    """
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
    """
    Compute multi-classification accuracy
    """
    def compute(self, labels, pred_scores, normalize=True):
        return accuracy_score(labels, pred_scores, normalize)
