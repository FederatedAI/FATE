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

from collections import defaultdict
import numpy as np
import logging

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
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.manager.tracking import Tracking
from fate_flow.storage.fate_storage import FateStorage

from federatedml.param import EvaluateParam
from federatedml.util import consts
from federatedml.model_base import ModelBase

LOGGER = log_utils.getLogger()


class Evaluation(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = EvaluateParam()
        self.eval_results = {}

        self.eval_func = {
            consts.AUC: self.auc,
            consts.KS: self.ks,
            consts.LIFT: self.lift,
            consts.PRECISION: self.precision,
            consts.RECALL: self.recall,
            consts.ACCURACY: self.accuracy,
            consts.EXPLAINED_VARIANCE: self.explained_variance,
            consts.MEAN_ABSOLUTE_ERROR: self.mean_absolute_error,
            consts.MEAN_SQUARED_ERROR: self.mean_squared_error,
            consts.MEAN_SQUARED_LOG_ERROR: self.mean_squared_log_error,
            consts.MEDIAN_ABSOLUTE_ERROR: self.median_absolute_error,
            consts.R2_SCORE: self.r2_score,
            consts.ROOT_MEAN_SQUARED_ERROR: self.root_mean_squared_error,
            consts.ROC: self.roc,
            consts.GAIN: self.gain
        }

        self.save_single_value_metric_list = [consts.AUC,
                                              consts.EXPLAINED_VARIANCE,
                                              consts.MEAN_ABSOLUTE_ERROR,
                                              consts.MEAN_SQUARED_ERROR,
                                              consts.MEAN_SQUARED_LOG_ERROR,
                                              consts.MEDIAN_ABSOLUTE_ERROR,
                                              consts.R2_SCORE,
                                              consts.ROOT_MEAN_SQUARED_ERROR]
        self.save_curve_metric_list = [consts.KS, consts.ROC, consts.LIFT, consts.GAIN, consts.PRECISION, consts.RECALL,
                                       consts.ACCURACY]

        self.regression_support_func = [
            consts.EXPLAINED_VARIANCE,
            consts.MEAN_ABSOLUTE_ERROR,
            consts.MEAN_SQUARED_ERROR,
            consts.MEAN_SQUARED_LOG_ERROR,
            consts.MEDIAN_ABSOLUTE_ERROR,
            consts.R2_SCORE,
            consts.ROOT_MEAN_SQUARED_ERROR
        ]

        FateStorage.init_storage()
        self.tracker = Tracking('123456', 'hetero_lr')

    def _init_model(self, model):
        self.model_param = model
        self.eval_type = self.model_param.eval_type
        self.thresholds = self.model_param.thresholds
        self.metrics = self.model_param.metrics
        self.pos_label = self.model_param.pos_label

    def _run_data(self, data_sets=None, stage=None):
        data = {}
        for data_key in data_sets:
            if data_sets[data_key].get("data", None):
                data[data_key] = data_sets[data_key]["data"]

        if stage == "fit":
            self.data_output = self.fit(data)
        else:
            LOGGER.warning("Evaluation has not transform, return")

    def _param_check(self):
        if self.eval_type == consts.REGRESSION:
            new_metrics = []
            for metric in self.metrics:
                if metric in self.regression_support_func:
                    new_metrics.append(metric)
            metrics = new_metrics
            if len(metrics) == 0:
                LOGGER.warning("Not metrics can be found in evaluation of regression, return None")
                return False

        return True

    def fit(self, data):
        if not self._param_check():
            LOGGER.warning("Evaluation parameter checker may not be right, not evaluate and return None")
            return None

        if len(data) <= 0:
            return

        self.eval_results.clear()
        for (key, eval_data) in data.items():
            eval_data_local = list(eval_data.collect())

            labels = []
            pred_scores = []
            pred_labels = []

            data_type = key
            mode = "eval"
            if len(eval_data_local[0][1]) >= 3:
                mode = eval_data_local[0][1][3]

            for d in eval_data_local:
                labels.append(d[1][0])
                pred_scores.append(d[1][1])
                pred_labels.append(d[1][2])

            if self.eval_type == consts.BINARY or self.eval_type == consts.REGRESSION:
                if self.pos_label:
                    new_labels = []
                    for label in labels:
                        if self.pos_label == label:
                            new_labels.append(1)
                        else:
                            new_labels.append(0)
                    labels = new_labels

                pred_results = pred_scores
            else:
                pred_results = pred_labels

            eval_result = defaultdict(list)
            for eval_metric in self.model_param.metrics:
                res = getattr(self, eval_metric)(labels, pred_results)
                if res:
                    eval_result[eval_metric].append(mode)
                    eval_result[eval_metric].append(res)

            self.eval_results[data_type] = eval_result

    def __save_single_value(self, result, metric_name, metric_namespace, eval_name):
        self.tracker.log_metric_data(metric_namespace, metric_name, [Metric(eval_name, result)])

    def __save_curve_data(self, x_axis_list, y_axis_list, metric_name, metric_namespace):
        points = []
        for i, value in enumerate(x_axis_list):
            points.append((value, y_axis_list[i]))
        points.sort(key=lambda x: x[0])

        metric_points = [Metric(point[0], point[1]) for point in points]
        self.tracker.log_metric_data(metric_namespace, metric_name, metric_points)

    def __save_curve_meta(self, metric_name, metric_namespace, metric_type, unit_name=None, ordinate_name=None,
                          curve_name=None, best=None):
        extra_metas = {}
        key_list = ["unit_name", "unit_name", "ordinate_name", "best"]
        for key in key_list:
            value = locals()[key]
            if value:
                extra_metas[key] = value

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric_type, extra_metas=extra_metas))

    def save_data(self):
        for (data_type, eval_res) in self.eval_results.items():
            precision_recall = {}
            for (metric, metric_res) in eval_res.items():
                metric_name = '_'.join([data_type, metric])
                metric_namespace = metric_res[0]

                if metric in self.save_single_value_metric_list:
                    self.__save_single_value(metric_res[1], metric_name, metric_namespace, metric)
                elif metric == consts.KS or metric == consts.ROC:
                    if metric == consts.KS:
                        fpr, tpr, thresholds = metric_res[1][1:]
                    else:
                        fpr, tpr, thresholds = metric_res[1]

                    metric_name_fpr = '_'.join([metric_name, "fpr"])
                    self.__save_curve_data(thresholds, fpr, metric_name_fpr, metric_namespace)
                    self.__save_curve_meta(metric_name_fpr, metric_namespace, metric, unit_name="threshold",
                                           curve_name=metric_name_fpr)

                    metric_name_tpr = '_'.join([metric_name, "tpr"])
                    self.__save_curve_data(thresholds, tpr, metric_name_tpr, metric_namespace)
                    self.__save_curve_meta(metric_name_tpr, metric_namespace, metric, unit_name="threshold",
                                           curve_name=metric_name_tpr)

                elif metric in [consts.ACCURACY, consts.LIFT, consts.GAIN]:
                    score, thresholds = metric_res[1]
                    self.__save_curve_data(thresholds, score, metric_name, metric_namespace)
                    self.__save_curve_meta(metric_name, metric_namespace, metric, unit_name="threshold",
                                           curve_name=metric_name)
                elif metric in [consts.PRECISION, consts.RECALL]:
                    precision_recall[metric] = metric_res
                    if len(precision_recall) < 2:
                        continue

                    precision_res = precision_recall.get(consts.PRECISION)
                    recall_res = precision_recall.get(consts.RECALL)

                    if precision_res[0] != recall_res[0]:
                        raise ValueError(
                            "precision mode:{} is not equal to recall mode:{}".format(precision_res[0], recall_res[0]))
                    metric_namespace = precision_res[0]

                    metric_name_precision = '_'.join([data_type, "precision"])
                    self.__save_curve_data(precision_res[1][1], precision_res[1][0], metric_name_precision,
                                           metric_namespace)
                    self.__save_curve_meta(metric_name_precision, metric_namespace, consts.PRECISION,
                                           unit_name="threshold")

                    metric_name_recall = '_'.join([data_type, "recall"])
                    self.__save_curve_data(recall_res[1][1], recall_res[1][0], metric_name_recall,
                                           metric_namespace)
                    self.__save_curve_meta(metric_name_recall, metric_namespace, consts.RECALL,
                                           unit_name="threshold")
                else:
                    LOGGER.warning("Unknown metric:{}".format(metric))

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
            logging.warning("auc is just suppose Binary Classification! return None as results")
            return None

    def explained_variance(self, labels, pred_scores):
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

    def roc(self, labels, pred_scores):
        if self.eval_type == consts.BINARY:
            fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(pred_scores), drop_intermediate=0)
            fpr, tpr, thresholds = list(fpr), list(tpr), list(thresholds)
        else:
            logging.warning("roc_curve is just suppose Binary Classification! return None as results")
            fpr, tpr, thresholds = None, None, None

        return fpr, tpr, thresholds

    def ks(self, labels, pred_scores):
        """
        Compute Kolmogorov-Smirnov
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        max_ks_interval: float max value of each tpr - fpt
        fpr:
        """
        max_ks_interval = None
        fpr = None
        tpr = None
        thresholds = None
        if self.eval_type == consts.BINARY:
            fpr, tpr, thresholds = self.roc(labels, pred_scores)
            max_ks_interval = max(np.array(tpr) - np.array(fpr))
        else:
            logging.warning("ks is just suppose Binary Classification! return None as results")

        return max_ks_interval, fpr, tpr, thresholds

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
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            lift_operator = Lift()
            return lift_operator.compute(labels, pred_scores, thresholds=thresholds)
        else:
            logging.warning("lift is just suppose Binary Classification! return None as results")
            return None

    def gain(self, labels, pred_scores, thresholds=None):
        """
        Compute gain of binary classification.
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        thresholds: value list. This parameter effective only for 'binary'. The predict scores will be 1 if it larger than thresholds, if not,
                    if will be 0. If not only one threshold in it, it will return several results according to the thresholds. default None
        Returns
        ----------
        float
            The gain
        """
        if thresholds is None:
            thresholds = self.thresholds

        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            gain_operator = Gain()
            return gain_operator.compute(labels, pred_scores, thresholds=thresholds)
        else:
            logging.warning("gain is just suppose Binary Classification! return None as results")
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
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassPrecision()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores, result_filter)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

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
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassRecall()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassRecall()
            return precision_operator.compute(labels, pred_scores, result_filter)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

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
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassAccuracy()
            return precision_operator.compute(labels, pred_scores, thresholds, normalize)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassAccuracy()
            return precision_operator.compute(labels, pred_scores, normalize)
        else:
            logging.warning("error:can not find classification type:".format(self.eval_type))


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

        return lifts, thresholds


class Gain(object):
    """
    Compute Gain
    """

    def __init__(self):
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value >= threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def __compute_gain(self, label, pred_scores_one_hot, pos_label="1"):
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(label, pred_scores_one_hot).ravel()

        if pos_label == '0':
            self.tp, self.tn = self.tn, self.tp
            self.fp, self.fn = self.fn, self.fp

        if self.tp + self.fp == 0:
            gain = 0
        else:
            gain = self.tp / (self.tp + self.fp)
        return gain

    def compute(self, labels, pred_scores, thresholds=None):
        gains = []

        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            label_type = ['0', '1']
            gain_type = []
            for lt in label_type:
                lift = self.__compute_gain(labels, pred_scores_one_hot, pos_label=lt)
                gain_type.append(lift)
            gains.append(gains)

        return gains, thresholds


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

        return scores, thresholds


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
            logging.warning("label type size != scores size, exit")
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

        return scores, thresholds


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
            logging.warning("error:label type size != scores size, return -1 as result")
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

        return scores, thresholds


class MultiClassAccuracy(object):
    """
    Compute multi-classification accuracy
    """

    def compute(self, labels, pred_scores, normalize=True):
        return accuracy_score(labels, pred_scores, normalize)
