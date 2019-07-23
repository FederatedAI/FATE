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

        self.binary_classification_support_func = [
            consts.AUC,
            consts.KS,
            consts.LIFT,
            consts.GAIN,
            consts.ACCURACY,
            consts.PRECISION,
            consts.RECALL,
            consts.ROC
        ]

        self.multi_classification_support_func = [
            consts.ACCURACY,
            consts.PRECISION,
            consts.RECALL
        ]

        self.metrics = {consts.BINARY: self.binary_classification_support_func,
                        consts.MULTY: self.multi_classification_support_func,
                        consts.REGRESSION: self.regression_support_func}

        self.round_num = 6
        FateStorage.init_storage()


    def _init_model(self, model):
        self.model_param = model
        self.eval_type = self.model_param.eval_type
        self.pos_label = self.model_param.pos_label
        self.filter_point_num = 100

    def _run_data(self, data_sets=None, stage=None):
        data = {}
        for data_key in data_sets:
            if data_sets[data_key].get("data", None):
                data[data_key] = data_sets[data_key]["data"]

        if stage == "fit":
            self.data_output = self.fit(data)
        else:
            LOGGER.warning("Evaluation has not transform, return")

    def fit(self, data):
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
            if len(eval_data_local[0][1]) >= 4:
                mode = eval_data_local[0][1][4]

            for d in eval_data_local:
                labels.append(d[1][0])
                pred_labels.append(d[1][1])
                pred_scores.append(d[1][2])

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

            try:
                metrics = self.metrics[self.eval_type]
            except:
                LOGGER.warning("Unknown eval_type of {}".format(self.eval_type))
                metrics = []

            for eval_metric in metrics:
                if None in pred_results:
                    continue
                res = getattr(self, eval_metric)(labels, pred_results)
                if res:
                    eval_result[eval_metric].append(mode)
                    eval_result[eval_metric].append(res)

            self.eval_results[data_type] = eval_result

    def __save_single_value(self, result, metric_name, metric_namespace, eval_name):
        self.tracker.log_metric_data(metric_namespace, metric_name, [Metric(eval_name, np.round(result, self.round_num))])
        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type="EVALUATION_SUMMARY"))

    def __filter(self, x_list, filter_num):
        x_size = len(x_list)
        index = [i for i in range(x_size)]
        if x_size > filter_num:
            np.random.shuffle(index)
            index = index[:filter_num]

        return index

    def __save_curve_data(self, x_axis_list, y_axis_list, metric_name, metric_namespace):
        points = []
        for i, value in enumerate(x_axis_list):
            if isinstance(value, float):
                value = np.round(value, self.round_num)
            points.append((value, np.round(y_axis_list[i], self.round_num)))
        points.sort(key=lambda x: x[0])

        metric_points = [Metric(point[0], point[1]) for point in points]
        self.tracker.log_metric_data(metric_namespace, metric_name, metric_points)

    def __save_curve_meta(self, metric_name, metric_namespace, metric_type, unit_name=None, ordinate_name=None,
                          curve_name=None, best=None, pair_type=None):
        extra_metas = {}
        metric_type = "_".join([metric_type, "EVALUATION"])

        key_list = ["unit_name", "ordinate_name", "curve_name", "best", "pair_type"]
        for key in key_list:
            value = locals()[key]
            if value:
                extra_metas[key] = value

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric_type, extra_metas=extra_metas))

    def __save_roc(self, data_type, metric_name, metric_namespace, metric_res):
        fpr, tpr, thresholds = metric_res
        index = self.__filter(thresholds, self.filter_point_num)
        # thresholds = [thresholds[i] for i in index]
        fpr = [fpr[i] for i in index]
        tpr = [tpr[i] for i in index]

        self.__save_curve_data(fpr, tpr, metric_name, metric_namespace)
        self.__save_curve_meta(metric_name=metric_name, metric_namespace=metric_namespace,
                               metric_type="ROC", unit_name="fpr", ordinate_name="tpr",
                               curve_name=data_type)

    def save_data(self):
        for (data_type, eval_res) in self.eval_results.items():
            precision_recall = {}
            for (metric, metric_res) in eval_res.items():
                metric_namespace = metric_res[0]
                metric_name = '_'.join([data_type, metric])

                if metric in self.save_single_value_metric_list:
                    self.__save_single_value(metric_res[1], metric_name=data_type, metric_namespace=metric_namespace,
                                             eval_name=metric)
                elif metric == consts.KS:
                    best_ks, fpr, tpr, thresholds = metric_res[1]
                    self.__save_single_value(best_ks, metric_name=data_type,
                                             metric_namespace=metric_namespace,
                                             eval_name=metric)

                    index = self.__filter(thresholds, self.filter_point_num)
                    thresholds = [thresholds[i] for i in index]
                    fpr = [fpr[i] for i in index]
                    tpr = [tpr[i] for i in index]

                    metric_name_fpr = '_'.join([metric_name, "fpr"])
                    self.__save_curve_data(thresholds, fpr, metric_name_fpr, metric_namespace)
                    self.__save_curve_meta(metric_name=metric_name_fpr, metric_namespace=metric_namespace,
                                           metric_type=metric.upper(), unit_name="threshold",
                                           curve_name=metric_name_fpr, pair_type=data_type)

                    metric_name_tpr = '_'.join([metric_name, "tpr"])
                    self.__save_curve_data(thresholds, tpr, metric_name_tpr, metric_namespace)
                    self.__save_curve_meta(metric_name_tpr, metric_namespace, metric.upper(), unit_name="threshold",
                                           curve_name=metric_name_tpr, pair_type=data_type)

                elif metric == consts.ROC:
                    self.__save_roc(data_type, metric_name, metric_namespace, metric_res[1])

                elif metric in [consts.ACCURACY, consts.LIFT, consts.GAIN]:
                    if self.eval_type == consts.MULTY and metric == consts.ACCURACY:
                        self.__save_single_value(metric_res[1], metric_name=data_type,
                                                 metric_namespace=metric_namespace,
                                                 eval_name=metric)
                        continue

                    score, thresholds = metric_res[1]

                    if metric in [consts.LIFT, consts.GAIN]:
                        score = [float(s[1]) for s in score]

                    index = self.__filter(thresholds, self.filter_point_num)
                    thresholds = [thresholds[i] for i in index]
                    score = [score[i] for i in index]

                    self.__save_curve_data(thresholds, score, metric_name, metric_namespace)
                    self.__save_curve_meta(metric_name=metric_name, metric_namespace=metric_namespace,
                                           metric_type=metric.upper(), unit_name="threshold",
                                           curve_name=data_type)
                elif metric in [consts.PRECISION, consts.RECALL]:
                    precision_recall[metric] = metric_res
                    if len(precision_recall) < 2:
                        continue

                    precision_res = precision_recall.get(consts.PRECISION)
                    recall_res = precision_recall.get(consts.RECALL)

                    if precision_res[0] != recall_res[0]:
                        LOGGER.warning(
                            "precision mode:{} is not equal to recall mode:{}".format(precision_res[0], recall_res[0]))
                        continue

                    metric_namespace = precision_res[0]
                    metric_name_precision = '_'.join([data_type, "precision"])

                    precision_thresholds = precision_res[1][1]
                    pos_precision_score = precision_res[1][0]
                    recall_thresholds = recall_res[1][1]
                    pos_recall_score = recall_res[1][0]

                    unit_name = "class"

                    # filter if the number of precision is lager than self.filter_point_num for binary classification
                    if self.eval_type == consts.BINARY:
                        index = self.__filter(precision_thresholds, self.filter_point_num)
                        precision_thresholds = [precision_thresholds[i] for i in index]
                        pos_precision_score = [score[1] for score in pos_precision_score]
                        pos_precision_score = [pos_precision_score[i] for i in index]

                        recall_thresholds = [recall_thresholds[i] for i in index]
                        pos_recall_score = [score[1] for score in pos_recall_score]
                        pos_recall_score = [pos_recall_score[i] for i in index]

                        unit_name = "threshold"

                    self.__save_curve_data(precision_thresholds, pos_precision_score, metric_name_precision,
                                           metric_namespace)
                    self.__save_curve_meta(metric_name_precision, metric_namespace, "_".join([consts.PRECISION.upper(), self.eval_type.upper()]),
                                           unit_name=unit_name, ordinate_name="Precision", curve_name=data_type,
                                           pair_type=data_type)

                    metric_name_recall = '_'.join([data_type, "recall"])
                    self.__save_curve_data(recall_thresholds, pos_recall_score, metric_name_recall,
                                           metric_namespace)
                    self.__save_curve_meta(metric_name_recall, metric_namespace, "_".join([consts.RECALL.upper(), self.eval_type.upper()]),
                                           unit_name=unit_name, ordinate_name="Recall", curve_name=data_type,
                                           pair_type=data_type)
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
            fpr, tpr, thresholds = list(map(float, fpr)), list(map(float, tpr)), list(map(float, thresholds))
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
            max_ks_interval = float(max(np.array(tpr) - np.array(fpr)))
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
        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassPrecision()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

    def recall(self, labels, pred_scores, thresholds=None):
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
        if thresholds is None and self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds.sort()

        if self.eval_type == consts.BINARY:
            precision_operator = BiClassRecall()
            return precision_operator.compute(labels, pred_scores, thresholds)
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassRecall()
            return precision_operator.compute(labels, pred_scores)
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
                gain = self.__compute_gain(labels, pred_scores_one_hot, pos_label=lt)
                gain_type.append(gain)
            gains.append(gain_type)

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
        scores = []
        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            score = list(map(float, precision_score(labels, pred_scores_one_hot, average=None)))
            scores.append(score)

        return scores, thresholds


class MultiClassPrecision(object):
    """
    Compute multi-classification precision
    """

    def compute(self, labels, pred_scores):
        all_labels = list(set(labels).union(set(pred_scores)))
        all_labels.sort()
        return precision_score(labels, pred_scores, average=None), all_labels


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
        scores = []

        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            score = list(map(float, recall_score(labels, pred_scores_one_hot, average=None)))
            scores.append(score)

        return scores, thresholds


class MultiClassRecall(object):
    """
    Compute multi-classification recall
    """

    def compute(self, labels, pred_scores):
        all_labels = list(set(labels).union(set(pred_scores)))
        all_labels.sort()
        return recall_score(labels, pred_scores, average=None), all_labels


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
