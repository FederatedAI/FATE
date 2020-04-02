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
import sys
from collections import defaultdict
import math
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

from federatedml.param import EvaluateParam
from federatedml.util import consts
from federatedml.model_base import ModelBase

LOGGER = log_utils.getLogger()

class PerformanceRecorder():

    """
    This class record performance(single value metrics during the training process)
    """

    def __init__(self):

        # all of them are single value metrics
        self.allowed_metric = [consts.AUC,
                              consts.EXPLAINED_VARIANCE,
                              consts.MEAN_ABSOLUTE_ERROR,
                              consts.MEAN_SQUARED_ERROR,
                              consts.MEAN_SQUARED_LOG_ERROR,
                              consts.MEDIAN_ABSOLUTE_ERROR,
                              consts.R2_SCORE,
                              consts.ROOT_MEAN_SQUARED_ERROR,
                              consts.PRECISION,
                              consts.RECALL,
                              consts.ACCURACY,
                              consts.KS
                            ]


        self.larger_is_better = [consts.AUC,
                                 consts.R2_SCORE,
                                 consts.PRECISION,
                                 consts.RECALL,
                                 consts.EXPLAINED_VARIANCE,
                                 consts.ACCURACY,
                                 consts.KS
                                 ]

        self.smaller_is_better = [consts.ROOT_MEAN_SQUARED_ERROR,
                                  consts.MEAN_ABSOLUTE_ERROR,
                                  consts.MEAN_SQUARED_ERROR,
                                  consts.MEAN_SQUARED_LOG_ERROR]

        self.cur_best_performance = {}

        self.no_improvement_round = {} # record no improvement round of all metrics

    def has_improved(self, val: float, metric: str, cur_best: dict):

        if metric not in cur_best:
            return True

        if metric in self.larger_is_better and val > cur_best[metric]:
            return True

        elif metric in self.smaller_is_better and val < cur_best[metric]:
            return True

        return False

    def update(self, eval_dict: dict):
        """

        Parameters
        ----------
        eval_dict dict, {metric_name:metric_val}, e.g. {'auc':0.99}

        Returns stop flag, if should stop return True, else False
        -------
        """
        if len(eval_dict) == 0:
            return

        for metric in eval_dict:
            if metric not in self.allowed_metric:
                continue
            if self.has_improved(eval_dict[metric], metric, self.cur_best_performance):
                self.cur_best_performance[metric] = eval_dict[metric]
                self.no_improvement_round[metric] = 0
            else:
                self.no_improvement_round[metric] += 1


class Evaluation(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = EvaluateParam()
        self.eval_results = defaultdict(list)

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
	
	# record name of train and validate dataset
        self.validate_key = set()
        self.train_key = set()

        self.validate_metric = {}
        self.train_metric = {}

    def _init_model(self, model):
        self.model_param = model
        self.eval_type = self.model_param.eval_type
        self.pos_label = self.model_param.pos_label

    def _run_data(self, data_sets=None, stage=None):
        if not self.need_run:
            return

        data = {}
        for data_key in data_sets:
            if data_sets[data_key].get("data", None):
                data[data_key] = data_sets[data_key]["data"]

        if stage == "fit":
            self.data_output = self.fit(data)
        else:
            LOGGER.warning("Evaluation has not transform, return")

    def split_data_with_type(self, data: list) -> dict:

        split_result = defaultdict(list)
        for value in data:
            mode = value[1][4]
            split_result[mode].append(value)

        return split_result

    def evaluate_metircs(self, mode: str, data: list) -> dict:
        labels = []
        pred_scores = []
        pred_labels = []

        for d in data:
            labels.append(d[1][0])
            pred_labels.append(d[1][1])
            pred_scores.append(d[1][2])

        if self.eval_type == consts.BINARY or self.eval_type == consts.REGRESSION:
            if self.pos_label and self.eval_type == consts.BINARY:
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

        if self.eval_type in self.metrics:
            metrics = self.metrics[self.eval_type]
        else:
            LOGGER.warning("Unknown eval_type of {}".format(self.eval_type))
            metrics = []

        for eval_metric in metrics:
            res = getattr(self, eval_metric)(labels, pred_results)
            if res is not None:
                try:
                    if math.isinf(res):
                        res = float(-9999999)
                        LOGGER.info("res is inf, set to {}".format(res))
                except:
                    pass
                   
                eval_result[eval_metric].append(mode)
                eval_result[eval_metric].append(res)

        return eval_result

    def fit(self, data, return_result=False):
        if len(data) <= 0:
            return

        self.eval_results.clear()
        for (key, eval_data) in data.items():
            eval_data_local = list(eval_data.collect())
            split_data_with_label = self.split_data_with_type(eval_data_local)
            for mode, data in split_data_with_label.items():
                eval_result = self.evaluate_metircs(mode, data)
                self.eval_results[key].append(eval_result)
                LOGGER.debug('mode is {}'.format(mode))
                if mode == 'validate':
                    self.validate_key.add(key)
                elif mode == 'train':
                    self.train_key.add(key)

        return self.callback_metric_data(return_single_val_metrics=return_result)

    def __save_single_value(self, result, metric_name, metric_namespace, eval_name):
        self.tracker.log_metric_data(metric_namespace, metric_name,
                                     [Metric(eval_name, np.round(result, self.round_num))])
        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type="EVALUATION_SUMMARY"))

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
                          curve_name=None, best=None, pair_type=None, thresholds=None):
        extra_metas = {}
        metric_type = "_".join([metric_type, "EVALUATION"])

        key_list = ["unit_name", "ordinate_name", "curve_name", "best", "pair_type", "thresholds"]
        for key in key_list:
            value = locals()[key]
            if value:
                if key == "thresholds":
                    value = np.round(value, self.round_num).tolist()
                extra_metas[key] = value

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric_type, extra_metas=extra_metas))

    def __filt_override_unit_ordinate_coordinate(self, x_sets, y_sets):
        max_y_dict = {}
        for idx, x_value in enumerate(x_sets):
            if x_value not in max_y_dict:
                max_y_dict[x_value] = {"max_y": y_sets[idx], "idx": idx}
            else:
                max_y = max_y_dict[x_value]["max_y"]
                if max_y < y_sets[idx]:
                    max_y_dict[x_value] = {"max_y": y_sets[idx], "idx": idx}

        x = []
        y = []
        idx_list = []
        for key, value in max_y_dict.items():
            x.append(key)
            y.append(value["max_y"])
            idx_list.append(value["idx"])

        return x, y, idx_list

    def __save_roc(self, data_type, metric_name, metric_namespace, metric_res):
        fpr, tpr, thresholds, _ = metric_res

        # set roc edge value
        fpr.append(1.0)
        tpr.append(1.0)

        fpr, tpr, idx_list = self.__filt_override_unit_ordinate_coordinate(fpr, tpr)
        edge_idx = idx_list[-1]
        if edge_idx == len(thresholds):
            idx_list = idx_list[:-1]
        thresholds = [thresholds[idx] for idx in idx_list]

        self.__save_curve_data(fpr, tpr, metric_name, metric_namespace)
        self.__save_curve_meta(metric_name=metric_name, metric_namespace=metric_namespace,
                               metric_type="ROC", unit_name="fpr", ordinate_name="tpr",
                               curve_name=data_type, thresholds=thresholds)

    def callback_metric_data(self, return_single_val_metrics=False):

        """
        Parameters
        ----------
        return_single_val_metrics if True return single_val_metrics

        Returns None or return_result dict
        -------
        """

        collect_dict = {}
        LOGGER.debug('callback metric called')

        for (data_type, eval_res_list) in self.eval_results.items():

            if data_type in self.validate_key:
                collect_dict = self.validate_metric
            elif data_type in self.train_key:
                collect_dict = self.train_metric

            precision_recall = {}
            for eval_res in eval_res_list:
                for (metric, metric_res) in eval_res.items():
                    metric_namespace = metric_res[0]
                    metric_name = '_'.join([data_type, metric])

                    if metric in self.save_single_value_metric_list:
                        self.__save_single_value(metric_res[1], metric_name=data_type, metric_namespace=metric_namespace
                                                 ,eval_name=metric)
                        collect_dict[metric] = metric_res[1]

                    elif metric == consts.KS:
                        best_ks, fpr, tpr, thresholds, cuts = metric_res[1]
                        self.__save_single_value(best_ks, metric_name=data_type,
                                                 metric_namespace=metric_namespace,
                                                 eval_name=metric)
                        collect_dict[metric] = best_ks

                        metric_name_fpr = '_'.join([metric_name, "fpr"])
                        curve_name_fpr = "_".join([data_type, "fpr"])
                        self.__save_curve_data(cuts, fpr, metric_name_fpr, metric_namespace)
                        self.__save_curve_meta(metric_name=metric_name_fpr, metric_namespace=metric_namespace,
                                               metric_type=metric.upper(), unit_name="",
                                               curve_name=curve_name_fpr, pair_type=data_type, thresholds=thresholds)

                        metric_name_tpr = '_'.join([metric_name, "tpr"])
                        curve_name_tpr = "_".join([data_type, "tpr"])
                        self.__save_curve_data(cuts, tpr, metric_name_tpr, metric_namespace)
                        self.__save_curve_meta(metric_name_tpr, metric_namespace, metric.upper(), unit_name="",
                                               curve_name=curve_name_tpr, pair_type=data_type, thresholds=thresholds)

                    elif metric == consts.ROC:
                        self.__save_roc(data_type, metric_name, metric_namespace, metric_res[1])

                    elif metric in [consts.ACCURACY, consts.LIFT, consts.GAIN]:
                        if self.eval_type == consts.MULTY and metric == consts.ACCURACY:
                            self.__save_single_value(metric_res[1], metric_name=data_type,
                                                     metric_namespace=metric_namespace,
                                                     eval_name=metric)
                            collect_dict[metric] = metric_res[1]
                            continue

                        score, cuts, thresholds = metric_res[1]

                        if metric in [consts.LIFT, consts.GAIN]:
                            score = [float(s[1]) for s in score]
                            cuts = [float(c[1]) for c in cuts]
                            cuts, score, idx_list = self.__filt_override_unit_ordinate_coordinate(cuts, score)
                            thresholds = [thresholds[idx] for idx in idx_list]

                            score.append(1.0)
                            cuts.append(1.0)
                            thresholds.append(0.0)

                        self.__save_curve_data(cuts, score, metric_name, metric_namespace)
                        self.__save_curve_meta(metric_name=metric_name, metric_namespace=metric_namespace,
                                               metric_type=metric.upper(), unit_name="",
                                               curve_name=data_type, thresholds=thresholds)

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
                        metric_name_recall = '_'.join([data_type, "recall"])

                        pos_precision_score = precision_res[1][0]
                        precision_cuts = precision_res[1][1]
                        if len(precision_res[1]) >= 3:
                            precision_thresholds = precision_res[1][2]
                        else:
                            precision_thresholds = None

                        pos_recall_score = recall_res[1][0]
                        recall_cuts = recall_res[1][1]

                        if len(recall_res[1]) >= 3:
                            recall_thresholds = recall_res[1][2]
                        else:
                            recall_thresholds = None

                        precision_curve_name = data_type
                        recall_curve_name = data_type
                        if self.eval_type == consts.BINARY:
                            pos_precision_score = [score[1] for score in pos_precision_score]
                            pos_recall_score = [score[1] for score in pos_recall_score]

                            pos_recall_score, pos_precision_score, idx_list = self.__filt_override_unit_ordinate_coordinate(
                                pos_recall_score, pos_precision_score)

                            precision_cuts = [precision_cuts[idx] for idx in idx_list]
                            recall_cuts = [recall_cuts[idx] for idx in idx_list]

                            edge_idx = idx_list[-1]
                            if edge_idx == len(precision_thresholds) - 1:
                                idx_list = idx_list[:-1]
                            precision_thresholds = [precision_thresholds[idx] for idx in idx_list]
                            recall_thresholds = [recall_thresholds[idx] for idx in idx_list]

                        elif self.eval_type == consts.MULTY:
                            average_precision = float(np.array(pos_precision_score).mean())
                            average_recall = float(np.array(pos_recall_score).mean())
                            self.__save_single_value(average_precision, metric_name=data_type,
                                                     metric_namespace=metric_namespace,
                                                     eval_name="precision")
                            self.__save_single_value(average_recall, metric_name=data_type,
                                                     metric_namespace=metric_namespace,
                                                     eval_name="recall")
                            collect_dict[consts.PRECISION] = average_precision
                            collect_dict[consts.RECALL] = average_recall

                            precision_curve_name = metric_name_precision
                            recall_curve_name = metric_name_recall

                        self.__save_curve_data(precision_cuts, pos_precision_score, metric_name_precision,
                                               metric_namespace)
                        self.__save_curve_meta(metric_name_precision, metric_namespace,
                                               "_".join([consts.PRECISION.upper(), self.eval_type.upper()]),
                                               unit_name="", ordinate_name="Precision", curve_name=precision_curve_name,
                                               pair_type=data_type, thresholds=precision_thresholds)

                        self.__save_curve_data(recall_cuts, pos_recall_score, metric_name_recall,
                                               metric_namespace)
                        self.__save_curve_meta(metric_name_recall, metric_namespace,
                                               "_".join([consts.RECALL.upper(), self.eval_type.upper()]),
                                               unit_name="", ordinate_name="Recall", curve_name=recall_curve_name,
                                               pair_type=data_type, thresholds=recall_thresholds)
                    else:
                        LOGGER.warning("Unknown metric:{}".format(metric))

        if return_single_val_metrics:
            if len(self.validate_metric) !=0:
                LOGGER.debug("return validate metric")
                return self.validate_metric
            else:
                LOGGER.debug("validate metric is empty, return train metric")
                return self.train_metric

    def __filt_threshold(self, thresholds, step):
        cuts = list(map(float, np.arange(0, 1, step)))
        size = len(list(thresholds))
        thresholds.sort(reverse=True)
        index_list = [int(size * cut) for cut in cuts]
        new_thresholds = [thresholds[idx] for idx in index_list]

        return new_thresholds, cuts

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
            fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(pred_scores), drop_intermediate=1)
            fpr, tpr, thresholds = list(map(float, fpr)), list(map(float, tpr)), list(map(float, thresholds))

            filt_thresholds, cuts = self.__filt_threshold(thresholds=thresholds, step=0.01)
            new_thresholds = []
            new_tpr = []
            new_fpr = []
            for threshold in filt_thresholds:
                index = thresholds.index(threshold)
                new_tpr.append(tpr[index])
                new_fpr.append(fpr[index])
                new_thresholds.append(threshold)

            fpr = new_fpr
            tpr = new_tpr
            thresholds = new_thresholds
            return fpr, tpr, thresholds, cuts
        else:
            logging.warning("roc_curve is just suppose Binary Classification! return None as results")
            fpr, tpr, thresholds, cuts = None, None, None, None

            return fpr, tpr, thresholds, cuts

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
        score_label_list = []
        for i, label in enumerate(labels):
            score_label_list.append((pred_scores[i], label))

        score_label_list.sort(key=lambda x: x[0], reverse=True)
        cuts = [c / 100 for c in range(100)]
        data_size = len(pred_scores)
        indexs = [int(data_size * cut) for cut in cuts]
        score_threshold = [score_label_list[idx][0] for idx in indexs]

        fpr = []
        tpr = []
        ks = []
        for i, index in enumerate(indexs):
            positive = 0
            positive_recall = 0
            negative = 0
            false_positive = 0
            for score_label in score_label_list:
                pre_score = score_label[0]
                label = score_label[1]
                if label == self.pos_label:
                    positive += 1

                    if pre_score > score_threshold[i]:
                        positive_recall += 1

                if label == 0:
                    negative += 1
                    if pre_score > score_threshold[i]:
                        false_positive += 1

            if positive == 0 or negative == 0:
                raise ValueError("all labels are positive or negative, please check your data!")

            _tpr = positive_recall / positive
            _fpr = false_positive / negative
            _ks = _tpr - _fpr
            tpr.append(_tpr)
            fpr.append(_fpr)
            ks.append(_ks)

        fpr.append(1.0)
        tpr.append(1.0)
        cuts.append(1.0)

        return max(ks), fpr, tpr, score_threshold, cuts

    def lift(self, labels, pred_scores):
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
        if self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds, cuts = self.__filt_threshold(thresholds, 0.01)
            lift_operator = Lift()
            lift_y, lift_x = lift_operator.compute(labels, pred_scores, thresholds=thresholds)
            return lift_y, lift_x, thresholds
        else:
            logging.warning("lift is just suppose Binary Classification! return None as results")
            return None

    def gain(self, labels, pred_scores):
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

        if self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds, cuts = self.__filt_threshold(thresholds, 0.01)
            gain_operator = Gain()
            gain_x, gain_y = gain_operator.compute(labels, pred_scores, thresholds=thresholds)
            return gain_y, gain_x, thresholds
        else:
            logging.warning("gain is just suppose Binary Classification! return None as results")
            return None

    def precision(self, labels, pred_scores):
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
        if self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds, cuts = self.__filt_threshold(thresholds, 0.01)

            # set for recall edge value
            thresholds.append(min(thresholds) - 0.001)
            cuts.append(1)

            precision_operator = BiClassPrecision()
            precision_res, thresholds = precision_operator.compute(labels, pred_scores, thresholds)
            return precision_res, cuts, thresholds
        elif self.eval_type == consts.MULTY:
            precision_operator = MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

    def recall(self, labels, pred_scores):
        """
        Compute the recall
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        Returns
        ----------
        dict
            The key is threshold and the value is another dic, which key is label in parameter labels, and value is the label's recall.
        """
        if self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds, cuts = self.__filt_threshold(thresholds, 0.01)

            # set for recall edge value
            thresholds.append(min(thresholds) - 0.001)
            cuts.append(1)

            recall_operator = BiClassRecall()
            recall_res, thresholds = recall_operator.compute(labels, pred_scores, thresholds)
            return recall_res, cuts, thresholds
        elif self.eval_type == consts.MULTY:
            recall_operator = MultiClassRecall()
            return recall_operator.compute(labels, pred_scores)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

    def accuracy(self, labels, pred_scores, normalize=True):
        """
        Compute the accuracy
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each data.
        normalize: bool. If true, return the fraction of correctly classified samples, else returns the number of correctly classified samples
        Returns
        ----------
        dict
            the key is threshold and the value is the accuracy of this threshold.
        """

        if self.eval_type == consts.BINARY:
            thresholds = list(set(pred_scores))
            thresholds, cuts = self.__filt_threshold(thresholds, 0.01)
            acc_operator = BiClassAccuracy()
            acc_res, thresholds = acc_operator.compute(labels, pred_scores, thresholds, normalize)
            return acc_res, cuts, thresholds
        elif self.eval_type == consts.MULTY:
            acc_operator = MultiClassAccuracy()
            return acc_operator.compute(labels, pred_scores, normalize)
        else:
            logging.warning("error:can not find classification type:".format(self.eval_type))

    @staticmethod
    def extract_data(data: dict):
        return data


class Lift(object):
    """
    Compute lift
    """

    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value > threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def __compute_lift(self, labels, pred_scores_one_hot, pos_label="1"):
        tn, fp, fn, tp = confusion_matrix(labels, pred_scores_one_hot).ravel()

        if pos_label == '0':
            tp, tn = tn, tp
            fp, fn = fn, fp

        labels_num = len(labels)
        if labels_num == 0:
            lift_x = 1
            denominator = 1
        else:
            lift_x = (tp + fp) / labels_num
            denominator = (tp + fn) / labels_num

        if tp + fp == 0:
            numerator = 1
        else:
            numerator = tp / (tp + fp)

        if denominator == 0:
            lift_y = sys.float_info.max
        else:
            lift_y = numerator / denominator

        return lift_x, lift_y

    def compute(self, labels, pred_scores, thresholds=None):
        lifts_x = []
        lifts_y = []

        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            label_type = ['0', '1']
            lift_x_type = []
            lift_y_type = []
            for lt in label_type:
                lift_x, lift_y = self.__compute_lift(labels, pred_scores_one_hot, pos_label=lt)
                lift_x_type.append(lift_x)
                lift_y_type.append(lift_y)
            lifts_x.append(lift_x_type)
            lifts_y.append(lift_y_type)

        return lifts_y, lifts_x


class Gain(object):
    """
    Compute Gain
    """

    def __init__(self):
        pass

    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        for value in pred_value:
            if value > threshold:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def __compute_gain(self, label, pred_scores_one_hot, pos_label="1"):
        tn, fp, fn, tp = confusion_matrix(label, pred_scores_one_hot).ravel()

        if pos_label == '0':
            tp, tn = tn, tp
            fp, fn = fn, fp

        num_label = len(label)
        if num_label == 0:
            gain_x = 1
        else:
            gain_x = float((tp + fp) / num_label)

        num_positives = tp + fn
        if num_positives == 0:
            gain_y = 1
        else:
            gain_y = float(tp / num_positives)

        return gain_x, gain_y

    def compute(self, labels, pred_scores, thresholds=None):
        gains_x = []
        gains_y = []

        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            label_type = ['0', '1']
            gain_x_type = []
            gain_y_type = []
            for lt in label_type:
                gain_x, gain_y = self.__compute_gain(labels, pred_scores_one_hot, pos_label=lt)
                gain_x_type.append(gain_x)
                gain_y_type.append(gain_y)
            gains_x.append(gain_x_type)
            gains_y.append(gain_y_type)

        return gains_x, gains_y


class BiClassPrecision(object):
    """
    Compute binary classification precision
    """

    def __init__(self):
        self.total_positives = 0

    def __predict_value_to_one_hot(self, pred_value, threshold):
        one_hot = []
        self.total_positives = 0
        for value in pred_value:
            if value > threshold:
                one_hot.append(1)
                self.total_positives += 1
            else:
                one_hot.append(0)

        return one_hot

    def compute(self, labels, pred_scores, thresholds):
        scores = []
        for threshold in thresholds:
            pred_scores_one_hot = self.__predict_value_to_one_hot(pred_scores, threshold)
            score = list(map(float, precision_score(labels, pred_scores_one_hot, average=None)))
            if self.total_positives == 0:
                score[1] = 1.0
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
            if value > threshold:
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
            if value > threshold:
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


class IC(object):
    """
    Compute Information Criterion with a given dTable and loss
        When k = 2, result is genuine AIC;
        when k = log(n), results is BIC, also called SBC, SIC, SBIC.
    """

    def compute(self, k, n, dfe, loss):
        aic_score = k * dfe + 2 * n * loss
        return aic_score


class IC_Approx(object):
    """
    Compute Information Criterion value with a given dTable and loss
        When k = 2, result is genuine AIC;
        when k = log(n), results is BIC, also called SBC, SIC, SBIC.
        Note that this formula for linear regression dismisses the constant term n * np.log(2 * np.pi) for sake of simplicity, so the absolute value of result will be small.
    """

    def compute(self, k, n, dfe, loss):
        aic_score = k * dfe + n * np.log(loss * 2)
        return aic_score
