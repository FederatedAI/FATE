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
from federatedml.util import LOGGER
from fate_flow.entity.metric import Metric, MetricMeta

from federatedml.param import EvaluateParam
from federatedml.util import consts
from federatedml.model_base import ModelBase
from federatedml.evaluation.metric_interface import MetricInterface

LOGGER = log_utils.getLogger()


class PerformanceRecorder(object):

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

        self.no_improvement_round = {}  # record no improvement round of all metrics

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
                                              consts.ROOT_MEAN_SQUARED_ERROR,

                                              consts.JACCARD_SIMILARITY_SCORE,
                                              consts.ADJUSTED_RAND_SCORE,
                                              consts.FOWLKES_MALLOWS_SCORE,
                                              consts.DAVIES_BOULDIN_INDEX
                                              ]

        self.special_metric_list = [consts.PSI]

        self.metrics = None
        self.round_num = 6

        self.validate_metric = {}
        self.train_metric = {}

        # where to call metric computations
        self.metric_interface: MetricInterface = None

        self.psi_train_scores, self.psi_validate_scores = None, None
        self.psi_train_labels, self.psi_validate_labels = None, None

    def _init_model(self, model):
        self.model_param = model
        self.eval_type = self.model_param.eval_type
        self.pos_label = self.model_param.pos_label
        self.metrics = model.metrics
        self.metric_interface = MetricInterface(pos_label=self.pos_label, eval_type=self.eval_type,)

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

    def evaluate_metrics(self, mode: str, data: list) -> dict:
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

        metrics = self.metrics

        for eval_metric in metrics:

            if eval_metric not in self.special_metric_list:
                res = getattr(self.metric_interface, eval_metric)(labels, pred_results)
                if res is not None:
                    try:
                        if math.isinf(res):
                            res = float(-9999999)
                            LOGGER.info("res is inf, set to {}".format(res))
                    except:
                        pass

                    eval_result[eval_metric].append(mode)
                    eval_result[eval_metric].append(res)

            elif eval_metric == consts.PSI:
                if mode == 'train':
                    self.psi_train_scores = pred_results
                    self.psi_train_labels = labels
                elif mode == 'validate':
                    self.psi_validate_scores = pred_results
                    self.psi_validate_labels = labels

                if self.psi_train_scores is not None and self.psi_validate_scores is not None:
                    res = self.metric_interface.psi(self.psi_train_scores, self.psi_validate_scores, self.psi_train_labels, self.psi_validate_labels)
                    eval_result[eval_metric].append(mode)
                    eval_result[eval_metric].append(res)
                    # delete saved scores after computing a psi pair

                    self.psi_train_scores, self.psi_validate_scores = None, None

        return eval_result

    def fit(self, data, return_result=False):
        if len(data) <= 0:
            return

        self.eval_results.clear()
        for (key, eval_data) in data.items():
            eval_data_local = list(eval_data.collect())
            split_data_with_label = self.split_data_with_type(eval_data_local)
            for mode, data in split_data_with_label.items():
                eval_result = self.evaluate_metrics(mode, data)
                self.eval_results[key].append(eval_result)

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

    @staticmethod
    def __filt_override_unit_ordinate_coordinate(x_sets, y_sets):

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

    def __process_single_value_data(self, metric, metric_res):

        single_val_metric = None

        if metric in self.save_single_value_metric_list or \
           (metric is consts.ACCURACY and self.eval_type == consts.MULTY):

            single_val_metric = metric_res[1]

        elif metric == consts.KS:
            best_ks, fpr, tpr, thresholds, cuts = metric_res[1]
            single_val_metric = best_ks

        elif metric in [consts.RECALL, consts.PRECISION] and self.eval_type == consts.MULTY:
            pos_score = metric_res[1][0]
            single_val_metric = float(np.array(pos_score).mean())

        return single_val_metric

    @staticmethod
    def __filter_duplicate_roc_data_point(fpr, tpr, thresholds):

        data_point_set = set()
        new_fpr, new_tpr, new_threshold = [], [], []
        for fpr_, tpr_, thres in zip(fpr, tpr, thresholds):
            if (fpr_, tpr_, thres) not in data_point_set:
                data_point_set.add((fpr_, tpr_, thres))
                new_fpr.append(fpr_)
                new_tpr.append(tpr_)
                new_threshold.append(thres)

        return new_fpr, new_tpr, new_threshold

    def __save_roc_curve(self, data_name, metric_name, metric_namespace, metric_res):
        fpr, tpr, thresholds, _ = metric_res

        # set roc edge value
        fpr.append(1.0)
        tpr.append(1.0)

        fpr, tpr, thresholds = self.__filter_duplicate_roc_data_point(fpr, tpr, thresholds)

        self.__save_curve_data(fpr, tpr, metric_name, metric_namespace)
        self.__save_curve_meta(metric_name=metric_name, metric_namespace=metric_namespace,
                               metric_type="ROC", unit_name="fpr", ordinate_name="tpr",
                               curve_name=data_name, thresholds=thresholds)

    def __save_ks_curve(self, metric, metric_res, metric_name, metric_namespace, data_name):

        best_ks, fpr, tpr, thresholds, cuts = metric_res[1]

        for curve_name, curve_data in zip(["fpr", "tpr"], [fpr, tpr]):

            metric_name_fpr = '_'.join([metric_name, curve_name])
            curve_name_fpr = "_".join([data_name, curve_name])
            self.__save_curve_data(cuts, curve_data, metric_name_fpr, metric_namespace)
            self.__save_curve_meta(metric_name=metric_name_fpr, metric_namespace=metric_namespace,
                                   metric_type=metric.upper(), unit_name="",
                                   curve_name=curve_name_fpr, pair_type=data_name,
                                   thresholds=thresholds)

    def __save_lift_gain_curve(self, metric, metric_res, metric_name, metric_namespace, data_name):

        score, cuts, thresholds = metric_res[1]

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
                               curve_name=data_name, thresholds=thresholds)

    def __save_accuracy_curve(self, metric, metric_res, metric_name, metric_namespace, data_name):

        if self.eval_type == consts.MULTY:
            return

        score, cuts, thresholds = metric_res[1]

        self.__save_curve_data(cuts, score, metric_name, metric_namespace)
        self.__save_curve_meta(metric_name=metric_name, metric_namespace=metric_namespace,
                               metric_type=metric.upper(), unit_name="",
                               curve_name=data_name, thresholds=thresholds)

    def __save_pr_curve(self, precision_and_recall, data_name):

        precision_res = precision_and_recall[consts.PRECISION]
        recall_res = precision_and_recall[consts.RECALL]

        if precision_res[0] != recall_res[0]:
            LOGGER.warning(
                "precision mode:{} is not equal to recall mode:{}".format(precision_res[0],
                                                                          recall_res[0]))
            return

        metric_namespace = precision_res[0]
        metric_name_precision = '_'.join([data_name, "precision"])
        metric_name_recall = '_'.join([data_name, "recall"])

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

        precision_curve_name = data_name
        recall_curve_name = data_name

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

        self.__save_curve_data(precision_cuts, pos_precision_score, metric_name_precision,
                               metric_namespace)
        self.__save_curve_meta(metric_name_precision, metric_namespace,
                               "_".join([consts.PRECISION.upper(), self.eval_type.upper()]),
                               unit_name="", ordinate_name="Precision",
                               curve_name=precision_curve_name,
                               pair_type=data_name, thresholds=precision_thresholds)

        self.__save_curve_data(recall_cuts, pos_recall_score, metric_name_recall,
                               metric_namespace)
        self.__save_curve_meta(metric_name_recall, metric_namespace,
                               "_".join([consts.RECALL.upper(), self.eval_type.upper()]),
                               unit_name="", ordinate_name="Recall", curve_name=recall_curve_name,
                               pair_type=data_name, thresholds=recall_thresholds)

    def __save_confusion_mat_table(self, metric, confusion_mat, thresholds, metric_name, metric_namespace):

        extra_metas = {'tp': list(confusion_mat['tp']), 'tn': list(confusion_mat['tn']), 'fp': list(confusion_mat['fp']),
                       'fn': list(confusion_mat['fn']), 'thresholds': list(np.round(thresholds, self.round_num))}

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __save_f1_score_table(self, metric, f1_scores, thresholds, metric_name, metric_namespace):

        extra_metas = {'f1_scores': list(np.round(f1_scores, self.round_num)),
                       'thresholds': list(np.round(thresholds, self.round_num))}

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __save_psi_table(self, metric, metric_res, metric_name, metric_namespace):

        psi_scores, total_psi, expected_interval, expected_percentage, actual_interval, actual_percentage, \
        train_pos_perc, validate_pos_perc, intervals = metric_res[1]

        extra_metas = {'psi_scores': list(np.round(psi_scores, self.round_num)), 'total_psi': round(total_psi, self.round_num),
                        'expected_interval': list(expected_interval),
                       'expected_percentage': list(expected_percentage), 'actual_interval': list(actual_interval),
                       'actual_percentage': list(actual_percentage), 'intervals': list(intervals),
                       'train_pos_perc': train_pos_perc, 'validate_pos_perc': validate_pos_perc
                       }

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __save_pr_table(self, metric, metric_res, metric_name, metric_namespace):

        p_scores, r_scores, score_threshold = metric_res[1]

        extra_metas = {'p_scores': list(map(list, np.round(p_scores, self.round_num))),
                       'r_scores': list(map(list, np.round(r_scores, self.round_num))),
                       'thresholds': list(np.round(score_threshold, self.round_num))}

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def callback_metric_data(self, return_single_val_metrics=False):

        collect_dict = {}
        LOGGER.debug('callback metric called')

        for (data_type, eval_res_list) in self.eval_results.items():

            precision_recall = {}

            for eval_res in eval_res_list:
                for (metric, metric_res) in eval_res.items():

                    metric_namespace = metric_res[0]

                    if metric_namespace == 'validate':
                        collect_dict = self.validate_metric
                    elif metric_namespace == 'train':
                        collect_dict = self.train_metric

                    metric_name = '_'.join([data_type, metric])

                    single_val_metric = self.__process_single_value_data(metric, metric_res)

                    if single_val_metric is not None:
                        self.__save_single_value(single_val_metric, metric_name=data_type,
                                                 metric_namespace=metric_namespace
                                                 , eval_name=metric)
                        collect_dict[metric] = single_val_metric

                    if metric == consts.KS:
                        self.__save_ks_curve(metric, metric_res, metric_name, metric_namespace, data_type)

                    elif metric == consts.ROC:
                        self.__save_roc_curve(data_type, metric_name, metric_namespace, metric_res[1])

                    elif metric == consts.ACCURACY:
                        self.__save_accuracy_curve(metric, metric_res, metric_name, metric_namespace, data_type)

                    elif metric in [consts.GAIN, consts.LIFT]:
                        self.__save_lift_gain_curve(metric, metric_res, metric_name, metric_namespace, data_type)

                    elif metric in [consts.PRECISION, consts.RECALL]:
                        precision_recall[metric] = metric_res
                        if len(precision_recall) < 2:
                            continue

                        self.__save_pr_curve(precision_recall, data_type)

                        precision_recall = {}  # reset cached dict

                    elif metric == consts.PSI:
                        self.__save_psi_table(metric, metric_res, metric_name, metric_namespace)

                    elif metric == consts.CONFUSION_MAT:
                        confusion_mat, cuts, score_threshold= metric_res[1]
                        self.__save_confusion_mat_table(metric, confusion_mat, score_threshold, metric_name,
                                                        metric_namespace)

                    elif metric == consts.F1_SCORE:
                        f1_scores, cuts, score_threshold = metric_res[1]
                        self.__save_f1_score_table(metric, f1_scores, score_threshold, metric_name, metric_namespace)

                    elif metric == consts.QUANTILE_PR:
                        LOGGER.debug('pr quantile called')
                        self.__save_pr_table(metric, metric_res, metric_name, metric_namespace)

        if return_single_val_metrics:
            if len(self.validate_metric) != 0:
                LOGGER.debug("return validate metric")
                LOGGER.debug('validate metric is {}'.format(self.validate_metric))
                return self.validate_metric
            else:
                LOGGER.debug("validate metric is empty, return train metric")
                LOGGER.debug('train metric is {}'.format(self.train_metric))
                return self.train_metric

        else:
            return None

    @staticmethod
    def extract_data(data: dict):
        result = {}
        for k, v in data.items():
            result[".".join(k.split(".")[:-1])] = v
        return result
