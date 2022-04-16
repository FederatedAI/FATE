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
import math
from federatedml.util import LOGGER
from federatedml.model_base import Metric, MetricMeta
from federatedml.param import EvaluateParam
from federatedml.util import consts
from federatedml.model_base import ModelBase
from federatedml.evaluation.metric_interface import MetricInterface

import numpy as np


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

        self.clustering_intra_metric_list = [consts.DAVIES_BOULDIN_INDEX, consts.DISTANCE_MEASURE]

        self.metrics = None
        self.round_num = 6

        self.eval_type = None

        # where to call metric computations
        self.metric_interface: MetricInterface = None

        self.psi_train_scores, self.psi_validate_scores = None, None
        self.psi_train_labels, self.psi_validate_labels = None, None

        # multi unfold setting
        self.need_unfold_multi_result = False

        # summaries
        self.metric_summaries = {}

    def _init_model(self, model):
        self.model_param = model
        self.eval_type = self.model_param.eval_type
        self.pos_label = self.model_param.pos_label
        self.need_unfold_multi_result = self.model_param.unfold_multi_result
        self.metrics = model.metrics
        self.metric_interface = MetricInterface(pos_label=self.pos_label, eval_type=self.eval_type, )

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
            mode = value[1][-1]
            split_result[mode].append(value)

        return split_result

    def _classification_and_regression_extract(self, data):
        """
        extract labels and predict results from data in classification/regression type format
        """

        labels = []
        pred_scores = []
        pred_labels = []
        for d in data:
            labels.append(d[1][0])
            pred_labels.append(d[1][1])
            pred_scores.append(d[1][2])
        if self.eval_type == consts.BINARY or self.eval_type == consts.REGRESSION:
            if self.pos_label and self.eval_type == consts.BINARY:
                labels_arr = np.array(labels)
                labels_arr[labels_arr == self.pos_label] = 1
                labels_arr[labels_arr != self.pos_label] = 0
                labels = list(labels_arr)
            pred_results = pred_scores
        else:
            pred_results = pred_labels

        return labels, pred_results

    def _clustering_extract(self, data):
        """
        extract data according to data format
        """

        true_cluster_index, predicted_cluster_index = [], []
        intra_cluster_data, inter_cluster_dist = {'avg_dist': [], 'max_radius': []}, []

        run_intra_metrics = False  # run intra metrics or outer metrics ?
        if len(data[0][1]) == 3:
            # [int int] -> [true_label, predicted label] -> outer metric
            # [int np.array] - > [predicted label, distance] -> need no metric computation
            if not (isinstance(data[0][1][0], int) and isinstance(data[0][1][1], int)):
                return None, None, run_intra_metrics

        if len(data[0][1]) == 5:  # the input format is for intra metrics
            run_intra_metrics = True

        cluster_index_list = []
        for d in data:
            if run_intra_metrics:
                cluster_index_list.append(d[0])
                intra_cluster_data['avg_dist'].append(d[1][1])
                intra_cluster_data['max_radius'].append(d[1][2])
                if len(inter_cluster_dist) == 0:
                    inter_cluster_dist += d[1][3]
            else:
                true_cluster_index.append(d[1][0])
                predicted_cluster_index.append(d[1][1])

        # if cluster related data exists, sort by cluster index
        if len(cluster_index_list) != 0:
            to_sort = list(zip(cluster_index_list, intra_cluster_data['avg_dist'], intra_cluster_data['max_radius']))
            sort_rs = sorted(to_sort, key=lambda x: x[0])  # cluster index
            intra_cluster_data['avg_dist'] = [i[1] for i in sort_rs]
            intra_cluster_data['max_radius'] = [i[2] for i in sort_rs]

        return (true_cluster_index, predicted_cluster_index, run_intra_metrics) if not run_intra_metrics else \
            (intra_cluster_data, inter_cluster_dist, run_intra_metrics)

    def _evaluate_classification_and_regression_metrics(self, mode, data):

        labels, pred_results = self._classification_and_regression_extract(data)
        eval_result = defaultdict(list)
        for eval_metric in self.metrics:
            if eval_metric not in self.special_metric_list:
                res = getattr(self.metric_interface, eval_metric)(labels, pred_results)
                if res is not None:
                    try:
                        if math.isinf(res):
                            res = float(-9999999)
                            LOGGER.info("res is inf, set to {}".format(res))
                    except BaseException:
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
                    res = self.metric_interface.psi(self.psi_train_scores, self.psi_validate_scores,
                                                    self.psi_train_labels, self.psi_validate_labels)
                    eval_result[eval_metric].append(mode)
                    eval_result[eval_metric].append(res)
                    # delete saved scores after computing a psi pair
                    self.psi_train_scores, self.psi_validate_scores = None, None

        return eval_result

    def _evaluate_clustering_metrics(self, mode, data):

        eval_result = defaultdict(list)
        rs0, rs1, run_outer_metric = self._clustering_extract(data)
        if rs0 is None and rs1 is None:  # skip evaluation computation if get this input format
            LOGGER.debug('skip computing, this clustering format is not for metric computation')
            return eval_result

        if not run_outer_metric:
            no_label = set(rs0) == {None}
            if no_label:
                LOGGER.debug('no label found in clustering result, skip metric computation')
                return eval_result

        for eval_metric in self.metrics:

            # if input format and required metrics matches ? XNOR
            if not ((not (eval_metric in self.clustering_intra_metric_list) and not run_outer_metric) +
                    ((eval_metric in self.clustering_intra_metric_list) and run_outer_metric)):
                LOGGER.warning('input data format does not match current clustering metric: {}'.format(eval_metric))
                continue

            LOGGER.debug('clustering_metrics is {}'.format(eval_metric))

            if run_outer_metric:

                if eval_metric == consts.DISTANCE_MEASURE:
                    res = getattr(self.metric_interface, eval_metric)(rs0['avg_dist'], rs1, rs0['max_radius'])
                else:
                    res = getattr(self.metric_interface, eval_metric)(rs0['avg_dist'], rs1)
            else:
                res = getattr(self.metric_interface, eval_metric)(rs0, rs1)
            eval_result[eval_metric].append(mode)
            eval_result[eval_metric].append(res)

        return eval_result

    @staticmethod
    def _check_clustering_input(data):
        # one evaluation component is only available for one kmeans component in current version
        input_num = len(data.items())
        if input_num > 1:
            raise ValueError('multiple input detected, '
                             'one evaluation component is only available '
                             'for one clustering(kmean) component in current version')

    @staticmethod
    def _unfold_multi_result(score_list):
        """
        one-vs-rest transformation: multi classification result to several binary classification results
        """

        binary_result = {}
        for key, multi_result in score_list:
            true_label = multi_result[0]
            predicted_label = multi_result[1]
            multi_score = multi_result[3]
            data_type = multi_result[-1]
            # to binary predict result format
            for multi_label in multi_score:
                bin_label = 1 if str(multi_label) == str(true_label) else 0
                bin_predicted_label = 1 if str(multi_label) == str(predicted_label) else 0
                bin_score = multi_score[multi_label]
                neg_bin_score = 1 - bin_score
                result_list = [bin_label, bin_predicted_label, bin_score, {1: bin_score, 0: neg_bin_score}, data_type]
                if multi_label not in binary_result:
                    binary_result[multi_label] = []
                binary_result[multi_label].append((key, result_list))

        return binary_result

    def evaluate_metrics(self, mode: str, data: list) -> dict:

        eval_result = None
        if self.eval_type != consts.CLUSTERING:
            eval_result = self._evaluate_classification_and_regression_metrics(mode, data)
        elif self.eval_type == consts.CLUSTERING:
            LOGGER.debug('running clustering')
            eval_result = self._evaluate_clustering_metrics(mode, data)

        return eval_result

    def obtain_data(self, data_list):
        return data_list

    def check_data(self, data):

        if len(data) <= 0:
            return

        if self.eval_type == consts.CLUSTERING:
            self._check_clustering_input(data)
        else:
            for key, eval_data in data.items():
                if eval_data is None:
                    continue
                sample = eval_data.take(1)[0]
                # label, predict_type, predict_score, predict_detail, type
                if not isinstance(sample[1].features, list) or len(sample[1].features) != 5:
                    raise ValueError('length of table header mismatch, expected length is 5, got:{},'
                                     'please check the input of the Evaluation Module, result of '
                                     'cross validation is not supported.'.format(sample))

    def fit(self, data, return_result=False):

        self.check_data(data)

        LOGGER.debug(f'running eval, data: {data}')
        self.eval_results.clear()
        for (key, eval_data) in data.items():

            if eval_data is None:
                LOGGER.debug('data with {} is None, skip metric computation'.format(key))
                continue

            collected_data = list(eval_data.collect())
            if len(collected_data) == 0:
                continue

            eval_data_local = []
            for k, v in collected_data:
                eval_data_local.append((k, v.features))

            split_data_with_label = self.split_data_with_type(eval_data_local)

            for mode, data in split_data_with_label.items():
                eval_result = self.evaluate_metrics(mode, data)
                self.eval_results[key].append(eval_result)

            if self.need_unfold_multi_result and self.eval_type == consts.MULTY:
                unfold_binary_eval_result = defaultdict(list)

                # set work mode to binary evaluation
                self.eval_type = consts.BINARY
                self.metric_interface.eval_type = consts.ONE_VS_REST
                back_up_metric = self.metrics
                self.metrics = [consts.AUC, consts.KS]

                for mode, data in split_data_with_label.items():
                    unfold_multi_data = self._unfold_multi_result(eval_data_local)
                    for multi_label, marginal_bin_result in unfold_multi_data.items():
                        eval_result = self.evaluate_metrics(mode, marginal_bin_result)
                        new_key = key + '_class_{}'.format(multi_label)
                        unfold_binary_eval_result[new_key].append(eval_result)

                self.callback_ovr_metric_data(unfold_binary_eval_result)

                # recover work mode
                self.eval_type = consts.MULTY
                self.metric_interface.eval_type = consts.MULTY
                self.metrics = back_up_metric

        return self.callback_metric_data(self.eval_results, return_single_val_metrics=return_result)

    def __save_single_value(self, result, metric_name, metric_namespace, eval_name):

        metric_type = 'EVALUATION_SUMMARY'
        if eval_name in consts.ALL_CLUSTER_METRICS:
            metric_type = 'CLUSTERING_EVALUATION_SUMMARY'

        self.tracker.log_metric_data(metric_namespace, metric_name,
                                     [Metric(eval_name, np.round(result, self.round_num))])
        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric_type))

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
    def __multi_class_label_padding(metrics, label_indices):

        # in case some labels don't appear when running homo-multi-class algo
        label_num = np.max(label_indices) + 1
        index_result_mapping = dict(zip(label_indices, metrics))
        new_metrics, new_label_indices = [], []
        for i in range(label_num):
            if i in index_result_mapping:
                new_metrics.append(index_result_mapping[i])
            else:
                new_metrics.append(0.0)
            new_label_indices.append(i)

        return new_metrics, new_label_indices

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
                (metric == consts.ACCURACY and self.eval_type == consts.MULTY):

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

        fpr, tpr, thresholds = self.__filter_duplicate_roc_data_point(fpr, tpr, thresholds)

        # set roc edge value
        fpr.append(1.0)
        tpr.append(1.0)
        thresholds.append(1.0)

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

        elif self.eval_type == consts.MULTY:

            pos_recall_score, recall_cuts = self.__multi_class_label_padding(pos_recall_score, recall_cuts)
            pos_precision_score, precision_cuts = self.__multi_class_label_padding(pos_precision_score, precision_cuts)

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

        extra_metas = {'tp': list(confusion_mat['tp']), 'tn': list(confusion_mat['tn']),
                       'fp': list(confusion_mat['fp']),
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

        extra_metas = {'psi_scores': list(np.round(psi_scores, self.round_num)),
                       'total_psi': round(total_psi, self.round_num),
                       'expected_interval': list(expected_interval),
                       'expected_percentage': list(expected_percentage), 'actual_interval': list(actual_interval),
                       'actual_percentage': list(actual_percentage), 'intervals': list(intervals),
                       'train_pos_perc': train_pos_perc, 'validate_pos_perc': validate_pos_perc
                       }

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __save_pr_table(self, metric, metric_res, metric_name, metric_namespace):

        p_scores, r_scores, score_threshold = metric_res

        extra_metas = {'p_scores': list(map(list, np.round(p_scores, self.round_num))),
                       'r_scores': list(map(list, np.round(r_scores, self.round_num))),
                       'thresholds': list(np.round(score_threshold, self.round_num))}

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __save_contingency_matrix(self, metric, metric_res, metric_name, metric_namespace):

        result_array, unique_predicted_label, unique_true_label = metric_res
        true_labels = list(map(int, unique_true_label))
        predicted_label = list(map(int, unique_predicted_label))
        result_table = []
        for l_ in result_array:
            result_table.append(list(map(int, l_)))

        extra_metas = {'true_labels': true_labels, 'predicted_labels': predicted_label, 'result_table': result_table}

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __save_distance_measure(self, metric, metric_res: dict, metric_name, metric_namespace):

        extra_metas = {}
        cluster_index = [k for k in metric_res.keys()]
        radius, neareast_idx = [], []
        for k in metric_res:
            radius.append(metric_res[k][0])
            neareast_idx.append(metric_res[k][1])

        extra_metas['cluster_index'] = cluster_index
        extra_metas['radius'] = radius
        extra_metas['nearest_idx'] = neareast_idx

        self.tracker.set_metric_meta(metric_namespace, metric_name,
                                     MetricMeta(name=metric_name, metric_type=metric.upper(), extra_metas=extra_metas))

    def __update_summary(self, data_type, namespace, metric, metric_val):
        if data_type not in self.metric_summaries:
            self.metric_summaries[data_type] = {}
        if namespace not in self.metric_summaries[data_type]:
            self.metric_summaries[data_type][namespace] = {}
        self.metric_summaries[data_type][namespace][metric] = metric_val

    def __save_summary(self):
        LOGGER.info('eval summary is {}'.format(self.metric_summaries))
        self.set_summary(self.metric_summaries)

    def callback_ovr_metric_data(self, eval_results):

        for model_name, eval_rs in eval_results.items():

            train_callback_meta = defaultdict(dict)
            validate_callback_meta = defaultdict(dict)
            split_list = model_name.split('_')
            label = split_list[-1]
            origin_model_name_list = split_list[:-2]  # remove ' "class" label_index'
            origin_model_name = ''
            for s in origin_model_name_list:
                origin_model_name += (s + '_')
            origin_model_name = origin_model_name[:-1]

            for rs_dict in eval_rs:
                for metric_name, metric_rs in rs_dict.items():
                    if metric_name == consts.KS:
                        metric_rs = [metric_rs[0], metric_rs[1][0]]  # ks value only, curve data is not needed
                    metric_namespace = metric_rs[0]
                    if metric_namespace == 'train':
                        callback_meta = train_callback_meta
                    else:
                        callback_meta = validate_callback_meta
                    callback_meta[label][metric_name] = metric_rs[1]

            self.tracker.set_metric_meta("train", model_name + '_' + 'ovr',
                                         MetricMeta(name=origin_model_name, metric_type='ovr',
                                                    extra_metas=train_callback_meta))
            self.tracker.set_metric_meta("validate", model_name + '_' + 'ovr',
                                         MetricMeta(name=origin_model_name, metric_type='ovr',
                                                    extra_metas=validate_callback_meta))

            LOGGER.debug('callback data {} {}'.format(train_callback_meta, validate_callback_meta))

    def callback_metric_data(self, eval_results, return_single_val_metrics=False):

        # collect single val metric for validation strategy
        validate_metric = {}
        train_metric = {}
        collect_dict = {}
        LOGGER.debug('callback metric called')

        for (data_type, eval_res_list) in eval_results.items():

            precision_recall = {}
            for eval_res in eval_res_list:
                for (metric, metric_res) in eval_res.items():

                    metric_namespace = metric_res[0]

                    if metric_namespace == 'validate':
                        collect_dict = validate_metric
                    elif metric_namespace == 'train':
                        collect_dict = train_metric

                    metric_name = '_'.join([data_type, metric])

                    single_val_metric = self.__process_single_value_data(metric, metric_res)

                    if single_val_metric is not None:
                        self.__save_single_value(single_val_metric, metric_name=data_type,
                                                 metric_namespace=metric_namespace, eval_name=metric)
                        collect_dict[metric] = single_val_metric
                        # update pipeline summary
                        self.__update_summary(data_type, metric_namespace, metric, single_val_metric)

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
                        confusion_mat, cuts, score_threshold = metric_res[1]
                        self.__save_confusion_mat_table(metric, confusion_mat, score_threshold, metric_name,
                                                        metric_namespace)

                    elif metric == consts.F1_SCORE:
                        f1_scores, cuts, score_threshold = metric_res[1]
                        self.__save_f1_score_table(metric, f1_scores, score_threshold, metric_name, metric_namespace)

                    elif metric == consts.QUANTILE_PR:
                        self.__save_pr_table(metric, metric_res[1], metric_name, metric_namespace)

                    elif metric == consts.CONTINGENCY_MATRIX:
                        self.__save_contingency_matrix(metric, metric_res[1], metric_name, metric_namespace)

                    elif metric == consts.DISTANCE_MEASURE:
                        self.__save_distance_measure(metric, metric_res[1], metric_name, metric_namespace)

        self.__save_summary()

        if return_single_val_metrics:
            if len(validate_metric) != 0:
                LOGGER.debug("return validate metric")
                LOGGER.debug('validate metric is {}'.format(validate_metric))
                return validate_metric
            else:
                LOGGER.debug("validate metric is empty, return train metric")
                LOGGER.debug('train metric is {}'.format(train_metric))
                return train_metric

        else:
            return None

    @staticmethod
    def extract_data(data: dict):
        result = {}
        for k, v in data.items():
            result[".".join(k.split(".")[:1])] = v
        return result
