import sys
import copy
import pandas as pd
from typing import Dict
import numpy as np
import torch
from fate.ml.evaluation.metric_base import Metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from fate.ml.evaluation.metric_base import EvalResult


"""
Single Value Metrics
"""


class AUC(Metric):

    metric_name = "auc"

    def __init__(self):
        super().__init__()

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        auc_score = roc_auc_score(label, predict)
        return EvalResult(self.metric_name, auc_score)


class BinaryMetricWithThreshold(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold


class MultiAccuracy(Metric):

    metric_name = "multi_accuracy"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict, flatten=False)
        label = self.to_np_format(label).astype(np.int32)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1).astype(np.int32)
        acc = accuracy_score(label, predict)
        return EvalResult(self.metric_name, acc)


class MultiRecall(Metric):

    metric_name = "multi_recall"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict, flatten=False)
        label = self.to_np_format(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        recall = recall_score(label, predict, average="macro")
        return EvalResult(self.metric_name, recall)


class MultiPrecision(Metric):

    metric_name = "multi_precision"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict, flatten=False)
        label = self.to_np_format(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        precision = precision_score(label, predict, average="macro")
        return EvalResult(self.metric_name, precision)


class BinaryAccuracy(MultiAccuracy, BinaryMetricWithThreshold):

    metric_name = "binary_accuracy"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_np_format(label)
        acc = accuracy_score(label, predict)
        return EvalResult(self.metric_name, acc)


class BinaryRecall(MultiRecall, BinaryMetricWithThreshold):

    metric_name = "binary_recall"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_np_format(label)
        recall = recall_score(label, predict)
        return EvalResult(self.metric_name, recall)


class BinaryPrecision(MultiPrecision, BinaryMetricWithThreshold):

    metric_name = "binary_precision"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_np_format(label)
        precision = precision_score(label, predict)
        return EvalResult(self.metric_name, precision)


class MultiF1Score(Metric):

    metric_name = "multi_f1_score"

    def __init__(self, average="micro"):
        super().__init__()
        self.average = average

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict, flatten=False)
        label = self.to_np_format(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        f1 = f1_score(label, predict, average=self.average)
        return EvalResult(self.metric_name, f1)


class BinaryF1Score(MultiF1Score, BinaryMetricWithThreshold):

    metric_name = "binary_f1_score"

    def __init__(self, threshold=0.5, average="binary"):
        super().__init__(average)
        self.threshold = threshold

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_np_format(label)
        f1 = f1_score(label, predict, average=self.average)
        return EvalResult(self.metric_name, f1)


"""
Functions for Metrics with Cruve/Table Results
"""

ROUND_NUM = 6


def neg_pos_count(labels: np.ndarray, pos_label: int):
    pos_num = ((labels == pos_label) + 0).sum()
    neg_num = len(labels) - pos_num
    return pos_num, neg_num


def sort_score_and_label(labels: np.ndarray, pred_scores: np.ndarray):
    labels = np.array(labels)
    pred_scores = np.array(pred_scores)

    sort_idx = np.flip(pred_scores.argsort())
    sorted_labels = labels[sort_idx]
    sorted_scores = pred_scores[sort_idx]

    return sorted_labels, sorted_scores


class _ConfusionMatrix(object):
    @staticmethod
    def compute(sorted_labels: list, sorted_pred_scores: list, score_thresholds: list, ret: list, pos_label=1):

        for ret_type in ret:
            assert ret_type in ["tp", "tn", "fp", "fn"]

        sorted_labels = np.array(sorted_labels)
        sorted_scores = np.array(sorted_pred_scores)
        sorted_labels[sorted_labels != pos_label] = 0
        sorted_labels[sorted_labels == pos_label] = 1
        score_thresholds = np.array([score_thresholds]).transpose()
        pred_labels = (sorted_scores > score_thresholds) + 0

        ret_dict = {}
        if "tp" in ret or "tn" in ret:
            match_arr = pred_labels + sorted_labels
            if "tp" in ret:
                tp_num = (match_arr == 2).sum(axis=-1)
                ret_dict["tp"] = tp_num
            if "tn" in ret:
                tn_num = (match_arr == 0).sum(axis=-1)
                ret_dict["tn"] = tn_num

        if "fp" in ret or "fn" in ret:
            match_arr = sorted_labels - pred_labels
            if "fp" in ret:
                fp_num = (match_arr == -1).sum(axis=-1)
                ret_dict["fp"] = fp_num
            if "fn" in ret:
                fn_num = (match_arr == 1).sum(axis=-1)
                ret_dict["fn"] = fn_num

        return ret_dict


class ThresholdCutter(object):
    @staticmethod
    def cut_by_step(sorted_scores, steps=0.01):
        assert isinstance(steps, float) and (0 < steps < 1)
        thresholds = list(set(sorted_scores))
        thresholds, cuts = ThresholdCutter.__filt_threshold(thresholds, 0.01)
        score_threshold = thresholds

        return score_threshold, cuts

    @staticmethod
    def fixed_interval_threshold(steps=0.01):
        intervals = np.array([i for i in range(0, 100)])
        intervals = intervals * steps
        return intervals

    @staticmethod
    def cut_by_index(sorted_scores):
        cuts = np.array([c / 100 for c in range(100)])
        data_size = len(sorted_scores)
        indexs = [int(data_size * cut) for cut in cuts]
        score_threshold = [sorted_scores[idx] for idx in indexs]
        return score_threshold, cuts

    @staticmethod
    def __filt_threshold(thresholds, step):
        cuts = list(map(float, np.arange(0, 1, step)))
        size = len(list(thresholds))
        thresholds.sort(reverse=True)
        index_list = [int(size * cut) for cut in cuts]
        new_thresholds = [thresholds[idx] for idx in index_list]

        return new_thresholds, cuts

    @staticmethod
    def cut_by_quantile(scores, quantile_list=None, interpolation="nearest", remove_duplicate=True):

        if quantile_list is None:  # default is 20 intervals
            quantile_list = [round(i * 0.05, 3) for i in range(20)] + [1.0]
        quantile_val = np.quantile(scores, quantile_list, interpolation=interpolation)
        if remove_duplicate:
            quantile_val = sorted(list(set(quantile_val)))
        else:
            quantile_val = sorted(list(quantile_val))

        if len(quantile_val) == 1:
            quantile_val = [np.min(scores), np.max(scores)]

        return quantile_val


class BiClassMetric(object):
    def __init__(self, cut_method="step", remove_duplicate=False, pos_label=1):
        assert cut_method in ["step", "quantile"]
        self.cut_method = cut_method
        self.remove_duplicate = remove_duplicate  # available when cut_method is quantile
        self.pos_label = pos_label

    def prepare_confusion_mat(
        self,
        labels,
        scores,
        add_to_end=True,
    ):
        sorted_labels, sorted_scores = sort_score_and_label(labels, scores)

        score_threshold, cuts = None, None

        if self.cut_method == "step":
            score_threshold, cuts = ThresholdCutter.cut_by_step(sorted_scores, steps=0.01)
            if add_to_end:
                score_threshold.append(min(score_threshold) - 0.001)
                cuts.append(1)

        elif self.cut_method == "quantile":
            score_threshold = ThresholdCutter.cut_by_quantile(sorted_scores, remove_duplicate=self.remove_duplicate)
            score_threshold = list(np.flip(score_threshold))

        confusion_mat = _ConfusionMatrix.compute(
            sorted_labels, sorted_scores, score_threshold, ret=["tp", "fp", "fn", "tn"], pos_label=self.pos_label
        )

        return confusion_mat, score_threshold, cuts

    def compute(
        self,
        labels,
        scores,
    ):
        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(
            labels,
            scores,
        )
        metric_scores = self.compute_metric_from_confusion_mat(confusion_mat)
        return list(metric_scores), score_threshold, cuts

    def compute_metric_from_confusion_mat(self, *args):
        raise NotImplementedError()


"""
Metrics with Cruve/Table Results
"""


class KS(Metric):

    metric_name = "ks"

    def __init__(self):
        super().__init__()

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)

        sorted_labels, sorted_scores = sort_score_and_label(label, predict)
        threshold, cuts = ThresholdCutter.cut_by_index(sorted_scores)
        confusion_mat = _ConfusionMatrix.compute(
            sorted_labels, sorted_scores, threshold, ret=["tp", "fp"], pos_label=1
        )
        pos_num, neg_num = neg_pos_count(sorted_labels, pos_label=1)

        assert pos_num > 0 and neg_num > 0, (
            "error when computing KS metric, pos sample number and neg sample number" "must be larger than 0"
        )

        tpr_arr = confusion_mat["tp"] / pos_num
        fpr_arr = confusion_mat["fp"] / neg_num

        tpr = np.append(tpr_arr, np.array([1.0]))
        fpr = np.append(fpr_arr, np.array([1.0]))
        cuts = np.append(cuts, np.array([1.0]))
        threshold.append(0.0)
        ks_curve = tpr[:-1] - fpr[:-1]
        ks_val = np.max(ks_curve)

        return EvalResult(self.metric_name, ks_val), EvalResult(
            self.metric_name + "_table", pd.DataFrame({"tpr": tpr, "fpr": fpr, "threshold": threshold, "cuts": cuts})
        )


class ConfusionMatrix(Metric):

    metric_name = "confusion_matrix"

    def __init__(self):
        super().__init__()

    def __call__(self, predict, label, **kwargs):

        predict = self.to_np_format(predict)
        label = self.to_np_format(label)

        sorted_labels, sorted_scores = sort_score_and_label(label, predict)
        threshold, cuts = ThresholdCutter.cut_by_index(sorted_scores)
        confusion_mat = _ConfusionMatrix.compute(
            sorted_labels, sorted_scores, threshold, ret=["tp", "tn", "fp", "fn"], pos_label=1
        )
        confusion_mat["cuts"] = cuts
        confusion_mat["threshold"] = threshold
        return EvalResult(self.metric_name, pd.DataFrame(confusion_mat))


class Lift(Metric, BiClassMetric):

    metric_name = "lift"

    def __init__(self, *args, **kwargs):
        Metric.__init__(self)
        BiClassMetric.__init__(self, cut_method="step", remove_duplicate=False, pos_label=1)

    @staticmethod
    def _lift_helper(val):

        tp, fp, fn, tn, labels_num = val[0], val[1], val[2], val[3], val[4]

        lift_x_type, lift_y_type = [], []

        for label_type in ["1", "0"]:

            if label_type == "0":
                tp, tn = tn, tp
                fp, fn = fn, fp

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

            lift_x_type.insert(0, lift_x)
            lift_y_type.insert(0, lift_y)

        return lift_x_type, lift_y_type

    def compute_metric_from_confusion_mat(
        self,
        confusion_mat,
        labels_len,
    ):

        labels_nums = np.zeros(len(confusion_mat["tp"])) + labels_len

        rs = map(
            self._lift_helper,
            zip(confusion_mat["tp"], confusion_mat["fp"], confusion_mat["fn"], confusion_mat["tn"], labels_nums),
        )

        rs = list(rs)

        lifts_x, lifts_y = [i[0] for i in rs], [i[1] for i in rs]

        return lifts_y, lifts_x

    def __call__(self, predict, label, **kwargs):

        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(
            label,
            predict,
            add_to_end=False,
        )

        lifts_y, lifts_x = self.compute_metric_from_confusion_mat(
            confusion_mat,
            len(label),
        )

        return EvalResult(
            self.metric_name, pd.DataFrame({"liftx": lifts_x, "lifty": lifts_y, "threshold": list(score_threshold)})
        )


class Gain(Metric, BiClassMetric):

    metric_name = "gain"

    def __init__(self, *args, **kwargs):
        Metric.__init__(self)
        BiClassMetric.__init__(self, cut_method="step", remove_duplicate=False, pos_label=1)

    @staticmethod
    def _gain_helper(val):

        tp, fp, fn, tn, num_label = val[0], val[1], val[2], val[3], val[4]

        gain_x_type, gain_y_type = [], []

        for pos_label in ["1", "0"]:

            if pos_label == "0":
                tp, tn = tn, tp
                fp, fn = fn, fp

            if num_label == 0:
                gain_x = 1
            else:
                gain_x = float((tp + fp) / num_label)

            num_positives = tp + fn
            if num_positives == 0:
                gain_y = 1
            else:
                gain_y = float(tp / num_positives)

            gain_x_type.insert(0, gain_x)
            gain_y_type.insert(0, gain_y)

        return gain_x_type, gain_y_type

    def compute_metric_from_confusion_mat(self, confusion_mat, labels_len):

        labels_nums = np.zeros(len(confusion_mat["tp"])) + labels_len

        rs = map(
            self._gain_helper,
            zip(confusion_mat["tp"], confusion_mat["fp"], confusion_mat["fn"], confusion_mat["tn"], labels_nums),
        )

        rs = list(rs)

        gain_x, gain_y = [i[0] for i in rs], [i[1] for i in rs]

        return gain_y, gain_x

    def __call__(self, predict, label, **kwargs):

        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(
            label,
            predict,
            add_to_end=False,
        )

        gain_y, gain_x = self.compute_metric_from_confusion_mat(confusion_mat, len(label))

        return EvalResult(
            self.metric_name, pd.DataFrame({"gainx": gain_x, "gainy": gain_y, "threshold": list(score_threshold)})
        )


class BiClassPrecisionTable(Metric, BiClassMetric):
    """
    Compute binary classification precision using multiple thresholds
    """

    metric_name = "biclass_precision_table"

    def __init__(self, *args, **kwargs):
        Metric.__init__(self)
        BiClassMetric.__init__(self, cut_method="step", remove_duplicate=False, pos_label=1)

    def compute_metric_from_confusion_mat(self, confusion_mat, impute_val=1.0):
        numerator = confusion_mat["tp"]
        denominator = confusion_mat["tp"] + confusion_mat["fp"]
        zero_indexes = denominator == 0
        denominator[zero_indexes] = 1
        precision_scores = numerator / denominator
        precision_scores[zero_indexes] = impute_val  # impute_val is for prettifying when drawing pr curves

        return precision_scores

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        p, threshold, cuts = self.compute(label, predict)
        return EvalResult(self.metric_name, pd.DataFrame({"p": p, "threshold": threshold, "cuts": cuts}))


class BiClassRecallTable(Metric, BiClassMetric):
    """
    Compute binary classification recall using multiple thresholds
    """

    metric_name = "biclass_recall_table"

    def __init__(self, *args, **kwargs):
        Metric.__init__(self)
        BiClassMetric.__init__(self, cut_method="step", remove_duplicate=False, pos_label=1)

    def compute_metric_from_confusion_mat(self, confusion_mat, formatted=True):
        recall_scores = confusion_mat["tp"] / (confusion_mat["tp"] + confusion_mat["fn"])
        return recall_scores

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        r, threshold, cuts = self.compute(label, predict)
        return EvalResult(self.metric_name, pd.DataFrame({"r": r, "threshold": threshold, "cuts": cuts}))


class BiClassAccuracyTable(Metric, BiClassMetric):
    """
    Compute binary classification accuracy using multiple thresholds
    """

    metric_name = "biclass_accuracy_table"

    def __init__(self, *args, **kwargs):
        Metric.__init__(self)
        BiClassMetric.__init__(self, cut_method="step", remove_duplicate=False, pos_label=1)

    def compute(self, labels, scores, normalize=True):
        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(labels, scores)
        metric_scores = self.compute_metric_from_confusion_mat(confusion_mat, normalize=normalize)
        return list(metric_scores), score_threshold[: len(metric_scores)], cuts[: len(metric_scores)]

    def compute_metric_from_confusion_mat(self, confusion_mat, normalize=True):
        rs = (
            (confusion_mat["tp"] + confusion_mat["tn"])
            / (confusion_mat["tp"] + confusion_mat["tn"] + confusion_mat["fn"] + confusion_mat["fp"])
            if normalize
            else (confusion_mat["tp"] + confusion_mat["tn"])
        )
        return rs[:-1]

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        accuracy, threshold, cuts = self.compute(label, predict)
        return EvalResult(self.metric_name, pd.DataFrame({"accuracy": accuracy, "threshold": threshold, "cuts": cuts}))


class FScoreTable(Metric):
    """
    Compute F score from bi-class confusion mat
    """

    metric_name = "fscore_table"

    def __call__(self, predict, label, beta=1):

        predict = self.to_np_format(predict)
        label = self.to_np_format(label)

        sorted_labels, sorted_scores = sort_score_and_label(label, predict)
        _, cuts = ThresholdCutter.cut_by_step(sorted_scores, steps=0.01)
        fixed_interval_threshold = ThresholdCutter.fixed_interval_threshold()
        confusion_mat = _ConfusionMatrix.compute(
            sorted_labels, sorted_scores, fixed_interval_threshold, ret=["tp", "fp", "fn", "tn"]
        )
        precision_computer = BiClassPrecisionTable()
        recall_computer = BiClassRecallTable()
        p_score = precision_computer.compute_metric_from_confusion_mat(confusion_mat)
        r_score = recall_computer.compute_metric_from_confusion_mat(confusion_mat)
        beta_2 = beta * beta
        denominator = beta_2 * p_score + r_score
        denominator[denominator == 0] = 1e-6  # in case denominator is 0
        numerator = (1 + beta_2) * (p_score * r_score)
        f_score = numerator / denominator

        return EvalResult(
            self.metric_name, pd.DataFrame({"f_score": f_score, "threshold": fixed_interval_threshold, "cuts": cuts})
        )


class PSI(Metric):

    metric_name = "psi"

    def __call__(self, predict: dict, label: dict, **kwargs) -> Dict:

        """
        train/validate scores: predicted scores on train/validate set
        train/validate labels: true labels
        debug: print debug message
        if train&validate labels are not None, count positive sample percentage in every interval
        """

        str_intervals = False
        round_num = 3
        pos_label = 1

        if not isinstance(predict, dict) or (label is not None and not isinstance(label, dict)):
            raise ValueError("Input 'predict' must be a dictionary, and 'label' must be either None or a dictionary.")

        train_scores = predict.get("train_scores")
        validate_scores = predict.get("validate_scores")

        if train_scores is None or validate_scores is None:
            raise ValueError(
                "Input 'predict' should contain the following keys: 'train_scores', 'validate_scores'. "
                "Please make sure both keys are present."
            )

        train_labels = label.get("train_labels") if label is not None else None
        validate_labels = label.get("validate_labels") if label is not None else None

        train_scores = np.array(train_scores)
        validate_scores = np.array(validate_scores)
        quantile_points = ThresholdCutter().cut_by_quantile(train_scores)

        train_count = self.quantile_binning_and_count(train_scores, quantile_points)
        validate_count = self.quantile_binning_and_count(validate_scores, quantile_points)

        train_pos_perc, validate_pos_perc = None, None

        if train_labels is not None and validate_labels is not None:
            assert len(train_labels) == len(train_scores) and len(validate_labels) == len(validate_scores)
            train_labels, validate_labels = np.array(train_labels), np.array(validate_labels)
            train_pos_count = self.quantile_binning_and_count(train_scores[train_labels == pos_label], quantile_points)
            validate_pos_count = self.quantile_binning_and_count(
                validate_scores[validate_labels == pos_label], quantile_points
            )

            train_pos_perc = np.array(train_pos_count["count"]) / np.array(train_count["count"])
            validate_pos_perc = np.array(validate_pos_count["count"]) / np.array(validate_count["count"])

            # handle special cases
            train_pos_perc[train_pos_perc == np.inf] = -1
            validate_pos_perc[validate_pos_perc == np.inf] = -1
            train_pos_perc[np.isnan(train_pos_perc)] = 0
            validate_pos_perc[np.isnan(validate_pos_perc)] = 0

        assert train_count["interval"] == validate_count["interval"], (
            "train count interval is not equal to " "validate count interval"
        )

        expected_interval = np.array(train_count["count"])
        actual_interval = np.array(validate_count["count"])

        expected_interval = expected_interval.astype(np.float)
        actual_interval = actual_interval.astype(np.float)

        (
            psi_scores,
            total_psi,
            expected_interval,
            actual_interval,
            expected_percentage,
            actual_percentage,
        ) = self.psi_score(expected_interval, actual_interval, len(train_scores), len(validate_scores))

        intervals = (
            train_count["interval"]
            if not str_intervals
            else PSI.intervals_to_str(train_count["interval"], round_num=round_num)
        )

        total_psi = EvalResult("total_psi", total_psi)

        if train_labels is None and validate_labels is None:
            psi_table = EvalResult(
                "psi_table",
                pd.DataFrame(
                    {
                        "psi_scores": psi_scores,
                        "expected_interval": expected_interval,
                        "actual_interval": actual_interval,
                        "expected_percentage": expected_percentage,
                        "actual_percentage": actual_percentage,
                        "interval": intervals,
                    }
                ),
            )
        else:
            psi_table = EvalResult(
                "psi_table",
                pd.DataFrame(
                    {
                        "psi_scores": psi_scores,
                        "expected_interval": expected_interval,
                        "actual_interval": actual_interval,
                        "expected_percentage": expected_percentage,
                        "actual_percentage": actual_percentage,
                        "train_pos_perc": train_pos_perc,
                        "validate_pos_perc": validate_pos_perc,
                        "interval": intervals,
                    }
                ),
            )

        return psi_table, total_psi

    @staticmethod
    def quantile_binning_and_count(scores, quantile_points):
        """
        left edge and right edge of last interval are closed
        """

        assert len(quantile_points) >= 2

        left_bounds = copy.deepcopy(quantile_points[:-1])
        right_bounds = copy.deepcopy(quantile_points[1:])

        last_interval_left = left_bounds.pop()
        last_interval_right = right_bounds.pop()

        bin_result_1, bin_result_2 = None, None

        if len(left_bounds) != 0 and len(right_bounds) != 0:
            bin_result_1 = pd.cut(scores, pd.IntervalIndex.from_arrays(left_bounds, right_bounds, closed="left"))

        bin_result_2 = pd.cut(
            scores, pd.IntervalIndex.from_arrays([last_interval_left], [last_interval_right], closed="both")
        )

        count1 = None if bin_result_1 is None else bin_result_1.value_counts().reset_index()
        count2 = bin_result_2.value_counts().reset_index()

        # if predict scores are the same, count1 will be None, only one interval exists
        final_interval = list(count1["index"]) + list(count2["index"]) if count1 is not None else list(count2["index"])
        final_count = list(count1[0]) + list(count2[0]) if count1 is not None else list(count2[0])
        rs = {"interval": final_interval, "count": final_count}

        return rs

    @staticmethod
    def interval_psi_score(val):
        expected, actual = val[0], val[1]
        return (actual - expected) * np.log(actual / expected)

    @staticmethod
    def intervals_to_str(intervals, round_num=3):
        str_intervals = []
        for interval in intervals:
            left_bound, right_bound = "[", "]"
            if interval.closed == "left":
                right_bound = ")"
            elif interval.closed == "right":
                left_bound = "("
            str_intervals.append(
                "{}{}, {}{}".format(
                    left_bound, round(interval.left, round_num), round(interval.right, round_num), right_bound
                )
            )

        return str_intervals

    @staticmethod
    def psi_score(
        expected_interval: np.ndarray, actual_interval: np.ndarray, expect_total_num, actual_total_num, debug=False
    ):

        expected_interval[expected_interval == 0] = 1e-6  # in case no overlap samples

        actual_interval[actual_interval == 0] = 1e-6  # in case no overlap samples

        expected_percentage = expected_interval / expect_total_num
        actual_percentage = actual_interval / actual_total_num

        if debug:
            print(expected_interval)
            print(actual_interval)
            print(expected_percentage)
            print(actual_percentage)

        psi_scores = list(map(PSI.interval_psi_score, zip(expected_percentage, actual_percentage)))
        psi_scores = np.array(psi_scores)
        total_psi = psi_scores.sum()
        return psi_scores, total_psi, expected_interval, actual_interval, expected_percentage, actual_percentage
