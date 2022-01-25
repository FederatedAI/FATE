import copy
import sys

import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

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


class ConfusionMatrix(object):

    @staticmethod
    def compute(sorted_labels: list, sorted_pred_scores: list, score_thresholds: list, ret: list, pos_label=1):

        for ret_type in ret:
            assert ret_type in ['tp', 'tn', 'fp', 'fn']

        sorted_labels = np.array(sorted_labels)
        sorted_scores = np.array(sorted_pred_scores)
        sorted_labels[sorted_labels != pos_label] = 0
        sorted_labels[sorted_labels == pos_label] = 1
        score_thresholds = np.array([score_thresholds]).transpose()
        pred_labels = (sorted_scores > score_thresholds) + 0

        ret_dict = {}
        if 'tp' in ret or 'tn' in ret:
            match_arr = (pred_labels + sorted_labels)
            if 'tp' in ret:
                tp_num = (match_arr == 2).sum(axis=-1)
                ret_dict['tp'] = tp_num
            if 'tn' in ret:
                tn_num = (match_arr == 0).sum(axis=-1)
                ret_dict['tn'] = tn_num

        if 'fp' in ret or 'fn' in ret:
            match_arr = (sorted_labels - pred_labels)
            if 'fp' in ret:
                fp_num = (match_arr == -1).sum(axis=-1)
                ret_dict['fp'] = fp_num
            if 'fn' in ret:
                fn_num = (match_arr == 1).sum(axis=-1)
                ret_dict['fn'] = fn_num

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
    def cut_by_quantile(scores, quantile_list=None, interpolation='nearest', remove_duplicate=True):

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


class KS(object):

    @staticmethod
    def compute(labels, pred_scores, pos_label=1, fixed_interval_threshold=True):
        sorted_labels, sorted_scores = sort_score_and_label(labels, pred_scores)

        threshold, cuts = ThresholdCutter.cut_by_index(sorted_scores)
        confusion_mat = ConfusionMatrix.compute(sorted_labels, sorted_scores, threshold, ret=['tp', 'fp'],
                                                pos_label=pos_label)
        pos_num, neg_num = neg_pos_count(sorted_labels, pos_label=pos_label)

        assert pos_num > 0 and neg_num > 0, "error when computing KS metric, pos sample number and neg sample number" \
                                            "must be larger than 0"

        tpr_arr = confusion_mat['tp'] / pos_num
        fpr_arr = confusion_mat['fp'] / neg_num

        tpr = np.append(tpr_arr, np.array([1.0]))
        fpr = np.append(fpr_arr, np.array([1.0]))
        cuts = np.append(cuts, np.array([1.0]))

        ks_curve = tpr[:-1] - fpr[:-1]
        ks_val = np.max(ks_curve)

        return ks_val, fpr, tpr, threshold, cuts


class BiClassMetric(object):

    def __init__(self, cut_method='step', remove_duplicate=False, pos_label=1):
        assert cut_method in ['step', 'quantile']
        self.cut_method = cut_method
        self.remove_duplicate = remove_duplicate  # available when cut_method is quantile
        self.pos_label = pos_label

    def prepare_confusion_mat(self, labels, scores, add_to_end=True, ):
        sorted_labels, sorted_scores = sort_score_and_label(labels, scores)

        score_threshold, cuts = None, None

        if self.cut_method == 'step':
            score_threshold, cuts = ThresholdCutter.cut_by_step(sorted_scores, steps=0.01)
            if add_to_end:
                score_threshold.append(min(score_threshold) - 0.001)
                cuts.append(1)

        elif self.cut_method == 'quantile':
            score_threshold = ThresholdCutter.cut_by_quantile(sorted_scores, remove_duplicate=self.remove_duplicate)
            score_threshold = list(np.flip(score_threshold))

        confusion_mat = ConfusionMatrix.compute(sorted_labels, sorted_scores, score_threshold,
                                                ret=['tp', 'fp', 'fn', 'tn'], pos_label=self.pos_label)

        return confusion_mat, score_threshold, cuts

    def compute(self, labels, scores, ):
        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(labels, scores, )
        metric_scores = self.compute_metric_from_confusion_mat(confusion_mat)
        return list(metric_scores), score_threshold, cuts

    def compute_metric_from_confusion_mat(self, *args):
        raise NotImplementedError()


class Lift(BiClassMetric):
    """
    Compute lift
    """

    @staticmethod
    def _lift_helper(val):

        tp, fp, fn, tn, labels_num = val[0], val[1], val[2], val[3], val[4]

        lift_x_type, lift_y_type = [], []

        for label_type in ['1', '0']:

            if label_type == '0':
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

    def compute(self, labels, pred_scores, pos_label=1):

        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(labels, pred_scores, add_to_end=False, )

        lifts_y, lifts_x = self.compute_metric_from_confusion_mat(confusion_mat, len(labels), )

        return lifts_y, lifts_x, list(score_threshold)

    def compute_metric_from_confusion_mat(self, confusion_mat, labels_len, ):

        labels_nums = np.zeros(len(confusion_mat['tp'])) + labels_len

        rs = map(self._lift_helper, zip(confusion_mat['tp'], confusion_mat['fp'],
                                        confusion_mat['fn'], confusion_mat['tn'], labels_nums))

        rs = list(rs)

        lifts_x, lifts_y = [i[0] for i in rs], [i[1] for i in rs]

        return lifts_y, lifts_x


class Gain(BiClassMetric):
    """
    Compute Gain
    """

    @staticmethod
    def _gain_helper(val):

        tp, fp, fn, tn, num_label = val[0], val[1], val[2], val[3], val[4]

        gain_x_type, gain_y_type = [], []

        for pos_label in ['1', '0']:

            if pos_label == '0':
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

    def compute(self, labels, pred_scores, pos_label=1):

        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(labels, pred_scores, add_to_end=False, )

        gain_y, gain_x = self.compute_metric_from_confusion_mat(confusion_mat, len(labels))

        return gain_y, gain_x, list(score_threshold)

    def compute_metric_from_confusion_mat(self, confusion_mat, labels_len):

        labels_nums = np.zeros(len(confusion_mat['tp'])) + labels_len

        rs = map(self._gain_helper, zip(confusion_mat['tp'], confusion_mat['fp'],
                                        confusion_mat['fn'], confusion_mat['tn'], labels_nums))

        rs = list(rs)

        gain_x, gain_y = [i[0] for i in rs], [i[1] for i in rs]

        return gain_y, gain_x


class BiClassPrecision(BiClassMetric):
    """
    Compute binary classification precision
    """

    def compute_metric_from_confusion_mat(self, confusion_mat, formatted=True, impute_val=1.0):
        numerator = confusion_mat['tp']
        denominator = (confusion_mat['tp'] + confusion_mat['fp'])
        zero_indexes = (denominator == 0)
        denominator[zero_indexes] = 1
        precision_scores = numerator / denominator
        precision_scores[zero_indexes] = impute_val  # impute_val is for prettifying when drawing pr curves

        if formatted:
            score_formatted = [[0, i] for i in precision_scores]
            return score_formatted
        else:
            return precision_scores


class MultiClassPrecision(object):
    """
    Compute multi-classification precision
    """

    def compute(self, labels, pred_scores):
        all_labels = sorted(set(labels).union(set(pred_scores)))
        return precision_score(labels, pred_scores, average=None), all_labels


class BiClassRecall(BiClassMetric):
    """
    Compute binary classification recall
    """

    def compute_metric_from_confusion_mat(self, confusion_mat, formatted=True):
        recall_scores = confusion_mat['tp'] / (confusion_mat['tp'] + confusion_mat['fn'])

        if formatted:
            score_formatted = [[0, i] for i in recall_scores]
            return score_formatted
        else:
            return recall_scores


class MultiClassRecall(object):
    """
    Compute multi-classification recall
    """

    def compute(self, labels, pred_scores):
        all_labels = sorted(set(labels).union(set(pred_scores)))
        return recall_score(labels, pred_scores, average=None), all_labels


class BiClassAccuracy(BiClassMetric):
    """
    Compute binary classification accuracy
    """

    def compute(self, labels, scores, normalize=True):
        confusion_mat, score_threshold, cuts = self.prepare_confusion_mat(labels, scores)
        metric_scores = self.compute_metric_from_confusion_mat(confusion_mat, normalize=normalize)
        return list(metric_scores), score_threshold[: len(metric_scores)], cuts[: len(metric_scores)]

    def compute_metric_from_confusion_mat(self, confusion_mat, normalize=True):
        rs = (confusion_mat['tp'] + confusion_mat['tn']) / \
             (confusion_mat['tp'] + confusion_mat['tn'] + confusion_mat['fn'] + confusion_mat['fp']) if normalize \
            else (confusion_mat['tp'] + confusion_mat['tn'])
        return rs[:-1]


class MultiClassAccuracy(object):
    """
    Compute multi-classification accuracy
    """

    def compute(self, labels, pred_scores, normalize=True):
        return accuracy_score(labels, pred_scores, normalize)


class FScore(object):
    """
    Compute F score from bi-class confusion mat
    """

    @staticmethod
    def compute(labels, pred_scores, beta=1, pos_label=1):
        sorted_labels, sorted_scores = sort_score_and_label(labels, pred_scores)
        _, cuts = ThresholdCutter.cut_by_step(sorted_scores, steps=0.01)
        fixed_interval_threshold = ThresholdCutter.fixed_interval_threshold()
        confusion_mat = ConfusionMatrix.compute(sorted_labels, sorted_scores,
                                                fixed_interval_threshold,
                                                ret=['tp', 'fp', 'fn', 'tn'], pos_label=pos_label)

        precision_computer = BiClassPrecision()
        recall_computer = BiClassRecall()
        p_score = precision_computer.compute_metric_from_confusion_mat(confusion_mat, formatted=False)
        r_score = recall_computer.compute_metric_from_confusion_mat(confusion_mat, formatted=False)

        beta_2 = beta * beta
        denominator = (beta_2 * p_score + r_score)
        denominator[denominator == 0] = 1e-6  # in case denominator is 0
        numerator = (1 + beta_2) * (p_score * r_score)
        f_score = numerator / denominator

        return f_score, fixed_interval_threshold, cuts


class PSI(object):

    def compute(self, train_scores: list, validate_scores: list, train_labels=None, validate_labels=None,
                debug=False, str_intervals=False, round_num=3, pos_label=1):
        """
        train/validate scores: predicted scores on train/validate set
        train/validate labels: true labels
        debug: print debug message
        if train&validate labels are not None, count positive sample percentage in every interval
        pos_label: pos label
        round_num： round number
        str_intervals: return str intervals
        """

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
            validate_pos_count = self.quantile_binning_and_count(validate_scores[validate_labels == pos_label],
                                                                 quantile_points)

            train_pos_perc = np.array(train_pos_count['count']) / np.array(train_count['count'])
            validate_pos_perc = np.array(validate_pos_count['count']) / np.array(validate_count['count'])

            # handle special cases
            train_pos_perc[train_pos_perc == np.inf] = -1
            validate_pos_perc[validate_pos_perc == np.inf] = -1
            train_pos_perc[np.isnan(train_pos_perc)] = 0
            validate_pos_perc[np.isnan(validate_pos_perc)] = 0

        if debug:
            print(train_count)
            print(validate_count)

        assert (train_count['interval'] == validate_count['interval']), 'train count interval is not equal to ' \
                                                                        'validate count interval'

        expected_interval = np.array(train_count['count'])
        actual_interval = np.array(validate_count['count'])

        expected_interval = expected_interval.astype(np.float)
        actual_interval = actual_interval.astype(np.float)

        psi_scores, total_psi, expected_interval, actual_interval, expected_percentage, actual_percentage \
            = self.psi_score(expected_interval, actual_interval, len(train_scores), len(validate_scores))

        intervals = train_count['interval'] if not str_intervals else PSI.intervals_to_str(train_count['interval'],
                                                                                           round_num=round_num)

        if train_labels is None and validate_labels is None:
            return psi_scores, total_psi, expected_interval, expected_percentage, actual_interval, actual_percentage, \
                intervals
        else:
            return psi_scores, total_psi, expected_interval, expected_percentage, actual_interval, actual_percentage, \
                train_pos_perc, validate_pos_perc, intervals

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
            bin_result_1 = pd.cut(scores, pd.IntervalIndex.from_arrays(left_bounds, right_bounds, closed='left'))

        bin_result_2 = pd.cut(scores, pd.IntervalIndex.from_arrays([last_interval_left], [last_interval_right],
                                                                   closed='both'))

        count1 = None if bin_result_1 is None else bin_result_1.value_counts().reset_index()
        count2 = bin_result_2.value_counts().reset_index()

        # if predict scores are the same, count1 will be None, only one interval exists
        final_interval = list(count1['index']) + list(count2['index']) if count1 is not None else list(count2['index'])
        final_count = list(count1[0]) + list(count2[0]) if count1 is not None else list(count2[0])
        rs = {'interval': final_interval, 'count': final_count}

        return rs

    @staticmethod
    def interval_psi_score(val):
        expected, actual = val[0], val[1]
        return (actual - expected) * np.log(actual / expected)

    @staticmethod
    def intervals_to_str(intervals, round_num=3):
        str_intervals = []
        for interval in intervals:
            left_bound, right_bound = '[', ']'
            if interval.closed == 'left':
                right_bound = ')'
            elif interval.closed == 'right':
                left_bound = '('
            str_intervals.append("{}{}, {}{}".format(left_bound, round(interval.left, round_num),
                                                     round(interval.right, round_num), right_bound))

        return str_intervals

    @staticmethod
    def psi_score(expected_interval: np.ndarray, actual_interval: np.ndarray, expect_total_num, actual_total_num,
                  debug=False):

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


class KSTest(object):

    @staticmethod
    def compute(train_scores, validate_scores):
        """
        train/validate scores: predicted scores on train/validate set
        """
        return stats.ks_2samp(train_scores, validate_scores).pvalue


class AveragePrecisionScore(object):

    @staticmethod
    def compute(train_scores, validate_scores, train_labels, validate_labels):
        """
            train/validate scores: predicted scores on train/validate set
            train/validate labels: true labels
        """
        train_mAP = average_precision_score(train_labels, train_scores)
        validate_mAP = average_precision_score(validate_labels, validate_scores)
        return abs(train_mAP - validate_mAP)


class Distribution(object):

    @staticmethod
    def compute(train_scores: list, validate_scores: list):
        """
        train/validate scores: predicted scores on train/validate set
        """
        train_scores = np.array(train_scores)
        validate_scores = np.array(validate_scores)
        validate_scores = dict(validate_scores)
        count = 0
        for key, value in train_scores:
            if key in validate_scores.keys() and value != validate_scores.get(key):
                count += 1
        return count / len(train_scores)
