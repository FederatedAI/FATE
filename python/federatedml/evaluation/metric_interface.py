from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np
import logging
from federatedml.util import consts
from federatedml.evaluation.metrics import classification_metric
from federatedml.evaluation.metrics import regression_metric
from federatedml.evaluation.metrics import clustering_metric

from functools import wraps


class MetricInterface(object):

    def __init__(self, pos_label: int, eval_type: str):

        self.pos_label = pos_label
        self.eval_type = eval_type

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
        elif self.eval_type == consts.ONE_VS_REST:
            try:
                score = roc_auc_score(labels, pred_scores)
            except BaseException:
                score = 0  # in case all labels are 0 or 1
                logging.warning("all true labels are 0/1 when running ovr AUC")
            return score
        else:
            logging.warning("auc is just suppose Binary Classification! return None as results")
            return None

    @staticmethod
    def explained_variance(labels, pred_scores):
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
        return regression_metric.ExplainedVariance().compute(labels, pred_scores)

    @staticmethod
    def mean_absolute_error(labels, pred_scores):
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
        return regression_metric.MAE().compute(labels, pred_scores)

    @staticmethod
    def mean_squared_error(labels, pred_scores):
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
        return regression_metric.MSE.compute(labels, pred_scores)

    @staticmethod
    def median_absolute_error(labels, pred_scores):
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
        return regression_metric.MedianAbsoluteError().compute(labels, pred_scores)

    @staticmethod
    def r2_score(labels, pred_scores):
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
        return regression_metric.R2Score().compute(labels, pred_scores)

    @staticmethod
    def root_mean_squared_error(labels, pred_scores):
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
        return regression_metric.RMSE.compute(labels, pred_scores)

    @staticmethod
    def __to_int_list(array: np.ndarray):
        return list(map(int, list(array)))

    @staticmethod
    def __filt_threshold(thresholds, step):
        cuts = list(map(float, np.arange(0, 1, step)))
        size = len(list(thresholds))
        thresholds.sort(reverse=True)
        index_list = [int(size * cut) for cut in cuts]
        new_thresholds = [thresholds[idx] for idx in index_list]

        return new_thresholds, cuts

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

        if self.eval_type == consts.ONE_VS_REST:
            try:
                rs = classification_metric.KS().compute(labels, pred_scores)
            except BaseException:
                rs = [0, [0], [0], [0], [0]]   # in case all labels are 0 or 1
                logging.warning("all true labels are 0/1 when running ovr KS")
            return rs
        else:
            return classification_metric.KS().compute(labels, pred_scores)

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
            return classification_metric.Lift().compute(labels, pred_scores)
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
            return classification_metric.Gain().compute(labels, pred_scores)
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
            precision_operator = classification_metric.BiClassPrecision()
            metric_scores, score_threshold, cuts = precision_operator.compute(labels, pred_scores)
            return metric_scores, cuts, score_threshold
        elif self.eval_type == consts.MULTY:
            precision_operator = classification_metric.MultiClassPrecision()
            return precision_operator.compute(labels, pred_scores)
        else:
            logging.warning("error:can not find classification type:{}".format(self.eval_type))

    def recall(self, labels, pred_scores):
        """
        Compute the recall
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_scores: pred_scores: value list. The predict results of model. It should be corresponding to labels each
        data.
        Returns
        ----------
        dict
            The key is threshold and the value is another dic, which key is label in parameter labels, and value is the
            label's recall.
        """
        if self.eval_type == consts.BINARY:
            recall_operator = classification_metric.BiClassRecall()
            recall_res, thresholds, cuts = recall_operator.compute(labels, pred_scores)
            return recall_res, cuts, thresholds
        elif self.eval_type == consts.MULTY:
            recall_operator = classification_metric.MultiClassRecall()
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
            acc_operator = classification_metric.BiClassAccuracy()
            acc_res, thresholds, cuts = acc_operator.compute(labels, pred_scores, normalize)
            return acc_res, cuts, thresholds
        elif self.eval_type == consts.MULTY:
            acc_operator = classification_metric.MultiClassAccuracy()
            return acc_operator.compute(labels, pred_scores, normalize)
        else:
            logging.warning("error:can not find classification type:".format(self.eval_type))

    def f1_score(self, labels, pred_scores):
        """
        compute f1_score for binary classification result
        """

        if self.eval_type == consts.BINARY:
            f1_scores, score_threshold, cuts = classification_metric.FScore().compute(labels, pred_scores)
            return list(f1_scores), list(cuts), list(score_threshold)
        else:
            logging.warning('error: f-score metric is for binary classification only')

    def confusion_mat(self, labels, pred_scores):
        """
        compute confusion matrix
        """

        if self.eval_type == consts.BINARY:

            sorted_labels, sorted_scores = classification_metric.sort_score_and_label(labels, pred_scores)
            _, cuts = classification_metric.ThresholdCutter.cut_by_step(sorted_scores, steps=0.01)
            fixed_interval_threshold = classification_metric.ThresholdCutter.fixed_interval_threshold()
            confusion_mat = classification_metric.ConfusionMatrix.compute(sorted_labels, sorted_scores,
                                                                          fixed_interval_threshold,
                                                                          ret=['tp', 'fp', 'fn', 'tn'])

            confusion_mat['tp'] = self.__to_int_list(confusion_mat['tp'])
            confusion_mat['fp'] = self.__to_int_list(confusion_mat['fp'])
            confusion_mat['fn'] = self.__to_int_list(confusion_mat['fn'])
            confusion_mat['tn'] = self.__to_int_list(confusion_mat['tn'])

            return confusion_mat, cuts, fixed_interval_threshold
        else:
            logging.warning('error: f-score metric is for binary classification only')

    def psi(self, train_scores, validate_scores, train_labels, validate_labels, debug=False):
        """
        Compute the PSI index
        Parameters
        ----------
        train_scores: The predict results of train data
        validate_scores: The predict results of validate data
        train_labels: labels of train set
        validate_labels: labels of validate set
        debug: print additional info
        """
        if self.eval_type == consts.BINARY:
            psi_computer = classification_metric.PSI()
            psi_scores, total_psi, expected_interval, expected_percentage, actual_interval, actual_percentage, \
                train_pos_perc, validate_pos_perc, intervals = psi_computer.compute(train_scores, validate_scores,
                                                                                    debug=debug, str_intervals=True,
                                                                                    round_num=6, train_labels=train_labels, validate_labels=validate_labels)

            len_list = np.array([len(psi_scores), len(expected_interval), len(expected_percentage),
                                 len(actual_interval), len(actual_percentage), len(intervals)])

            assert (len_list == len(psi_scores)).all()

            return list(psi_scores), total_psi, self.__to_int_list(expected_interval), list(expected_percentage), \
                self.__to_int_list(actual_interval), list(actual_percentage), list(train_pos_perc), \
                list(validate_pos_perc), intervals

        else:
            logging.warning('error: psi metric is for binary classification only')

    def quantile_pr(self, labels, pred_scores):
        if self.eval_type == consts.BINARY:
            p = classification_metric.BiClassPrecision(cut_method='quantile', remove_duplicate=False)
            r = classification_metric.BiClassRecall(cut_method='quantile', remove_duplicate=False)
            p_scores, score_threshold, cuts = p.compute(labels, pred_scores)
            r_scores, score_threshold, cuts = r.compute(labels, pred_scores)
            p_scores = list(map(list, np.flip(p_scores, axis=0)))
            r_scores = list(map(list, np.flip(r_scores, axis=0)))
            score_threshold = list(np.flip(score_threshold))
            return p_scores, r_scores, score_threshold
        else:
            logging.warning('error: pr quantile is for binary classification only')

    @staticmethod
    def jaccard_similarity_score(labels, pred_labels):
        """
        Compute the Jaccard similarity score
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_labels: value list. The predict results of model. It should be corresponding to labels each data.
        Return
        ----------
        float
            A positive floating point value
        """

        return clustering_metric.JaccardSimilarityScore().compute(labels, pred_labels)

    @staticmethod
    def fowlkes_mallows_score(labels, pred_labels):
        """
        Compute the Fowlkes Mallows score
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_labels: value list. The predict results of model. It should be corresponding to labels each data.
        Return
        ----------
        float
            A positive floating point value
        """

        return clustering_metric.FowlkesMallowsScore().compute(labels, pred_labels)

    @staticmethod
    def adjusted_rand_score(labels, pred_labels):
        """
        Compute the adjusted-rand score
        Parameters
        ----------
        labels: value list. The labels of data set.
        pred_labels: value list. The predict results of model. It should be corresponding to labels each data.
        Return
        ----------
        float
            A positive floating point value
        """

        return clustering_metric.AdjustedRandScore().compute(labels, pred_labels)

    @staticmethod
    def davies_bouldin_index(cluster_avg_intra_dist, cluster_inter_dist):
        """
        Compute the davies_bouldin_index
        Parameters

        """
        # process data from evaluation
        return clustering_metric.DaviesBouldinIndex().compute(cluster_avg_intra_dist, cluster_inter_dist)

    @staticmethod
    def contingency_matrix(labels, pred_labels):
        """

        """

        return clustering_metric.ContengincyMatrix().compute(labels, pred_labels)

    @staticmethod
    def distance_measure(cluster_avg_intra_dist, cluster_inter_dist, max_radius):
        """

        """
        return clustering_metric.DistanceMeasure().compute(cluster_avg_intra_dist, cluster_inter_dist, max_radius)
