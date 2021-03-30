from scipy.stats import stats
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import numpy as np


class RMSE(object):

    @staticmethod
    def compute(labels, pred_scores):
        return np.sqrt(mean_squared_error(labels, pred_scores))


class MAE(object):

    @staticmethod
    def compute(labels, pred_scores):
        return mean_absolute_error(labels, pred_scores)


class R2Score(object):

    @staticmethod
    def compute(labels, pred_scores):
        return r2_score(labels, pred_scores)


class MSE(object):

    @staticmethod
    def compute(labels, pred_scores):
        return mean_squared_error(labels, pred_scores)


class ExplainedVariance(object):

    @staticmethod
    def compute(labels, pred_scores):
        return explained_variance_score(labels, pred_scores)


class MedianAbsoluteError(object):

    @staticmethod
    def compute(labels, pred_scores):
        return median_absolute_error(labels, pred_scores)


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


class Describe(object):

    @staticmethod
    def compute(pred_scores):
        describe = stats.describe(pred_scores)
        metrics = {"min": describe.minmax[0], "max": describe.minmax[1], "mean": describe.mean,
                   "variance": describe.variance, "skewness": describe.skewness, "kurtosis": describe.kurtosis}
        return metrics
