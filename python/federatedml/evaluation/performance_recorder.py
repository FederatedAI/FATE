from federatedml.util import consts


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
