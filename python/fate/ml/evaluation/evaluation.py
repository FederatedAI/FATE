from sklearn import metrics


class BinaryEvaluator(object):
    def __init__(self):
        self._auc = None

    def fit(self, ctx, y_true, y_pred):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        self._auc = metrics.auc(fpr, tpr)

        self._report(ctx)

    def _report(self, ctx):
        ctx.metrics.log_auc("auc", self._auc)
