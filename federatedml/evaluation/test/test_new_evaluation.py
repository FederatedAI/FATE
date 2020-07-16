from federatedml.evaluation.metrics.classification_metric \
    import BiClassAccuracy, BiClassRecall, BiClassPrecision, KS, Lift, Gain, FScore

from federatedml.evaluation.metric_interface import MetricInterface

from federatedml.evaluation.metrics import classification_metric

import numpy as np

from federatedml.evaluation.backup.evaluation import BiClassPrecision as BiClassPrecision2


scores = np.random.random(100)
labels = (scores > 0.5) + 0

interface = MetricInterface(1, 'binary')
mat = interface.confusion_mat(labels, scores)

rs = classification_metric.ThresholdCutter.cut_by_quantile(scores,)
rs2 = classification_metric.BiClassRecall(cut_method='quantile').compute(labels, scores)
rs3 = classification_metric.BiClassPrecision(cut_method='quantile').compute(labels, scores)

rs4 = BiClassPrecision2().compute(labels, scores, rs2[1])

comp1 = [i[1] for i in rs3[0]]
comp2 = [i[1] for i in rs4[0]]

print(comp1 == comp2)
