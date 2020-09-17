from federatedml.evaluation.metrics.classification_metric \
    import BiClassAccuracy, BiClassRecall, BiClassPrecision, KS, Lift, Gain, FScore

from federatedml.evaluation.metric_interface import MetricInterface

from federatedml.evaluation.metrics import classification_metric

import numpy as np


scores = np.random.random(100)
labels = (scores > 0.5) + 0
labels2 = (scores > 0.5) + 1

interface = MetricInterface(1, 'binary')
interface2 = MetricInterface(2, 'binary')

rs1 = interface.ks(labels, scores)
rs2 = interface2.ks(labels2, scores)

mat1 = interface.confusion_mat(labels, scores)
mat2 = interface2.confusion_mat(labels2, scores)

rs3 = interface.lift(labels, scores)
rs4 = interface2.lift(labels2, scores)

rs5 = interface.accuracy(labels, scores)
rs6 = interface2.accuracy(labels2, scores)

rs7 = interface.f1_score(labels, scores)
rs8 = interface2.f1_score(labels2, scores)

# mat = interface.confusion_mat(labels, scores)
# rs = classification_metric.ThresholdCutter.cut_by_quantile(scores,)
# rs2 = classification_metric.BiClassRecall(cut_method='quantile').compute(labels, scores)
# rs3 = classification_metric.BiClassPrecision(cut_method='quantile').compute(labels, scores)
