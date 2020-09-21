from federatedml.evaluation.metrics.classification_metric \
    import BiClassAccuracy, BiClassRecall, BiClassPrecision, KS, Lift, Gain, FScore
from federatedml.evaluation.metric_interface import Metrics

from sklearn.metrics import f1_score

import numpy as np

import time

scores = np.random.random(10000)
labels = (scores > 0.5) + 0

f_score_computer = FScore()
f_scores, thres, cuts = f_score_computer.compute(labels, scores)

thres = np.array([thres]).transpose()
pred_labels = (scores > thres) + 0

true_f1 = []
for preds in pred_labels:
    f1 = f1_score(labels, preds)
    true_f1.append(f1)
