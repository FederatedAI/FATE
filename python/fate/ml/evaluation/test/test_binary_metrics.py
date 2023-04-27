from fate.ml.evaluation.classification import *
from fate.ml.evaluation.metric_base import MetricPipeline

# test metrics
fake_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
fake_label = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
# metrics
auc = AUC()
acc = BinaryAccuracy()
recall = BinaryRecall()
precision = BinaryPrecision()
# test
print(auc(fake_pred, fake_label))
print(acc(fake_pred, fake_label))
print(recall(fake_pred, fake_label))
print(precision(fake_pred, fake_label))
    

pipeline = MetricPipeline()
pipeline.add_metric(auc)
pipeline.add_metric(acc)
pipeline.add_metric(recall)
pipeline.add_metric(precision)
print(pipeline(fake_pred, fake_label))