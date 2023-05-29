import unittest
from fate.ml.evaluation.regression import *
from fate.ml.evaluation.classi import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.datasets import make_classification

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_binary_true = np.random.randint(0, 2, 100)
        self.y_binary_scores = np.random.random(100)
        self.y_multi_true = np.random.randint(0, 3, 100)
        self.y_multi_scores = np.random.random((100, 3))
        self.y_reg_true = np.random.random(100) * 10
        self.y_reg_pred = self.y_reg_true + np.random.randn(100)

    def test_binary_metrics(self):
        metrics = [
            (BinaryAccuracy(), accuracy_score),
            (BinaryRecall(), recall_score),
            (BinaryPrecision(), precision_score),
            (BinaryF1Score(), f1_score),
        ]
        for metric, sklearn_metric in metrics:
            our_result = metric(self.y_binary_scores, self.y_binary_true)["accuracy"]
            sklearn_result = sklearn_metric(self.y_binary_true, self.y_binary_scores > 0.5)
            self.assertAlmostEqual(our_result, sklearn_result, places=5)

    def test_multi_metrics(self):
        metrics = [
            (MultiAccuracy(), accuracy_score),
            (MultiRecall(), recall_score),
            (MultiPrecision(), precision_score),
            (MultiF1Score(), f1_score),
        ]
        for metric, sklearn_metric in metrics:
            our_result = metric(self.y_multi_scores, self.y_multi_true)["accuracy"]
            sklearn_result = sklearn_metric(self.y_multi_true, self.y_multi_scores.argmax(axis=-1), average='micro')
            self.assertAlmostEqual(our_result, sklearn_result, places=5)

    def test_regression_metrics(self):
        metrics = [
            (RMSE(), lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
            (MSE(), mean_squared_error),
            (MAE(), mean_absolute_error),
            (R2Score(), r2_score),
        ]
        for metric, sklearn_metric in metrics:
            our_result = metric(self.y_reg_pred, self.y_reg_true)["mse"]
            sklearn_result = sklearn_metric(self.y_reg_true, self.y_reg_pred)
            self.assertAlmostEqual(our_result, sklearn_result, places=5)


if __name__ == '__main__':
    unittest.main()
