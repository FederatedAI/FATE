import numpy as np
import unittest
from fate.ml.evaluation import classification as classi
from fate.ml.evaluation.metric_base import MetricEnsemble
from fate.ml.evaluation.tool import get_binary_metrics, get_specified_metrics, get_single_val_binary_metrics


def generate_predict_and_label(num):
    predict = np.random.random_sample(num)
    label = np.random.randint(0, 2, num)
    return predict, label


class TestMetric(unittest.TestCase):

    def test_binary_ensemble(self):
        
        binary_ensemble = get_binary_metrics()

        predict, label = generate_predict_and_label(1000)
        result = binary_ensemble(predict=predict, label=label)
        print(result)
        print(type(result))

    
    def test_selected(self):
        metrics = get_specified_metrics(['auc', 'ks', 'lift', 'gain', 'binary_accuracy'])
        predict, label = generate_predict_and_label(1000)
        result = metrics(predict=predict, label=label)
        print(result)
        print(type(result))

    def test_binary_ensemble(self):
        
        binary_ensemble = get_single_val_binary_metrics(threshold=0.8)

        predict, label = generate_predict_and_label(1000)
        result = binary_ensemble(predict=predict, label=label)
        print(result)
        print(type(result))



if __name__ == '__main__':
    unittest.main()