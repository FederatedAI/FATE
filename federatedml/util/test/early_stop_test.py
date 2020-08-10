import unittest
from federatedml.util.validation_strategy import ValidationStrategy
import numpy as np
from federatedml.util import consts
from federatedml.param.evaluation_param import EvaluateParam


class TestValidationStrategy(unittest.TestCase):

    def setUp(self) -> None:
        self.role = 'guest'
        self.mode = 'hetero'
        self.early_stopping_round = 1
        self.use_first_metric_only = False

    @staticmethod
    def generate_fake_eval_metrics(total_rounds, decrease_round, metrics=['ks', 'auc'], start_val=0.8):
        assert total_rounds >= decrease_round
        eval_result_list = []
        start_decrease_round = total_rounds - decrease_round
        for i in range(total_rounds):
            if i < start_decrease_round:
                start_val += 0.01
            else:
                start_val -= 0.01

            eval_dict = {metric: start_val for metric in metrics}
            eval_result_list.append(eval_dict)
        return eval_result_list

    def test_early_stopping(self):

        test_rounds = [i for i in range(10, 100)]
        decrease_rounds = [np.random.randint(i) for i in test_rounds]

        for test_round, decrease_round in zip(test_rounds, decrease_rounds):

            eval_dicts = self.generate_fake_eval_metrics(test_round, decrease_round, )
            self.early_stopping_round = decrease_round - 1

            if self.early_stopping_round <= 0:
                continue

            validation_strategy = ValidationStrategy(self.role, self.mode, early_stopping_rounds=self.early_stopping_round,
                                                     use_first_metric_only=self.use_first_metric_only)

            for idx, eval_res in enumerate(eval_dicts):
                validation_strategy.performance_recorder.update(eval_res)
                check_rs = validation_strategy.check_early_stopping()
                if check_rs:
                    self.assertTrue((test_round - decrease_round + self.early_stopping_round - 1) == idx)
                    print('test checking passed')
                    break

    def test_use_first_metric_only(self):

        def evaluate(param, early_stopping_rounds, use_first_metric_only):

            eval_type = param.eval_type
            metric_list = param.metrics
            first_metric = None

            if early_stopping_rounds and use_first_metric_only and len(metric_list) != 0:

                single_metric_list = None
                if eval_type == consts.BINARY:
                    single_metric_list = consts.BINARY_SINGLE_VALUE_METRIC
                elif eval_type == consts.REGRESSION:
                    single_metric_list = consts.REGRESSION_SINGLE_VALUE_METRICS
                elif eval_type == consts.MULTY:
                    single_metric_list = consts.MULTI_SINGLE_VALUE_METRIC

                for metric in metric_list:
                    if metric in single_metric_list:
                        first_metric = metric
                        break

            return first_metric

        param_0 = EvaluateParam(metrics=['roc', 'lift', 'ks', 'auc', 'gain'], eval_type='binary')
        param_1 = EvaluateParam(metrics=['acc', 'precision', 'auc'], eval_type='binary')
        param_2 = EvaluateParam(metrics=['acc', 'precision', 'gain', 'recall', 'lift'], eval_type='binary')
        param_3 = EvaluateParam(metrics=['acc', 'precision', 'gain', 'auc', 'recall'], eval_type='multi')

        print(evaluate(param_0, 10, True))
        print(evaluate(param_1, 10, True))
        print(evaluate(param_2, 10, True))
        print(evaluate(param_3, 10, True))

    def test_best_iter(self):

        test_rounds = [i for i in range(10, 100)]
        decrease_rounds = [np.random.randint(i) for i in test_rounds]

        for test_round, decrease_round in zip(test_rounds, decrease_rounds):

            eval_dicts = self.generate_fake_eval_metrics(test_round, decrease_round, )
            self.early_stopping_round = decrease_round - 1

            if self.early_stopping_round <= 0:
                continue

            validation_strategy = ValidationStrategy(self.role, self.mode,
                                                     early_stopping_rounds=self.early_stopping_round,
                                                     use_first_metric_only=self.use_first_metric_only)

            for idx, eval_res in enumerate(eval_dicts):
                validation_strategy.performance_recorder.update(eval_res)
                check_rs = validation_strategy.check_early_stopping()
                if check_rs:
                    best_perform = validation_strategy.performance_recorder.cur_best_performance
                    self.assertDictEqual(best_perform, eval_dicts[test_round-decrease_round-1])
                    print('best iter checking passed')
                    break

    def test_homo_checking(self):
        try:
            validation_strategy = ValidationStrategy(self.role, mode='homo',
                                                     early_stopping_rounds=1)
        except Exception as e:
            # throwing an error is expected
            print(e)
            print('error detected {}, homo checking passed'.format(e))


if __name__ == '__main__':
    tvs = TestValidationStrategy()
    tvs.setUp()
    tvs.test_use_first_metric_only()
    # tvs.test_early_stopping()
    # tvs.test_best_iter()
    # tvs.test_homo_checking()  # expect checking error !!!