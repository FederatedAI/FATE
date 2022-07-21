from federatedml.param.base_param import BaseParam


class TensorExampleParam(BaseParam):
    def __init__(self, seed=None, partition=1, data_num=100, feature_num=10):
        self.seed = seed
        self.partition = partition
        self.data_num = data_num
        self.feature_num = feature_num

    def check(self):
        if self.seed is not None and type(self.seed).__name__ != "int":
            raise ValueError("random seed should be None or integers")

        if type(self.partition).__name__ != "int" or self.partition < 1:
            raise ValueError("partition should be an integer large than 0")

        if type(self.data_num).__name__ != "int" or self.data_num < 1:
            raise ValueError("data_num should be an integer large than 0")
