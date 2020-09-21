import abc
from abc import ABC


class BasicAlgorithms(ABC):

    @abc.abstractmethod
    def get_sample_weights(self, *args):
        pass

    @abc.abstractmethod
    def fit(self, *args):
        pass

    @abc.abstractmethod
    def predict(self, *args):
        pass

    @abc.abstractmethod
    def get_model(self, *args):
        pass

    @abc.abstractmethod
    def load_model(self, *args):
        pass

