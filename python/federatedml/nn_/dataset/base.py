from torch.utils.data import Dataset as Dataset_
from federatedml.nn_.backend.util import ML_PATH
from federatedml.util import LOGGER
import importlib
import abc


class Dataset(Dataset_):

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self._sample_ids = None
        self._type = 'local'  # train/predict

    def has_dataset_type(self):
        return self._type is not None

    def set_type(self, _type):
        self._type = _type  # train/predict

    def get_type(self):
        return self._type

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def set_sample_ids(self, ids):
        self._sample_ids = ids

    def get_sample_ids(self):
        return list(self._sample_ids)

    def has_sample_ids(self):
        return self._sample_ids is not None

    def generate_sample_ids(self, prefix: str = None):
        if prefix is not None:
            assert isinstance(prefix, str), 'prefix must be a str, but got {}'.format(prefix)
        else:
            prefix = self._type
        generated_ids = []
        for i in range(0, self.__len__()):
            generated_ids.append(prefix + '_' + str(i))
        self._sample_ids = generated_ids

    @abc.abstractmethod
    def load(self, file_path):
        raise NotImplementedError('You must implement load function so that Client class can pass file-path to this '
                                  'class')


def get_dataset_class(dataset_module_name: str):

    ds_modules = importlib.import_module('{}.dataset.{}'.format(ML_PATH, dataset_module_name))
    try:

        for k, v in ds_modules.__dict__.items():
            if isinstance(v, type):
                if issubclass(v, Dataset) and v is not Dataset:
                    return v
        raise ValueError('Did not find any class in {}.py that is the subclass of Dataset class'.
                         format(dataset_module_name))
    except ValueError as e:
        raise e
