import abc
import importlib

import numpy as np
from torch.utils.data import Dataset as Dataset_

from federatedml.nn.backend.utils.common import ML_PATH


class Dataset(Dataset_):

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self._sample_ids = None
        self._type = 'local'  # train/predict

    @property
    def dataset_type(self):
        if not hasattr(self, '_type'):
            raise AttributeError('type variable not exists, call __init__ of super class')
        return self._type

    @dataset_type.setter
    def dataset_type(self, val):
        self._type = val

    @property
    def sample_ids(self):
        if not hasattr(self, '_sample_ids'):
            raise AttributeError('sample_ids variable not exists, call __init__ of super class')
        return self._sample_ids

    @sample_ids.setter
    def sample_ids(self, val):
        self._sample_ids = val

    def has_dataset_type(self):
        return self.dataset_type

    def set_type(self, _type):
        self.dataset_type = _type

    def get_type(self):
        return self.dataset_type

    def set_sample_ids(self, ids):
        self.sample_ids = ids

    def get_sample_ids(self):
        if self.sample_ids is not None:
            return list(self.sample_ids)
        else:
            return None

    def has_sample_ids(self):
        return self.sample_ids is not None

    def generate_sample_ids(self, prefix: str = None):
        if prefix is not None:
            assert isinstance(prefix, str), 'prefix must be a str, but got {}'.format(prefix)
        else:
            prefix = self._type
        generated_ids = []
        for i in range(0, self.__len__()):
            generated_ids.append(prefix + '_' + str(i))
        self._sample_ids = generated_ids

    """
    User to implement functions
    """

    @abc.abstractmethod
    def load(self, file_path):
        raise NotImplementedError('You must implement load function so that Client class can pass file-path to this '
                                  'class')

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_classes(self):
        pass


class ShuffleWrapDataset(Dataset_):

    def __init__(self, dataset: Dataset, shuffle_seed=100):
        super(ShuffleWrapDataset, self).__init__()
        self.ds = dataset
        ids = self.ds.get_sample_ids()
        sort_idx = np.argsort(np.array(ids))
        assert isinstance(dataset, Dataset)
        self.idx = sort_idx
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
            self.shuffled_idx = np.copy(self.idx)
            np.random.shuffle(self.shuffled_idx)
        else:
            self.shuffled_idx = np.copy(self.idx)
        self.idx_map = {k: v for k, v in zip(self.idx, self.shuffled_idx)}

    def __getitem__(self, item):
        return self.ds[self.idx_map[self.idx[item]]]

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.ds.__repr__()

    def has_sample_ids(self):
        return self.ds.has_sample_ids()

    def set_shuffled_idx(self, idx_map: dict):
        self.shuffled_idx = np.array(list(idx_map.values()))
        self.idx_map = idx_map

    def get_sample_ids(self):
        ids = self.ds.get_sample_ids()
        return np.array(ids)[self.shuffled_idx].tolist()

    def get_classes(self):
        return self.ds.get_classes()


def get_dataset_class(dataset_module_name: str):
    if dataset_module_name.endswith('.py'):
        dataset_module_name = dataset_module_name.replace('.py', '')
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
