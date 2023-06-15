from torch.utils.data import Dataset as Dataset_
from federatedml.nn.backend.utils.common import ML_PATH, LLM_PATH
import importlib
import abc
import numpy as np


class Dataset(Dataset_):

    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self._type = 'local'  # train/predict
        self._check = False
        self._generated_ids = None
        self.training = True

    @property
    def dataset_type(self):
        if not hasattr(self, '_type'):
            raise AttributeError(
                'type variable not exists, call __init__ of super class')
        return self._type

    @dataset_type.setter
    def dataset_type(self, val):
        self._type = val

    def has_dataset_type(self):
        return self.dataset_type

    def set_type(self, _type):
        self.dataset_type = _type

    def get_type(self):
        return self.dataset_type

    def has_sample_ids(self):

        # if not implement get_sample_ids, return False
        try:
            sample_ids = self.get_sample_ids()
        except NotImplementedError as e:
            return False
        except BaseException as e:
            raise e

        if sample_ids is None:
            return False
        else:
            if not self._check:
                assert isinstance(
                    sample_ids, list), 'get_sample_ids() must return a list contains str or integer'
                for id_ in sample_ids:
                    if (not isinstance(id_, str)) and (not isinstance(id_, int)):
                        raise RuntimeError(
                            'get_sample_ids() must return a list contains str or integer: got id of type {}:{}'.format(
                                id_, type(id_)))
                assert len(sample_ids) == len(
                    self), 'sample id len:{} != dataset length:{}'.format(len(sample_ids), len(self))
                self._check = True
            return True

    def init_sid_and_getfunc(self, prefix: str = None):
        if prefix is not None:
            assert isinstance(
                prefix, str), 'prefix must be a str, but got {}'.format(prefix)
        else:
            prefix = self._type
        generated_ids = []
        for i in range(0, self.__len__()):
            generated_ids.append(prefix + '_' + str(i))
        self._generated_ids = generated_ids

        def get_func():
            return self._generated_ids
        self.get_sample_ids = get_func

    """
    Functions for users
    """

    def train(self, ):
        self.training = True

    def eval(self, ):
        self.training = False

    # Function to implemented

    @abc.abstractmethod
    def load(self, file_path):
        raise NotImplementedError(
            'You must implement load function so that Client can pass file-path to this '
            'class')

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_classes(self):
        raise NotImplementedError()

    def get_sample_ids(self):
        raise NotImplementedError()


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

    def train(self, ):
        self.ds.train()

    def eval(self, ):
        self.ds.eval()

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
    try:
        ds_modules = importlib.import_module(
            '{}.dataset.{}'.format(
                ML_PATH, dataset_module_name)
        )
    except BaseException:
        ds_modules = importlib.import_module(
            '{}.dataset.{}'.format(
                LLM_PATH, dataset_module_name)
        )
    try:
        ds = []
        for k, v in ds_modules.__dict__.items():
            if isinstance(v, type):
                if issubclass(v, Dataset) and v is not Dataset:
                    ds.append(v)
        if len(ds) == 0:
            raise ValueError('Did not find any class in {}.py that is the subclass of Dataset class'.
                             format(dataset_module_name))
        else:
            return ds[-1]  # return the last defined class
    except ValueError as e:
        raise e
