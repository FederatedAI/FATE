import numpy as np
from torch.utils.data import Dataset as torchDataset
from federatedml.util import LOGGER
from federatedml.nn.dataset.base import Dataset, get_dataset_class
from federatedml.nn.dataset.image import ImageDataset
from federatedml.nn.dataset.table import TableDataset


def try_dataset_class(dataset_class, path, param):
    # try default dataset
    try:
        dataset_inst: Dataset = dataset_class(**param)
        dataset_inst.load(path)
        return dataset_inst
    except Exception as e:
        LOGGER.warning('try to load dataset failed, exception :{}'.format(e))
        return None


def load_dataset(dataset_name, data_path_or_dtable, param, dataset_cache: dict):
    # load dataset class
    if isinstance(data_path_or_dtable, str):
        cached_id = data_path_or_dtable
    else:
        cached_id = id(data_path_or_dtable)

    if cached_id in dataset_cache:
        LOGGER.debug('use cached dataset, cached id {}'.format(cached_id))
        return dataset_cache[cached_id]

    if dataset_name is None or dataset_name == '':
        # automatically match default dataset
        LOGGER.info('dataset is not specified, use auto inference')

        for ds_class in [TableDataset, ImageDataset]:
            dataset_inst = try_dataset_class(ds_class, data_path_or_dtable, param=param)
            if dataset_inst is not None:
                break
        if dataset_inst is None:
            raise ValueError('cannot find default dataset that can successfully load data from path {}, '
                             'please check the warning message for error details'.
                             format(data_path_or_dtable))
    else:
        # load specified dataset
        dataset_class = get_dataset_class(dataset_name)
        dataset_inst = dataset_class(**param)
        dataset_inst.load(data_path_or_dtable)

    if isinstance(data_path_or_dtable, str):
        dataset_cache[data_path_or_dtable] = dataset_inst
    else:
        dataset_cache[id(data_path_or_dtable)] = dataset_inst

    return dataset_inst
