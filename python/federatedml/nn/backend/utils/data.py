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
        cached_id = str(id(data_path_or_dtable))

    if cached_id in dataset_cache:
        LOGGER.debug('use cached dataset, cached id {}'.format(cached_id))
        return dataset_cache[cached_id]

    if dataset_name is None or dataset_name == '':
        # automatically match default dataset
        LOGGER.info('dataset is not specified, use auto inference')

        for ds_class in [TableDataset, ImageDataset]:
            dataset_inst = try_dataset_class(
                ds_class, data_path_or_dtable, param=param)
            if dataset_inst is not None:
                break
        if dataset_inst is None:
            raise ValueError(
                'cannot find default dataset that can successfully load data from path {}, '
                'please check the warning message for error details'. format(data_path_or_dtable))
    else:
        # load specified dataset
        dataset_class = get_dataset_class(dataset_name)
        dataset_inst = dataset_class(**param)
        dataset_inst.load(data_path_or_dtable)

    dataset_cache[cached_id] = dataset_inst

    return dataset_inst


def get_ret_predict_table(id_table, pred_table, classes, partitions, computing_session):

    id_dtable = computing_session.parallelize(
        id_table, partition=partitions, include_key=True)
    pred_dtable = computing_session.parallelize(
        pred_table, partition=partitions, include_key=True)

    return id_dtable, pred_dtable


def add_match_id(id_table: list, dataset_inst: TableDataset):
    assert isinstance(dataset_inst, TableDataset), 'when using match id your dataset must be a Table Dataset'
    for id_inst in id_table:
        id_inst[1].inst_id = dataset_inst.match_ids[id_inst[0]]
