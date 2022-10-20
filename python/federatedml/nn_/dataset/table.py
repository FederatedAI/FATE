import numpy as np
import pandas as pd
from federatedml.nn_.dataset.base import Dataset
from federatedml.util import LOGGER


class TableDataset(Dataset):

    def __init__(self, label_col=None, feature_dtype='float', label_dtype='float', label_shape=None,
                 flatten_label=False):
        super(TableDataset, self).__init__()
        self.features: np.ndarray = None
        self.label: np.ndarray = None
        self.origin_table: pd.DataFrame = pd.DataFrame()
        self.label = label_col
        self.f_dtype = self.check_dtype(feature_dtype)
        self.l_dtype = self.check_dtype(label_dtype)
        if label_shape is not None:
            assert isinstance(label_shape, tuple)
        self.label_shape = label_shape
        self.flatten_label = flatten_label

        if self.label is not None:
            assert isinstance(self.label, str) or isinstance(self.label, int),\
                'label columns parameter must be a str or an int'

    @staticmethod
    def check_dtype(dtype):
        if dtype is not None:
            avail = ['long', 'int', 'float', 'double']
            assert dtype in avail, 'available dtype is {}, but got {}'.format(avail, dtype)
            if dtype == 'long':
                return np.int64
            if dtype == 'int':
                return np.int32
            if dtype == 'float':
                return np.float32
            if dtype == 'double':
                return np.float64
        return dtype

    def __getitem__(self, item):
        return self.features[item], self.label[item]

    def __len__(self):
        return len(self.origin_table)

    def load(self, file_path):

        if isinstance(file_path, str):
            self.origin_table = pd.read_csv(file_path)
        elif isinstance(file_path, pd.DataFrame):
            self.origin_table = file_path
        else:
            # if is FATE DTable, collect data and transform to array format
            LOGGER.info('collecting FATE DTable')
            data_inst = file_path
            header = data_inst.schema["header"]
            LOGGER.debug('input dtable header is {}'.format(header))
            data = list(data_inst.collect())
            data_keys = [key for (key, val) in data]
            data_keys_map = dict(zip(sorted(data_keys), range(len(data_keys))))

            keys = [None for idx in range(len(data_keys))]
            x_ = [None for idx in range(len(data_keys))]
            y_ = [None for idx in range(len(data_keys))]

            for (key, inst) in data:
                idx = data_keys_map[key]
                keys[idx] = key
                x_[idx] = inst.features
                y_[idx] = inst.label
            x_ = np.asarray(x_)
            y_ = np.asarray(y_)
            df = pd.DataFrame(x_)
            df.columns = header
            df['label'] = y_
            df['id'] = data_keys
            self.origin_table = df

        label_col_candidates = ['y', 'label', 'target']

        # automatically set id columns
        id_col_candidates = ['id', 'sid']
        for id_col in id_col_candidates:
            if id_col in self.origin_table:
                self.set_sample_ids(self.origin_table[id_col].values.tolist())
                self.origin_table = self.origin_table.drop(columns=[id_col])
                break

        # infer column name
        label = self.label
        if label is None:
            for i in label_col_candidates:
                if i in self.origin_table:
                    label = i
                    break
            if label is None:
                raise ValueError('label default setting is "auto", but found no "y"/"label"/"target" in input'
                                 'table')

        else:
            if label not in self.origin_table:
                raise ValueError('label column {} not found in input table'.format(label))

        self.features = self.origin_table.drop(columns=[label]).values
        if self.f_dtype:
            self.features = self.features.astype(self.f_dtype)

        self.label = self.origin_table[label].values
        if self.l_dtype:
            self.label = self.label.astype(self.l_dtype)

        if self.label_shape:
            self.label = self.label.reshape(self.label_shape)
        else:
            self.label = self.label.reshape((len(self.features), -1))

        if self.flatten_label:
            self.label = self.label.flatten()

        assert self.features.shape[1] >= 1, 'feature number is less than 1'
