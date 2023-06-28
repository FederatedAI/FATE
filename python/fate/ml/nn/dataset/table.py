import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TableDataset(Dataset):

    """
     A Table Dataset, load data from a give csv path, or transform FATE DTable

     Parameters
     ----------
     label_col str, name of label column in csv, if None, will automatically take 'y' or 'label' or 'target' as label
     feature_dtype dtype of feature, supports int, long, float, double
     label_dtype: dtype of label, supports int, long, float, double
     label_shape: list or tuple, the shape of label
     flatten_label: bool, flatten extracted label column or not, default is False
     """

    def __init__(
            self,
            label_col=None,
            feature_dtype='float',
            label_dtype='float',
            label_shape=None,
            flatten_label=False):

        super(TableDataset, self).__init__()
        self.with_label = True
        self.with_sample_weight = False
        self.features: np.ndarray = None
        self.label: np.ndarray = None
        self.sample_weights: np.ndarray = None
        self.origin_table: pd.DataFrame = pd.DataFrame()
        self.label_col = label_col
        self.f_dtype = self.check_dtype(feature_dtype)
        self.l_dtype = self.check_dtype(label_dtype)
        if label_shape is not None:
            assert isinstance(label_shape, tuple) or isinstance(
                label_shape, list), 'label shape is {}'.format(label_shape)
        self.label_shape = label_shape
        self.flatten_label = flatten_label

        # ids, match ids is for FATE match id system
        self.sample_ids = None
        self.match_ids = None

        if self.label_col is not None:
            assert isinstance(self.label_col, str) or isinstance(
                self.label_col, int), 'label columns parameter must be a str or an int'

    @staticmethod
    def check_dtype(dtype):

        if dtype is not None:
            avail = ['long', 'int', 'float', 'double']
            assert dtype in avail, 'available dtype is {}, but got {}'.format(
                avail, dtype)
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

        if self.with_label:
            if self.with_sample_weight and self.training:
                return self.features[item], (self.label[item], self.sample_weights[item])
            else:
                return self.features[item], self.label[item]
        else:
            return self.features[item]

    def __len__(self):
        return len(self.origin_table)

    def load(self, file_path):

        if isinstance(file_path, str):
            self.origin_table = pd.read_csv(file_path)
        elif isinstance(file_path, pd.DataFrame):
            self.origin_table = file_path
        else:
            # if is FATE DTable, collect data and transform to array format
            data_inst = file_path
            self.with_sample_weight = None
            print('collecting FATE DTable, with sample weight is {}'.format(self.with_sample_weight))
            header = data_inst.schema["header"]
            print('input dtable header is {}'.format(header))
            data = list(data_inst.collect())
            data_keys = [key for (key, val) in data]
            data_keys_map = dict(zip(sorted(data_keys), range(len(data_keys))))

            keys = [None for idx in range(len(data_keys))]
            x_ = [None for idx in range(len(data_keys))]
            y_ = [None for idx in range(len(data_keys))]
            match_ids = {}
            sample_weights = [1 for idx in range(len(data_keys))]

            for (key, inst) in data:
                idx = data_keys_map[key]
                keys[idx] = key
                x_[idx] = inst.features
                y_[idx] = inst.label
                match_ids[key] = inst.inst_id
                if self.with_sample_weight:
                    sample_weights[idx] = inst.weight

            x_ = np.asarray(x_)
            y_ = np.asarray(y_)
            df = pd.DataFrame(x_)
            df.columns = header
            df['id'] = sorted(data_keys)
            df['label'] = y_
            # host data has no label, so this columns will all be None
            if df['label'].isna().all():
                df = df.drop(columns=['label'])

            self.origin_table = df
            self.sample_weights = np.array(sample_weights)
            self.match_ids = match_ids

        label_col_candidates = ['y', 'label', 'target']

        # automatically set id columns
        id_col_candidates = ['id', 'sid']
        for id_col in id_col_candidates:
            if id_col in self.origin_table:
                self.sample_ids = self.origin_table[id_col].values.tolist()
                self.origin_table = self.origin_table.drop(columns=[id_col])
                break

        # infer column name
        label = self.label_col
        if label is None:
            for i in label_col_candidates:
                if i in self.origin_table:
                    label = i
                    break
            if label is None:
                self.with_label = False
                print(
                    'label default setting is "auto", but found no "y"/"label"/"target" in input table')
        else:
            if label not in self.origin_table:
                raise ValueError(
                    'label column {} not found in input table'.format(label))

        if self.with_label:
            self.label = self.origin_table[label].values
            self.features = self.origin_table.drop(columns=[label]).values

            if self.l_dtype:
                self.label = self.label.astype(self.l_dtype)

            if self.label_shape:
                self.label = self.label.reshape(self.label_shape)
            else:
                self.label = self.label.reshape((len(self.features), -1))

            if self.flatten_label:
                self.label = self.label.flatten()

        else:
            self.label = None
            self.features = self.origin_table.values

        if self.f_dtype:
            self.features = self.features.astype(self.f_dtype)

    def get_classes(self):
        if self.label is not None:
            return np.unique(self.label).tolist()
        else:
            raise ValueError(
                'no label found, please check if self.label is set')

    def get_sample_ids(self):
        return self.sample_ids

    def get_match_ids(self):
        return self.match_ids
