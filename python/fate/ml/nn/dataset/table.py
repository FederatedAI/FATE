import numpy as np
import pandas as pd
from fate.arch.dataframe import DataFrame
from fate.ml.nn.dataset.base import Dataset
import logging
import torch as t


logger = logging.getLogger(__name__)


class TableDataset(Dataset):

    """
    A Table Dataset, load data from a give csv path, or transform FATE DTable

    Parameters
    ----------
    label_col str, name of label column in csv, if None, will automatically take 'y' or 'label' or 'target' as label
    match_id_col str, name of match id column in csv, if None, will automatically take 'id' or 'sid' as match id
    sample_id_col str, name of sample id column in csv, if None, will automatically generate sample id
    feature_dtype str, dtype of features, available: 'long', 'int', 'float', 'double'
    label_dtype str, dtype of label, available: 'long', 'int', 'float', 'double'
    label_shape tuple or list, shape of label, if None, will automatically infer from data
    flatten_label bool, whether to flatten label, if True, will flatten label to 1-d array
    to_tensor bool, whether to transform data to pytorch tensor, if True, will transform data to tensor
    return_dict bool, whether to return a dict in the format of {'x': xxx, 'label': xxx} if True, will return a dict, else will return a tuple
    """

    def __init__(
        self,
        label_col=None,
        match_id_col=None,
        sample_id_col=None,
        feature_dtype="float",
        label_dtype="float",
        label_shape=None,
        flatten_label=False,
        to_tensor=True,
        return_dict=False,
    ):
        super(TableDataset, self).__init__()
        self.features: np.ndarray = None
        self.label: np.ndarray = None
        self.sample_weights: np.ndarray = None
        self.origin_table: pd.DataFrame = pd.DataFrame()
        self.label_col = label_col
        self.match_id_col = match_id_col
        self.sample_id_col = sample_id_col
        self.f_dtype = self.check_dtype(feature_dtype)
        self.l_dtype = self.check_dtype(label_dtype)
        self.to_tensor = to_tensor
        self.return_dict = return_dict
        if label_shape is not None:
            assert isinstance(label_shape, tuple) or isinstance(label_shape, list), "label shape is {}".format(
                label_shape
            )
        self.label_shape = label_shape
        self.flatten_label = flatten_label

        # sample ids, match ids
        self.sample_ids = None
        self.match_ids = None

        if self.label_col is not None:
            assert isinstance(self.label_col, str) or isinstance(
                self.label_col, int
            ), "label columns parameter must be a str or an int"

    @staticmethod
    def check_dtype(dtype):
        if dtype is not None:
            avail = ["long", "int", "float", "double"]
            assert dtype in avail, "available dtype is {}, but got {}".format(avail, dtype)
            if dtype == "long":
                return np.int64
            if dtype == "int":
                return np.int32
            if dtype == "float":
                return np.float32
            if dtype == "double":
                return np.float64
        return dtype

    def __getitem__(self, item):
        if self.label is not None:
            feat = self.features[item]
            label = self.label[item]
            if self.to_tensor:
                feat = t.tensor(feat)
                label = t.tensor(label)
            if self.return_dict:
                return {"x": feat, "label": label}
            else:
                return feat, label
        else:
            feat = self.features[item]
            if self.to_tensor:
                feat = t.tensor(feat)
            if self.return_dict:
                return {"x": feat}
            else:
                return feat

    def __len__(self):
        return len(self.features)

    def load(self, data_or_path):
        if isinstance(data_or_path, str):
            self.origin_table = pd.read_csv(data_or_path)
            # if is FATE DTable, collect data and transform to array format
            label_col_candidates = ["y", "label", "target"]
            # automatically set id columns
            if self.match_id_col is not None:
                if self.match_id_col not in self.origin_table:
                    raise ValueError("match id column {} not found".format(self.match_id_col))
                else:
                    self.match_ids = self.origin_table[[self.match_id_col]]
                    self.origin_table = self.origin_table.drop(columns=[self.match_id_col])
            else:
                match_id_col_cadidaites = ["id", "sid"]
                for id_col in match_id_col_cadidaites:
                    if id_col in self.origin_table:
                        self.match_ids = self.origin_table[[id_col]]
                        self.origin_table = self.origin_table.drop(columns=[id_col])
                        break
                if self.match_ids is None:
                    logger.info("match id column not found, no match id will be set")

            # generate sample ids
            if self.sample_id_col is not None:
                if self.sample_id_col not in self.origin_table:
                    raise ValueError("sample id column {} not found".format(self.sample_id_col))
                self.sample_ids = self.origin_table[[self.sample_id_col]]
                self.origin_table = self.origin_table.drop(columns=[self.sample_id_col])
            else:
                self.sample_ids = pd.DataFrame()
                self.sample_ids["sample_id"] = range(len(self.origin_table))
                logger.info(
                    "sample id column not found, generate sample id from 0 to {}".format(len(self.origin_table))
                )

            # infer column name
            label = self.label_col
            if label is None:
                for i in label_col_candidates:
                    if i in self.origin_table:
                        label = i
                        logger.info('use "{}" as label column'.format(label))
                        break
                if label is None:
                    logger.info('found no "y"/"label"/"target" in input table, no label will be set')
            else:
                if label not in self.origin_table:
                    raise ValueError("label column {} not found in input table".format(label))

            self.label = self.origin_table[[label]].values
            self.origin_table = self.origin_table.drop(columns=[label])
            self.features = self.origin_table.values

        elif isinstance(data_or_path, DataFrame):
            schema = data_or_path.schema
            sample_id = schema.sample_id_name
            match_id = schema.match_id_name
            label = schema.label_name
            if label is None:
                logger.info("label column is None, not provided in the uploaded data")
            pd_df = data_or_path.as_pd_df()
            if label is None:
                labels = None
                features = pd_df.drop([sample_id, match_id], axis=1)
            else:
                labels = pd_df[[label]]
                features = pd_df.drop([sample_id, match_id, label], axis=1)
                self.label = labels.values
            sample_ids = pd_df[[sample_id]]
            match_ids = pd_df[[match_id]]
            self.sample_ids = sample_ids
            self.match_ids = match_ids
            self.features = features.values

        if self.label is not None:
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

        if self.f_dtype:
            self.features = self.features.astype(self.f_dtype)

    def get_classes(self):
        if self.label is not None:
            return np.unique(self.label).tolist()
        else:
            raise ValueError("no label found, please check if self.label is set")

    def get_sample_ids(self) -> np.ndarray:
        return self.sample_ids.values

    def get_match_ids(self) -> np.ndarray:
        return self.match_ids.values

    def get_sample_id_name(self) -> str:
        if self.sample_ids is not None and isinstance(self.sample_ids, pd.DataFrame):
            return self.sample_ids.columns[0]
        else:
            raise ValueError("Cannot get sample id name")

    def get_match_id_name(self) -> str:
        if self.match_ids is not None and isinstance(self.match_ids, pd.DataFrame):
            return self.match_ids.columns[0]
        else:
            raise ValueError("Cannot get match id name")

    def has_label(self) -> bool:
        return self.label is not None
