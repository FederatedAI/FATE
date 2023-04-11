import numpy as np
import pandas as pd
from federatedml.statistic.data_overview import with_weight
from federatedml.nn.dataset.base import Dataset
from torch_geometric.data import Data
import torch
from federatedml.util import LOGGER


class GraphDataset(Dataset):

    """
     A Graph Dataset includes feature table, edge table and input_nodes table. The data come from a given csv path, or transform from FATE DTable

     Parameters
     ----------
     id_col, name of the id column in csv, default 'id'
     label_col str, name of label column in csv, if None, will automatically take 'y' or 'label' or 'target' as label
     feature_dtype dtype of feature, supports int, long, float, double
     label_dtype: dtype of label, supports int, long, float, double
     feats_name: name of the node feature csv, default 'feats.csv'
     feats_dataset_col: name of the dataset column indicating to which dataset the node belongs, default 'dataset'
     feats_dataset_train: flag of the train set
     feats_dataset_vali: flag of the validation set
     feats_dataset_test: flag of the test set
     adj_name: name of the adjacent matrix, default 'adj.csv'
     adj_src_col: source node in the adjacent matrix, default 'node1'
     adj_dst_col: destination node in the adjacent matrix, default 'node2'
     """

    def __init__(
            self,
            id_col='id',
            label_col='y',
            feature_dtype='float',
            label_dtype='long',
            feats_name='feats.csv',
            feats_dataset_col='dataset',
            feats_dataset_train='train',
            feats_dataset_vali='vali',
            feats_dataset_test='test',
            adj_name='adj.csv',
            adj_src_col='node1',
            adj_dst_col='node2'):

        super(GraphDataset, self).__init__()
        self.key2idx: dict = {}
        self.f_dtype = self.check_dtype(feature_dtype)
        self.l_dtype = self.check_dtype(label_dtype)
        self.data: Data = Data()
        self.sample_ids = None
        self.input_nodes_train = None
        self.input_nodes_vali = None
        self.input_nodes_test = None
        self.id_col = id_col
        self.label_col = label_col
        self.feats_name = feats_name
        self.feats_dataset_col = feats_dataset_col
        self.feats_dataset_train = feats_dataset_train
        self.feats_dataset_vali = feats_dataset_vali
        self.feats_dataset_test = feats_dataset_test
        self.adj_name = adj_name
        self.adj_src_col = adj_src_col
        self.adj_dst_col = adj_dst_col

    def __len__(self):
        return self.data.num_nodes

    @staticmethod
    def check_dtype(dtype):
        if dtype is not None:
            avail = ['long', 'int', 'float', 'double']
            assert dtype in avail, 'available dtype is {}, but got {}'.format(
                avail, dtype)
            if dtype == 'long':
                return torch.int64
            if dtype == 'int':
                return torch.int32
            if dtype == 'float':
                return torch.float32
            if dtype == 'double':
                return torch.float64
        return dtype

    def __process_feats(self, data_path):
        LOGGER.info("processing feats")
        tmp = pd.read_csv(data_path + "/" + self.feats_name)
        self.input_nodes_train = tmp[tmp[self.feats_dataset_col] == self.feats_dataset_train].index.to_list()
        self.input_nodes_vali = tmp[tmp[self.feats_dataset_col] == self.feats_dataset_vali].index.to_list()
        self.input_nodes_test = tmp[tmp[self.feats_dataset_col] == self.feats_dataset_test].index.to_list()

        self.data.x = torch.tensor(tmp.drop([self.id_col, self.feats_dataset_col,
                                   self.label_col], axis=1).to_numpy(), dtype=self.f_dtype)
        self.data.y = torch.tensor(tmp[self.label_col], dtype=self.l_dtype)

    def __process_adj(self, data_path):
        LOGGER.info("processing edges")
        tmp = pd.read_csv(data_path + "/" + self.adj_name)
        self.data.edge_index = torch.tensor(tmp[[self.adj_src_col, self.adj_dst_col]].T.to_numpy(), dtype=torch.long)
        if len(tmp.columns) > 2:
            self.data.edge_attr = torch.tensor(
                tmp.drop([self.adj_src_col, self.adj_dst_col], axis=1).to_numpy(), dtype=torch.float)

    def load(self, data_path):
        LOGGER.info("Loading graph data...")
        self.__process_feats(data_path)
        self.__process_adj(data_path)
        # Assign each node its global node index:
        self.data.n_id = torch.arange(self.data.num_nodes)

    def get_sample_ids(self):
        return self.sample_ids
