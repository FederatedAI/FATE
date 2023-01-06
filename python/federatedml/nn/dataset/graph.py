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
            label_dtype='long',
            label_shape=None,
            flatten_label=False):

        super(GraphDataset, self).__init__()
        self.key2idx: dict = {}
        self.f_dtype = self.check_dtype(feature_dtype)
        self.l_dtype = self.check_dtype(label_dtype)
        self.data: Data = Data()

        # ids, match ids is for FATE match id system
        self.sample_ids = None

    def __len__(self):
        return self.input_cnt

    # def __getitem__(self, item):
    #     return self.x[item]

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


    def __process_feats(self, data_feats):
        LOGGER.info("processing feats")
        cnt = data_feats.count()
        if cnt <= 0:
            raise ValueError("empty data")

        _, one_data = data_feats.first()
        x_shape = one_data.features.shape
        x = np.zeros((cnt, *x_shape))
        y = np.zeros(cnt)
        index = 0
        for k, inst in data_feats.collect():
            x[index] = inst.features
            y[index] = inst.label
            self.key2idx[k] = index
            index += 1

        num_label = len(set(y))       
        if num_label == 2:
            self.output_shape = 1
        elif num_label > 2:
            self.output_shape = (num_label,)
        else:
            raise ValueError(f"num_label is {num_label}")

        self.data.x = torch.tensor(x, dtype=self.f_dtype)
        self.data.y = torch.tensor(y, dtype=self.l_dtype)


    def __process_adj(self, data_adj):
        LOGGER.info("processing edges")
        edges = data_adj.collect()
        srcs = []
        dsts = []
        vals = []
        for _, v in edges:
            if len(v.features) == 3:
                src, dst, val = int(v.features[0]), int(v.features[1]), float(v.features[2])
                vals.append(val)
            elif len(v.features) == 2:
                src, dst = int(v.features[0]), int(v.features[1])
            else:
                raise "Incorrect adj format"
            srcs.append(src)
            dsts.append(dst)
        
        self.data.edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        if len(vals) > 0:                
            self.data.edge_attr = torch.tensor(vals, dtype=torch.float)

    def __process_input_nodes(self, data_input_nodes):
        LOGGER.info("processing input nodes")
        tmp = np.array([False] * len(self.key2idx))
        self.sample_ids = []
        for k, _ in data_input_nodes.collect():
            tmp[self.key2idx[k]] = True
            self.sample_ids.append(k)
        self.input_cnt = data_input_nodes.count()
        self.input_nodes = torch.tensor(tmp, dtype=torch.bool)

    def load(self, data_inst):
        LOGGER.info("Loading graph data...")
        data_feats, data_adj, input_nodes = data_inst

        if isinstance(data_feats, str):
            self.origin_table['feats'] = pd.read_csv(data_feats)
            self.origin_table['adj'] = pd.read_csv(data_adj)
            self.origin_table['input_nodes'] = pd.read_csv(input_nodes)
    
        else:
            # if is FATE DTable, collect data and transform to array format
            LOGGER.info('collecting FATE DTable')

            self.__process_feats(data_feats)
            self.__process_adj(data_adj)
            self.__process_input_nodes(input_nodes)
            # Assign each node its global node index:
            self.data.n_id = torch.arange(self.data.num_nodes)

    def get_sample_ids(self):
        return self.sample_ids

