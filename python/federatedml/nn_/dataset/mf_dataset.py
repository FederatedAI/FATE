from federatedml.nn_.dataset.base import Dataset
import torch as t
import pandas as pd


class MFDataset(Dataset):

    def __init__(self):
        super(MFDataset, self).__init__()
        self.user_idx = None
        self.item_idx = None
        self.label = None

    def __getitem__(self, item):
        return (self.user_idx[item], self.item_idx[item]), self.label[item]

    def __len__(self):
        return len(self.label)

    def load(self, file_path):
        df = pd.read_csv(file_path)
        self.user_idx = t.Tensor(df['uid']).type(t.int64)
        self.item_idx = t.Tensor(df['mid']).type(t.int64)
        self.label = t.Tensor(df['label']).type(t.float)


if __name__ == '__main__':
    ds = MFDataset()
    ds.load('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/movielens_host_0.csv')