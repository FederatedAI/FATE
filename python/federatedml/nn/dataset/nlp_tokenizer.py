from federatedml.nn.dataset.base import Dataset
import pandas as pd
import torch as t
from federatedml.util import LOGGER
from transformers import BertTokenizerFast
import os

# avoid tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentDataset(Dataset):

    def __init__(self, truncation=True, text_max_length=128):
        super(SentimentDataset, self).__init__()
        self.text = None
        self.word_idx = None
        self.label = None
        self.truncation = truncation
        self.max_length = text_max_length
        self.tokenizer = None
        
    def load(self, file_path):
        tokenizer = BertTokenizerFast.from_pretrained('/data/projects/cwj/standalone_fate_install_1.9.0_release/bert/bert')
        self.tokenizer = tokenizer
        self.text = pd.read_csv(file_path)
        text_list = list(self.text.text)
        LOGGER.debug('tokenizing text')
        self.word_idx = tokenizer(text_list, padding=True, return_tensors='pt', truncation=self.truncation, max_length=self.max_length)['input_ids']
        self.label = t.Tensor(self.text.label)
        del tokenizer # avoid tokenizer parallelism
        self.label = self.label.reshape((len(self.word_idx), -1))
        
    def __getitem__(self, item):
        return self.word_idx[item], self.label[item]

    def __len__(self):
        return len(self.word_idx)

    def __repr__(self):
        return self.tokenizer.__repr__()