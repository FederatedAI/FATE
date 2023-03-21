from federatedml.nn.dataset.base import Dataset
import pandas as pd
import torch as t
from transformers import BertTokenizerFast
import os
import numpy as np

# avoid tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenizerDataset(Dataset):
    """
    A Dataset for some basic NLP Tasks, this dataset will automatically transform raw text into word indices
    using BertTokenizer from transformers library,
    see https://huggingface.co/docs/transformers/model_doc/bert?highlight=berttokenizer for details of BertTokenizer

    Parameters
    ----------
    truncation bool, truncate word sequence to 'text_max_length'
    text_max_length int, max length of word sequences
    tokenizer_name_or_path str, name of bert tokenizer(see transformers official for details) or path to local
                                transformer tokenizer folder
    return_label bool, return label or not, this option is for host dataset, when running hetero-NN
    """

    def __init__(self, truncation=True, text_max_length=128,
                 tokenizer_name_or_path="bert-base-uncased",
                 return_label=True):

        super(TokenizerDataset, self).__init__()
        self.text = None
        self.word_idx = None
        self.label = None
        self.tokenizer = None
        self.sample_ids = None
        self.truncation = truncation
        self.max_length = text_max_length
        self.with_label = return_label
        self.tokenizer_name_or_path = tokenizer_name_or_path

    def load(self, file_path):

        tokenizer = BertTokenizerFast.from_pretrained(
            self.tokenizer_name_or_path)
        self.tokenizer = tokenizer
        self.text = pd.read_csv(file_path)
        text_list = list(self.text.text)
        self.word_idx = tokenizer(
            text_list,
            padding=True,
            return_tensors='pt',
            truncation=self.truncation,
            max_length=self.max_length)['input_ids']
        if self.with_label:
            self.label = t.Tensor(self.text.label).detach().numpy()
            self.label = self.label.reshape((len(self.word_idx), -1))
        del tokenizer  # avoid tokenizer parallelism

        if 'id' in self.text:
            self.sample_ids = self.text['id'].values.tolist()

    def get_classes(self):
        return np.unique(self.label).tolist()

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_sample_ids(self):
        return self.sample_ids

    def __getitem__(self, item):
        if self.with_label:
            return self.word_idx[item], self.label[item]
        else:
            return self.word_idx[item]

    def __len__(self):
        return len(self.word_idx)

    def __repr__(self):
        return self.tokenizer.__repr__()
