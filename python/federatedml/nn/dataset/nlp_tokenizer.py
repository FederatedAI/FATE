from federatedml.nn.dataset.base import Dataset
import pandas as pd
import torch as t
from transformers import AutoTokenizer
import os
import numpy as np

# avoid tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenizerDataset(Dataset):
    """
    A Dataset for some basic NLP Tasks, this dataset will automatically transform raw text into word indices
    using AutoTokenizer from transformers library,

    Parameters
    ----------
    truncation bool, truncate word sequence to 'text_max_length'
    text_max_length int, max length of word sequences
    tokenizer_name_or_path str, name of bert tokenizer(see transformers official for details) or path to local
                                transformer tokenizer folder
    return_label bool, return label or not, this option is for host dataset, when running hetero-NN
    padding bool, whether to pad the word sequence to 'text_max_length'
    padding_side str, 'left' or 'right', where to pad the word sequence
    pad_token str, pad token, use this str as pad token, if None, use tokenizer.pad_token
    return_input_ids bool, whether to return input_ids or not, if False, return word_idx['input_ids']
    """

    def __init__(self, truncation=True, text_max_length=128,
                 tokenizer_name_or_path="bert-base-uncased",
                 return_label=True, padding=True, padding_side="right", pad_token=None,
                 return_input_ids=True
                 ):

        super(TokenizerDataset, self).__init__()
        self.text = None
        self.word_idx = None
        self.label = None
        self.tokenizer = None
        self.sample_ids = None
        self.padding = padding
        self.truncation = truncation
        self.max_length = text_max_length
        self.with_label = return_label
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        self.tokenizer.padding_side = padding_side
        self.return_input_ids = return_input_ids
        if pad_token is not None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})

    def load(self, file_path):

        tokenizer = self.tokenizer
        self.text = pd.read_csv(file_path)
        text_list = list(self.text.text)

        self.word_idx = tokenizer(
            text_list,
            padding=self.padding,
            return_tensors='pt',
            truncation=self.truncation,
            max_length=self.max_length)

        if self.return_input_ids:
            self.word_idx = self.word_idx['input_ids']

        if self.with_label:
            self.label = t.Tensor(self.text.label).detach().numpy()
            self.label = self.label.reshape((len(self.text), -1))

        if 'id' in self.text:
            self.sample_ids = self.text['id'].values.tolist()

    def get_classes(self):
        return np.unique(self.label).tolist()

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def get_sample_ids(self):
        return self.sample_ids

    def __getitem__(self, item):

        ret = None
        if self.return_input_ids:
            ret = self.word_idx[item]
        else:
            ret = {k: v[item] for k, v in self.word_idx.items()}

        if self.with_label:
            return ret, self.label[item]
        else:
            return ret

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return self.tokenizer.__repr__()
