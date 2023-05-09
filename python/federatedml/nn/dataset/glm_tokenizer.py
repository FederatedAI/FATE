from federatedml.nn.dataset.base import Dataset
import json
import pandas as pd
import torch
from transformers import AutoTokenizer


PROMPT_TEMPLATE = "{prompt}"


class GLMTokenizerDataset(Dataset):
    def __init__(self, truncation=True, text_max_length=256,
                 tokenizer_name_or_path=None,
                 padding=True, padding_side="right", pad_token=None,
                 trust_remote_code=True,
                 prompt_template=None,
                 prompt_column="content",
                 response_column="summary"
                 ):

        super(GLMTokenizerDataset, self).__init__()
        self.label = None
        self.tokenizer = None
        self.padding = padding
        self.truncation = truncation
        self.max_length = text_max_length
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = padding_side
        if pad_token is not None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})

        self.prompt_template = prompt_template if prompt_template else PROMPT_TEMPLATE
        self.prompt_column = prompt_column
        self.response_column = response_column
        self._data = None

    def load(self, file_path):
        df = pd.read_json(file_path, lines=True)
        self._data = df.apply(self._process_data, axis=1)

    def _process_data(self, line):
        _prompt = line[self.prompt_column]
        _response = line[self.response_column]

        prompt = self.prompt_template.format_map(dict(prompt=_prompt))
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(_response, add_special_tokens=False)

        if len(prompt_ids) > self.max_length - 1:
            prompt_ids = prompt_ids[: self.max_length - 1]
        if len(target_ids) > self.max_length - 2:
            target_ids = target_ids[: self.max_length - 2]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(prompt_ids, target_ids)

        seq_length = input_ids.index(self.tokenizer.bos_token_id)
        labels = [-100] * seq_length + input_ids[seq_length:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.tokenizer.__repr__()
