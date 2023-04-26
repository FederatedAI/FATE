from federatedml.nn.dataset.base import Dataset
import json
import torch
from transformers import AutoTokenizer


PROMPT_TEMPLATE = "{prompt}"


class PromptTokenizerDataset(Dataset):
    def __init__(self, truncation=True, text_max_length=256,
                 tokenizer_name_or_path=None,
                 padding=True, padding_side="right", pad_token=None,
                 trust_remote_code=True,
                 prompt_template=None,
                 prompt_column="content",
                 response_column="summary"
                 ):

        super(PromptTokenizerDataset, self).__init__()
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
        self._data = []

    def load(self, file_path):
        tokenizer = self.tokenizer

        _dataset = []
        with open(file_path, "r") as fin:
            for line in fin:
                _dataset.append(json.loads(line.strip()))

        for line in _dataset:
            _prompt = line[self.prompt_column]
            _response = line[self.response_column]

            prompt = self.prompt_template.format_map(dict(prompt=_prompt))
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(_response, add_special_tokens=False)

            if len(prompt_ids) > self.max_length - 1:
                prompt_ids = prompt_ids[: self.max_length - 1]
            if len(target_ids) > self.max_length - 2:
                target_ids = target_ids[: self.max_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, target_ids)
            # input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
            # if len(input_ids) > self.max_length * 2:
            #     input_ids = input_ids[:self.max_length * 2]

            # seq_length = len(prompt_ids)
            seq_length = input_ids.index(tokenizer.bos_token_id)

            self._data.append({"input_ids": input_ids, "seq_length": seq_length})

    def collate_fn(self, batch_data: list):
        # max_len = self.max_length * 2
        max_len = max([len(_data["input_ids"]) for _data in batch_data]) + 1
        input_ids_list = list()
        labels_list = list()
        for _data in batch_data:
            input_ids = _data["input_ids"]
            seq_length = _data["seq_length"]

            to_pad_len = max_len - len(input_ids)
            labels = [-100] * seq_length + input_ids[seq_length:] + to_pad_len * [-100]
            input_ids = input_ids + [self.tokenizer.pad_token_id] * to_pad_len

            labels_list.append(torch.LongTensor(labels))
            input_ids_list.append(torch.LongTensor(input_ids))

        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }, labels

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self.tokenizer.__repr__()
