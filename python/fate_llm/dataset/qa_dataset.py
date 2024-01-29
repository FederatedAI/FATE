#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
import torch as t
import os
from torch.utils.data import Dataset


"""
These Data pre-processing templates are from https://github.com/mit-han-lab/offsite-tuning
"""

class PIQA:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['goal']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        if -1 in examples["label"]:  # test set
            return [""] * len(examples["label"])
        else:
            gt_tuples = [("sol{}".format(label + 1), idx)
                         for idx, label in enumerate(examples['label'])]
            return [examples[k][i] for k, i in gt_tuples]


class SciQ:
    def __init__(self):
        self._template = "{}\nQuestion: {}\nAnswer:"

    def get_context(self, examples):
        sources = examples['support']
        queries = examples['question']
        return [self._template.format(s, q) for s, q in zip(sources, queries)]

    def get_target(self, examples):
        return examples['correct_answer']


class OpenBookQA:
    def get_context(self, examples):
        return examples['question_stem']

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        targets = []
        for choice, answer in zip(choices, answers):
            answer = ord(answer.strip()) - ord('A')
            targets.append(choice['text'][answer])
        return targets


task_dict = {
    "piqa": PIQA(),
    "sciq": SciQ(),
    "openbookqa": OpenBookQA()
}


def tokenize_qa_dataset(dataset_name, tokenizer, save_path, seq_max_len=1000):

    max_len = seq_max_len
    assert dataset_name in ['piqa', 'sciq', 'openbookqa'], "dataset name must be one of ['piqa', 'sciq', 'openbookqa']"
    raw_datasets = load_dataset(dataset_name) 
    task = task_dict[dataset_name]

    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        context = task.get_context(examples)
        target = task.get_target(examples)

        context = tokenizer(context)
        target = tokenizer(target)

        # if context is ending with special token, remove it
        if len(context['input_ids'][0]) > 0 and context['input_ids'][0][-1] in tokenizer.all_special_ids:
            context['input_ids'] = [i[:-1] for i in context['input_ids']]
            context['attention_mask'] = [a[:-1]
                                            for a in context['attention_mask']]

        # if target is starting with special token, remove it
        if len(target['input_ids'][0]) > 0 and target['input_ids'][0][0] in tokenizer.all_special_ids:
            target['input_ids'] = [i[1:] for i in target['input_ids']]
            target['attention_mask'] = [a[1:]
                                        for a in target['attention_mask']]

        out = {}
        out['input_ids'] = [i1 + i2 for i1,
                            i2 in zip(context['input_ids'], target['input_ids'])]
        out['attention_mask'] = [a1 + a2 for a1,
                                    a2 in zip(context['attention_mask'], target['attention_mask'])]

        # set -100 for context tokens
        out["labels"] = [
            [-100] * len(i1) + i2 for i1, i2 in zip(context['input_ids'], target['input_ids'])]

        return out

    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
        num_proc=32,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # pad all instances in lm_datasets to the max length of the dataset
    max_length = -1
    for v in tokenized_datasets.values():
        for x in v:
            max_length = max(max_length, len(x['input_ids']))

    # pad to the multiple of 8
    max_length = (max_length // 8 + 1) * 8

    block_size = max_len
    max_length = min(max_length, block_size)

    def pad_function(examples):
        examples["input_ids"] = [i + [tokenizer.pad_token_id] *
                                    (max_length - len(i)) for i in examples["input_ids"]]
        examples["attention_mask"] = [[1] * len(i) + [0] *
                                        (max_length - len(i)) for i in examples["attention_mask"]]
        examples["labels"] = [i + [-100] *
                                (max_length - len(i)) for i in examples["labels"]]
        # truncate to max_length
        examples["input_ids"] = [i[:max_length] for i in examples["input_ids"]]
        examples["attention_mask"] = [a[:max_length]
                                        for a in examples["attention_mask"]]
        examples["labels"] = [l[:max_length] for l in examples["labels"]]
        return examples


    tokenized_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=32,
            load_from_cache_file=True,
            desc=f"Padding dataset to max length {max_length}",
        )

    tokenized_datasets.save_to_disk(save_path)
    return tokenized_datasets


class QaDataset(Dataset):

    def __init__(self, tokenizer_name_or_path, select_num=None, start_idx=None):
        self.select_num = select_num
        self.start_idx = start_idx
        self.ds = None
        if 'llama' in tokenizer_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, unk_token="<unk>",  bos_token="<s>", eos_token="</s>", add_eos_token=True)   
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if 'gpt2' in tokenizer_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        

    def load(self, path):
        loaded = load_from_disk(path)
        self.ds = loaded['train']
        if self.select_num is not None:
            if self.start_idx is not None:
                self.ds = self.ds.select(range(self.start_idx, min(len(self.ds), self.start_idx + self.select_num)))
            else:
                self.ds = self.ds.select(range(self.select_num))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]
    