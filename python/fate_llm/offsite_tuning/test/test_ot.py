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


import sys
from datetime import datetime
from fate_llm.offsite_tuning.trainer import OffsiteTuningTrainerClient, OffsiteTuningTrainerServer, TrainingArguments, FedArguments
from fate_llm.model_zoo.offsite_tuning.gpt2 import GPT2LMHeadMainModel, GPT2LMHeadSubModel
from transformers import DataCollatorForSeq2Seq


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


guest = ("guest", "10000")
arbiter = ("arbiter", "9999")
host = ("host", "9998")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation.backends.standalone import StandaloneFederation
    import logging

    # prepare log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # init fate context
    computing = CSession(data_dir="./cession_dir")
    return Context(computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, arbiter, host]))


if __name__ == "__main__":

    party = sys.argv[1]
    import torch as t
    from fate_llm.dataset.qa_dataset import QaDataset

    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

    set_seed(42)

    if party == "guest" or party == "host":
        from fate_llm.dataset.qa_dataset import tokenize_qa_dataset
        from transformers import AutoTokenizer, AutoModel
        tokenizer_name_or_path = 'gpt2'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        if party == "guest":
            ctx = create_ctx(guest, get_current_datetime_str())
        elif party == "host":
            ctx = create_ctx(host, get_current_datetime_str())

        if 'llama' in tokenizer_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, unk_token="<unk>",  bos_token="<s>", eos_token="</s>", add_eos_token=True)   
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if 'gpt2' in tokenizer_name_or_path:
            tokenizer.pad_token = tokenizer.eos_token

        ds = QaDataset(tokenizer_name_or_path=tokenizer_name_or_path, select_num=100)
        ds.load('./sciq/')


        train_args = TrainingArguments(
            per_device_train_batch_size=1,
            learning_rate=5e-5,
            disable_tqdm=False,
            num_train_epochs=2,
            logging_steps=10,
            logging_strategy='steps'
        )

        model = GPT2LMHeadSubModel(
            model_name_or_path=tokenizer_name_or_path,
            emulator_layer_num=2,
            adapter_top_layer_num=2,
            adapter_bottom_layer_num=2
        )

        trainer = OffsiteTuningTrainerClient(
            ctx=ctx,
            model=model,
            training_args=train_args,
            train_set=ds,
            fed_args=FedArguments(),
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            aggregate_model=True
        )
        print('start training')
        trainer.train()

    elif party == "arbiter":
        ctx = create_ctx(arbiter, get_current_datetime_str())

        model = GPT2LMHeadMainModel(
            model_name_or_path='gpt2',
            emulator_layer_num=2,
            adapter_top_layer_num=2,
            adapter_bottom_layer_num=2
        )

        trainer = OffsiteTuningTrainerServer(
            ctx=ctx,
            model=model,
            aggregate_model=True
        )
        print('start training')
        trainer.train()
