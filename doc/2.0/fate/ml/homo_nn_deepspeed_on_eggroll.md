# Running Homo-NN with Deeepspeed on Eggroll

Our latest Homo framework enables accelerated model training using DeepSpeed. In this tutorial, we'll guide you through the process of training a federated GPT-2 model within an Eggroll environment.


## Prepare FATE Context

In FATE-2.0, the running environment, including the party settings (guest, host, arbiter, and their party IDs), is configured using a context object. This object can be created with the create_context function. The following Python code illustrates how to set up the context:

```python
from fate.arch.federation import FederationEngine
from fate.arch.computing import ComputingEngine
from fate.arch.context import create_context

def create_ctx(local_party, federation_session_id, computing_session_id):
    return create_context(
        local_party=local_party,
        parties=[guest, host, arbiter],
        federation_session_id=federation_session_id,
        federation_engine=FederationEngine.OSX,
        federation_conf={"federation.osx.host": "xxx", "federation.osx.port": xxx, "federation.osx.mode": "stream"},
        computing_session_id=computing_session_id,
        computing_engine=ComputingEngine.EGGROLL,
        computing_conf={"computing.eggroll.host": "xxx",
                        "computing.eggroll.port": xxx,
                        "computing.eggroll.options": {'eggroll.session.processors.per.node': 4, 'nodes': 1},
                        "computing.eggroll.config": None,
                        "computing.eggroll.config_options": None,
                        "computing.eggroll.config_properties_file": None
                        }
    )
```

In this example, creating a FATE context differs from other tutorials because the models are being trained in a distributed environment using multiple GPUs and an Eggroll backend. It's necessary to set the computing engine as EGGROLL and provide several configurations for the federation and Eggroll ports. The task will be running on a machine with serverl GPUs.
When you are using this example, remember to replace the host and port with your own values.

## Models and Dataset

Before submitting the training job to Eggroll, where DeepSpeed operates, it's essential to prepare the models and datasets in the same way as a regular neural network training job. The process begins with importing the necessary packages, including the FedAVGClient and FedAVGServer trainers, and the GPT2 classification model from transformers. Subsequently, a tokenizer dataset is developed to load and tokenize the dataset, as the codes shown below.

```python
import pandas as pd
import torch as t
from transformers import AutoTokenizer
import os
import numpy as np
import argparse
from fate.ml.nn.homo.fedavg import FedAVGClient, FedArguments, TrainingArguments, FedAVGServer
from transformers import GPT2ForSequenceClassification


class TokenizerDataset(Dataset):

    def __init__(
            self,
            truncation=True,
            text_max_length=128,
            tokenizer_name_or_path="bert-base-uncased",
            return_label=True,
            padding=True,
            padding_side="right",
            pad_token=None,
            return_input_ids=True):

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path)
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

        if self.return_input_ids:
            ret = self.word_idx[item]
        else:
            ret = {k: v[item] for k, v in self.word_idx.items()}

        if self.with_label:
            ret.update(dict(labels=self.label[item]))
            return ret

        return ret

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return self.tokenizer.__repr__()
```

In this example we adopt the IMDB movie review dataset, which is a binary classification dataset. You can download from [here](https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/examples/data/IMDB.csv). 

To complete the setup for running federated learning tasks, we define run_server() and run_client() functions. The script is divided into two main functions: run_server() and run_client(), each tailored to the specific roles within the federated learning setup.

- Server Setup (run_server() Function):

    The server (arbiter) initializes its context using the create_ctx function, incorporating the federation session ID and computing session ID.
    A FedAVGServer trainer instance is created and launched into the training process. During this phase, the server automatically collects and aggregates model updates from all participating clients.

- Client Setup (run_client() Function):

    The client's role (guest or host) determines its local party setting.
    Similar to the server, the client establishes its context with the relevant session IDs.
    The client loads a pretrained GPT-2 model from the transformers library.
    A TokenizerDataset instance is initialized and loads'IMDB.csv'. For simplicity, the same dataset is used for both guest and host in this example.
    The FedAVGClient trainer is instantiated with the model, federated arguments, training arguments, loss function, and tokenizer. And we define a deepspeed config for the trainer. When training, it will automatically load the config and use deepspeed to accelerate the training process. Since our trianer is developed based on the transformers trainer, the configuration codes of the trainer is nearly the same.
    Then training process is initiated by calling the train function on the FedAVGClient trainer.


- Key Considerations:

    Consistent Session IDs: It's critical to ensure that both the server and client use the same federation_session_id for successful communication and data exchange.


```python

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 16,
    "train_batch_size": "auto",
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-4
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0
        }
    },
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
}

def run_server():
    federation_session_id = args.federation_session_id
    computing_session_id = f"{federation_session_id}_{arbiter[0]}_{arbiter[1]}"
    ctx = create_ctx(arbiter, federation_session_id, computing_session_id)
    trainer = FedAVGServer(ctx)
    trainer.train()


def run_client():

    if args.role == "guest":
        local_party = guest
        save_path = './guest_model'
    else:
        local_party = host
        save_path = './host_model'
    federation_session_id = args.federation_session_id
    computing_session_id = f"{federation_session_id}_{local_party[0]}_{local_party[1]}"
    ctx = create_ctx(local_party, federation_session_id, computing_session_id)

    pretrained_path = "gpt2"
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_path, num_labels=1)
    model.config.pad_token_id = model.config.eos_token_id
    ds = TokenizerDataset(
        tokenizer_name_or_path=pretrained_path,
        text_max_length=128,
        padding_side="left",
        return_input_ids=False,
        pad_token='<|endoftext|>'
    )
    ds.load("./IMDB.csv")

    fed_args = FedArguments(aggregate_strategy="epoch", aggregate_freq=1, aggregator="secure_aggregate")
    training_args = TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        logging_strategy="steps",
        logging_steps=5,
        deepspeed=deepspeed_config,
    )
    trainer = FedAVGClient(
        ctx=ctx,
        model=model,
        fed_args=fed_args,
        training_args=training_args,
        loss_fn=t.nn.BCELoss(),
        train_set=ds,
        tokenizer=ds.tokenizer
    )
    trainer.train()
    trainer.save_model(save_path)
```

## Full Script

Here is the full script of this example:

```python   
import pandas as pd
import torch as t
from transformers import AutoTokenizer
import os
import numpy as np
import argparse
from fate.ml.nn.homo.fedavg import FedAVGClient, FedArguments, TrainingArguments, FedAVGServer
from transformers import GPT2ForSequenceClassification
from fate.arch.federation import FederationEngine
from fate.arch.computing import ComputingEngine
from fate.arch.context import create_context
from fate.ml.nn.dataset.base import Dataset

# avoid tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenizerDataset(Dataset):

    def __init__(
            self,
            truncation=True,
            text_max_length=128,
            tokenizer_name_or_path="bert-base-uncased",
            return_label=True,
            padding=True,
            padding_side="right",
            pad_token=None,
            return_input_ids=True):

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path)
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

        if self.return_input_ids:
            ret = self.word_idx[item]
        else:
            ret = {k: v[item] for k, v in self.word_idx.items()}

        if self.with_label:
            ret.update(dict(labels=self.label[item]))
            return ret

        return ret

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return self.tokenizer.__repr__()


def create_ctx(local_party, federation_session_id, computing_session_id):
    return create_context(
        local_party=local_party,
        parties=[guest, host, arbiter],
        federation_session_id=federation_session_id,
        federation_engine=FederationEngine.OSX,
        federation_conf={"federation.osx.host": "xxx", "federation.osx.port": xxx, "federation.osx.mode": "stream"},
        computing_session_id=computing_session_id,
        computing_engine=ComputingEngine.EGGROLL,
        computing_conf={"computing.eggroll.host": "xxx",
                        "computing.eggroll.port": xxx,
                        "computing.eggroll.options": {'eggroll.session.processors.per.node': 4, 'nodes': 1},
                        "computing.eggroll.config": None,
                        "computing.eggroll.config_options": None,
                        "computing.eggroll.config_properties_file": None
                        }
    )


def run_server():
    federation_session_id = args.federation_session_id
    computing_session_id = f"{federation_session_id}_{arbiter[0]}_{arbiter[1]}"
    ctx = create_ctx(arbiter, federation_session_id, computing_session_id)
    trainer = FedAVGServer(ctx)
    trainer.train()

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 16,
    "train_batch_size": "auto",
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-4
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0
        }
    },
    "fp16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
    }
}


def run_client():
    if args.role == "guest":
        local_party = guest
        save_path = './guest_model'
    else:
        local_party = host
        save_path = './host_model'
    federation_session_id = args.federation_session_id
    computing_session_id = f"{federation_session_id}_{local_party[0]}_{local_party[1]}"
    ctx = create_ctx(local_party, federation_session_id, computing_session_id)

    pretrained_path = "gpt2"
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_path, num_labels=1)
    model.config.pad_token_id = model.config.eos_token_id
    ds = TokenizerDataset(
        tokenizer_name_or_path=pretrained_path,
        text_max_length=128,
        padding_side="left",
        return_input_ids=False,
        pad_token='<|endoftext|>'
    )
    ds.load("./IMDB.csv")

    fed_args = FedArguments(aggregate_strategy="epoch", aggregate_freq=1, aggregator="secure_aggregate")
    training_args = TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        logging_strategy="steps",
        logging_steps=5,
        deepspeed=deepspeed_config,
    )
    trainer = FedAVGClient(
        ctx=ctx,
        model=model,
        fed_args=fed_args,
        training_args=training_args,
        loss_fn=t.nn.BCELoss(),
        train_set=ds,
        tokenizer=ds.tokenizer
    )
    trainer.train()
    trainer.save_model(save_path)


arbiter = ("arbiter", '9999')
guest = ("guest", '9999')
host = ("host", '9999')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--role", type=str)
    parser.add_argument("--federation_session_id", type=str)

    args = parser.parse_args()
    if args.role == "arbiter":
        run_server()
    else:
        run_client()
```

## Sumbmit the Job

Once the script is ready, we can submit job to eggroll and it will start the training process. The following command will submit the job to eggroll.

```bash
nohup eggroll task submit --script-path gpt2_ds.py --conf role=guest --conf federation_session_id=${federation_session_id} --num-gpus 1 > guest.log &
nohup eggroll task submit --script-path gpt2_ds.py --conf role=host --conf federation_session_id=${federation_session_id} --num-gpus 1 > host.log &
nohup python gpt2_ds.py --role arbiter --federation_session_id ${federation_session_id} > arbiter.log &
```

Below is the partial output of guest process:

```bash
[2023-12-29 13:02:46,862] [INFO] [config.py:976:print]   zero_enabled ................. True
[2023-12-29 13:02:46,862] [INFO] [config.py:976:print]   zero_force_ds_cpu_optimizer .. True
[2023-12-29 13:02:46,862] [INFO] [config.py:976:print]   zero_optimization_stage ...... 2
[2023-12-29 13:02:46,863] [INFO] [config.py:962:print_user_config]   json = {
    "train_micro_batch_size_per_gpu": 16,
    "train_batch_size": 16,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.0005
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0
        }
    },
    "fp16": {
        "enabled": false
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5.000000e+08,
        "overlap_comm": false,
        "reduce_scatter": true,
        "reduce_bucket_size": 5.000000e+08,
        "contiguous_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
{'loss': 2.075, 'learning_rate': 0.0002329900014453396, 'epoch': 0.04}
[2023-12-29 13:02:49,647] [INFO] [logging.py:96:log_dist] [Rank 0] step=10, skipped=0, lr=[0.0003333333333333334], mom=[(0.9, 0.999)]
[2023-12-29 13:02:49,654] [INFO] [timer.py:260:stop] epoch=0/micro_step=10/global_step=10, RunningAvgSamplesPerSec=110.86936118523683, CurrSamplesPerSec=111.33670009091573, MemA          llocated=1.87GB, MaxMemAllocated=6.12GB
{'loss': 0.8091, 'learning_rate': 0.0003333333333333334, 'epoch': 0.08}
{'loss': 0.5003, 'learning_rate': 0.00039203041968522714, 'epoch': 0.12}
[2023-12-29 13:02:51,108] [INFO] [logging.py:96:log_dist] [Rank 0] step=20, skipped=0, lr=[0.0004336766652213271], mom=[(0.9, 0.999)]
[2023-12-29 13:02:51,116] [INFO] [timer.py:260:stop] epoch=0/micro_step=20/global_step=20, RunningAvgSamplesPerSec=111.0120624893855, CurrSamplesPerSec=111.13499047776766, MemAl          located=1.87GB, MaxMemAllocated=6.12GB
{'loss': 0.466, 'learning_rate': 0.0004336766652213271, 'epoch': 0.16}
{'loss': 0.4338, 'learning_rate': 0.0004659800028906792, 'epoch': 0.2}
[2023-12-29 13:02:52,576] [INFO] [logging.py:96:log_dist] [Rank 0] step=30, skipped=0, lr=[0.0004923737515732208], mom=[(0.9, 0.999)]
[2023-12-29 13:02:52,584] [INFO] [timer.py:260:stop] epoch=0/micro_step=30/global_step=30, RunningAvgSamplesPerSec=110.88921855143799, CurrSamplesPerSec=110.59852763526753, MemA          llocated=1.87GB, MaxMemAllocated=6.12GB
{'loss': 0.319, 'learning_rate': 0.0004923737515732208, 'epoch': 0.24}
{'loss': 0.3094, 'learning_rate': 0.0005146893481167587, 'epoch': 0.28}
[2023-12-29 13:02:54,048] [INFO] [logging.py:96:log_dist] [Rank 0] step=40, skipped=0, lr=[0.0005340199971093208], mom=[(0.9, 0.999)]
[2023-12-29 13:02:54,055] [INFO] [timer.py:260:stop] epoch=0/micro_step=40/global_step=40, RunningAvgSamplesPerSec=110.76942622083511, CurrSamplesPerSec=109.53330286609649, MemA          llocated=1.87GB, MaxMemAllocated=6.12GB
```

We can see that the training is running correctly.
