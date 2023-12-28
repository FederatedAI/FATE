# Homo-NN Tutorial

The Homo(horizontal) federated learning allows parties to collaboratively train a neural network model without sharing their actual data. In a horizontally partitioned data setting, multiple parties have the same feature set but different user samples. In this scenario, each party trains the model locally on its own subset of data and only shares the model updates.

The Homo-NN framework is designed for horizontal federated neural network training. 
In this tutorial, we demonstrate how to run a Homo-NN task under FATE-2.0 locally without using a Pipeline and provide several demos to show you how to integrate linear models, image models, language models in federated
learning. These setting are particularly useful for local experimentation, model/training setting modifications and testing. 


## Setup Homo-NN Setup by Step

To run a Homo-NN task, we need to:
1. Import required classes
2. Prepare data, datasets, models, loss and optimizers
3. Configure clients(guest&hosts) parameters, trainers; configure sever(arbiter) parameters and trainer
4. Run the training script


## Import Required Classes

In FATE-2.0, our neural network (NN) framework is constructed on the foundations of PyTorch and transformers libraries. This integration facilitates the incorporation of existing models and datasets into federated training. Additionally, the framework supports accelerated computing resources and frameworks such as GPU and DeepSpeed. In our HomoNN module, we offer support for standard FedAVG algorithms, and with the FedAVGClient and FedAVGServer trainer classes, setting up horizontal federated learning tasks becomes quick and efficient. FedAVGClient is developed on the transformer trainer, enabling the uniform specification of training and federation parameters through TrainingArguments and FedAVGArguments.

```python
from fate.ml.nn.homo.fedavg import FedAVGClient, FedAVGServer
from fate.ml.nn.homo.fedavg import FedAVGArguments, TrainingArguments
import torch as t
import pandas as pd
from fate.arch import Context
```

## Tabluar Data Examples: Prepare data, datasets, models, loss and optimizers

Here we show you an example of using our NN framework, it is a binary classification task whose features are tabular data.

You can download our example data from: 

- [breast_homo_guest.csv](https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_homo_guest.csv)
- [breast_homo_host.csv](https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_homo_host.csv)


And put them in the same directory as this tutorial. This dataset has 30 features and binary labels, in total 569 samples.

### Fate Context and Parties

FATE-2.0 uses a context object to configure the running environment, including party setting(guest, host and theirs party ids). We can create a context object by calling the create_context function.

```python

def create_ctx(party, session_id='test_fate'):
    parties = [("guest", "9999"), ("host", "10000"), ("arbiter", "10000")]
    if party == "guest":
        local_party = ("guest", "9999")
    elif party == "host":
        local_party = ("host", "10000")
    else:
        local_party = ("arbiter", "10000")
    from fate.arch.context import create_context
    context = create_context(local_party, parties=parties, federation_session_id=session_id)
    return context
```

If we run our task with launch() (we will explain later), it can automatically handle the context creation, this chapter will introduce the concept of context and show you how to do it manually.

In the FATE Homo Federated Learning scenario, the guest and host act as clients, while the arbiter functions as the server.

### Prepare

Before initiating training, similar to PyTorch practices, our initial steps involve defining the model structure, preparing the data, selecting a loss function, and instantiating an optimizer. Here shows the code for preparing data, datasets, models, loss and optimizers.
We use our built-in dataset to load Tabular data and create a simple 2-layer model.
Please notice that we use ctx to check if we are on guest or host. The arbiter
side doesn't need to prepare any model and data beacuse it is the sever side who is only responsible for model aggregation.

```python
def get_tabular_task_setting(ctx: Context):

    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_homo_guest.csv")
    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_homo_host.csv")
    # prepare model
    model = t.nn.Sequential(
        t.nn.Linear(30, 16),
        t.nn.ReLU(),
        t.nn.Linear(16, 1),
        t.nn.Sigmoid()
    )
    # prepare loss
    loss = t.nn.BCELoss()
    # prepare optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    args = TrainingArguments(
        num_train_epochs=4,
        per_device_train_batch_size=256
    )
    fed_arg = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    return ds, model, optimizer, loss, args, fed_arg
```

### Test Your NN Task Locally

Since our framework is develop based on pytorch and transfomers, we can run our training task locally to ensure that we can go through the whole training process.
When running locally, the FedAVGClient operates similarly to the transformer trainer. **Rembember to set it to local model if you want to run it locally**

```python
from fate.ml.nn.homo.fedavg import FedAVGClient, FedAVGServer
from fate.ml.nn.homo.fedavg import FedAVGArguments, TrainingArguments
import torch as t
import pandas as pd
from fate.arch import Context


def create_ctx(party, session_id='test_fate'):
    parties = [("guest", "9999"), ("host", "10000"), ("arbiter", "10000")]
    if party == "guest":
        local_party = ("guest", "9999")
    elif party == "host":
        local_party = ("host", "10000")
    else:
        local_party = ("arbiter", "10000")
    from fate.arch.context import create_context
    context = create_context(local_party, parties=parties, federation_session_id=session_id)
    return context

def get_setting(ctx: Context):

    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_homo_guest.csv")
    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_homo_host.csv")
    # prepare model
    model = t.nn.Sequential(
        t.nn.Linear(30, 16),
        t.nn.ReLU(),
        t.nn.Linear(16, 1),
        t.nn.Sigmoid()
    )
    # prepare loss
    loss = t.nn.BCELoss()
    # prepare optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    args = TrainingArguments(
        num_train_epochs=4,
        per_device_train_batch_size=256
    )
    fed_arg = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    return ds, model, optimizer, loss, args, fed_arg

# host is also ok, guest & host are clients and hold their own model and data
# So we can test locally before federated learning
ctx = create_ctx('guest')  
dataset, model, optimizer, loss_func, args, fed_args = get_setting(ctx)
print('data' + '*' * 10)
print(dataset[0])
print('model' + '*' * 10)
print(model)

trainer = FedAVGClient(
    ctx=ctx,
    model=model,
    train_set=dataset,
    optimizer=optimizer,
    loss_fn=loss_func,
    training_args=args,
    fed_args=fed_args
)

trainer.set_local_mode()
trainer.train()

# compute auc here
from sklearn.metrics import roc_auc_score
pred = trainer.predict(dataset)
auc = roc_auc_score(pred.label_ids, pred.predictions)
print(auc)
```

And the console will output
```
data**********
(tensor([ 0.2549, -1.0466,  0.2097,  0.0742, -0.4414, -0.3776, -0.4859,  0.3471,
        -0.2876, -0.7335,  0.4495, -1.2472,  0.4132,  0.3038, -0.1238, -0.1842,
        -0.2191,  0.2685,  0.0160, -0.7893, -0.3374, -0.7282, -0.4426, -0.2728,
        -0.6080, -0.5772, -0.5011,  0.1434, -0.4664, -0.5541]), tensor([1.]))
model**********
Sequential(
  (0): Linear(in_features=30, out_features=16, bias=True)
  (1): ReLU()
  (2): Linear(in_features=16, out_features=1, bias=True)
  (3): Sigmoid()
)
{'loss': 0.7443, 'learning_rate': 0.01, 'epoch': 1.0}
{'loss': 0.6664, 'learning_rate': 0.01, 'epoch': 2.0}
{'loss': 0.6028, 'learning_rate': 0.01, 'epoch': 3.0}
{'loss': 0.5463, 'learning_rate': 0.01, 'epoch': 4.0}
{'train_runtime': 0.0098, 'train_samples_per_second': 92238.321, 'train_steps_per_second': 406.336, 'train_loss': 0.6399458050727844, 'epoch': 4.0}
0.9784415584415584
```

The task ran successfully, and we can see the loss is decreasing and we got an auc of 0.9784. Good!

### Add evaluation metrics
You can add evaluation metrics to trainer in order to evaluate the performance during training.

```python
args = TrainingArguments(
    num_train_epochs=4,
    per_device_train_batch_size=256,
    evaluation_strategy='epoch'
)

def auc(pred):
    from sklearn.metrics import roc_auc_score
    return {'auc': roc_auc_score(pred.label_ids, pred.predictions)}

trainer = FedAVGClient(
    ctx=ctx,
    model=model,
    train_set=dataset,
    val_set=dataset,
    optimizer=optimizer,
    loss_fn=loss_func,
    training_args=args,
    fed_args=fed_args,
    compute_metrics=auc
)
```

### Run in federated mode

There are several differences between running locally and running in federated model.
Firstly, we add the initalization of FedAVGServer, whose is responsible model aggregation.
Secondly, we add the run() function to execute the training code according to the party, and use launch as the main entrance to run the task.
The launch() will automatically start 3 process(guest, host and aribter) and initalize ctx for each party.
Then it will run a local simulated federated learning task.

```python
from fate.ml.nn.homo.fedavg import FedAVGArguments, FedAVGClient, FedAVGServer, TrainingArguments
from fate.arch import Context
import torch as t
from fate.arch.launchers.multiprocess_launcher import launch


def get_setting(ctx: Context):

    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_homo_guest.csv")
    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_homo_host.csv")
    # prepare model
    model = t.nn.Sequential(
        t.nn.Linear(30, 16),
        t.nn.ReLU(),
        t.nn.Linear(16, 1),
        t.nn.Sigmoid()
    )
    # prepare loss
    loss = t.nn.BCELoss()
    # prepare optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    args = TrainingArguments(
        num_train_epochs=4,
        per_device_train_batch_size=256
    )
    fed_arg = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    return ds, model, optimizer, loss, args, fed_arg


def train(ctx: Context, 
          dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          args: TrainingArguments = None, 
          fed_args: FedAVGArguments = None
          ):
    
    if ctx.is_on_guest or ctx.is_on_host:
        trainer = FedAVGClient(ctx=ctx,
                               model=model,
                               train_set=dataset,
                               optimizer=optimizer,
                               loss_fn=loss_func,
                               training_args=args,
                               fed_args=fed_args
                               )
    else:
        trainer = FedAVGServer(ctx)

    trainer.train()
    return trainer

def run(ctx: Context):

    if ctx.is_on_arbiter:
        train(ctx)
    else:
        train(ctx, *get_tabular_task_setting(ctx))

if __name__ == '__main__':
    launch(run)
```

We save the script above as nn.py and launch it:
```
python nn.py --parties guest:9999 host:10000 arbiter:10000 --log_level INFO
```

Here shows partial of the console output:
```
[20:19:00] INFO     [ Main ] ========================================================                                                                                                                                                                         multiprocess_launcher.py:277
           INFO     [ Main ] federation id: 20231227201900-1e054c                                                                                                                                                                                             multiprocess_launcher.py:278
           INFO     [ Main ] parties: ['guest:9999', 'host:10000', 'arbiter:10000']                                                                                                                                                                           multiprocess_launcher.py:279
           INFO     [ Main ] data dir: None                                                                                                                                                                                                                   multiprocess_launcher.py:280
           INFO     [ Main ] ========================================================                                                                                                                                                                         multiprocess_launcher.py:281
[20:19:01] INFO     [ Main ] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
           INFO     [ Main ] waiting for all processes to exit                                                                                                                                                                                                multiprocess_launcher.py:220
[20:19:02] INFO     [Rank:0] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
[20:19:02] INFO     [Rank:2] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
           INFO     [Rank:0] sample id column not found, generate sample id from 0 to 227                                                                                                                                                                                     table.py:139
           INFO     [Rank:0] use "y" as label column                                                                                                                                                                                                                          table.py:149
[20:19:02] INFO     [Rank:1] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
           INFO     [Rank:1] sample id column not found, generate sample id from 0 to 228                                                                                                                                                                                     table.py:139
           INFO     [Rank:1] use "y" as label column                                                                                                                                                                                                                          table.py:149
           INFO     [Rank:0] ***** Running training *****                                                                                                                                                                                                                  trainer.py:1706
           INFO     [Rank:0]   Num examples = 227                                                                                                                                                                                                                          trainer.py:1707
           INFO     [Rank:0]   Num Epochs = 3                                                                                                                                                                                                                              trainer.py:1708
           INFO     [Rank:0]   Instantaneous batch size per device = 256                                                                                                                                                                                                   trainer.py:1709
           INFO     [Rank:0]   Total train batch size (w. parallel, distributed & accumulation) = 256                                                                                                                                                                      trainer.py:1712
           INFO     [Rank:0]   Gradient Accumulation steps = 1                                                                                                                                                                                                             trainer.py:1713
           INFO     [Rank:0]   Total optimization steps = 3                                                                                                                                                                                                                trainer.py:1714
           INFO     [Rank:0]   Number of trainable parameters = 513                                                                                                                                                                                                        trainer.py:1715
           INFO     [Rank:0] Using secure_aggregate aggregator, rank=0                                                                                                                                                                                            aggregator_wrapper.py:50
           INFO     [Rank:1] ***** Running training *****                                                                                                                                                                                                                  trainer.py:1706
           INFO     [Rank:1]   Num examples = 228                                                                                                                                                                                                                          trainer.py:1707
           INFO     [Rank:1]   Num Epochs = 3                                                                                                                                                                                                                              trainer.py:1708
           INFO     [Rank:1]   Instantaneous batch size per device = 256                                                                                                                                                                                                   trainer.py:1709
           INFO     [Rank:1]   Total train batch size (w. parallel, distributed & accumulation) = 256                                                                                                                                                                      trainer.py:1712
           INFO     [Rank:1]   Gradient Accumulation steps = 1                                                                                                                                                                                                             trainer.py:1713
           INFO     [Rank:0] computing weights                                                                                                                                                                                                                                  base.py:84
           INFO     [Rank:1]   Total optimization steps = 3                                                                                                                                                                                                                trainer.py:1714
           INFO     [Rank:1]   Number of trainable parameters = 513                                                                                                                                                                                                        trainer.py:1715
           INFO     [Rank:1] Using secure_aggregate aggregator, rank=0                                                                                                                                                                                            aggregator_wrapper.py:50
           INFO     [Rank:1] computing weights                                                                                                                                                                                                                                  base.py:84
           INFO     [Rank:2] Using secure_aggregate aggregator                                                                                                                                                                                                   aggregator_wrapper.py:165
           INFO     [Rank:2] Initialized aggregator Done: <fate.ml.aggregator.aggregator_wrapper.AggregatorServerWrapper object at 0x7f251cdfca30>                                                                                                                    trainer_base.py:1173
           INFO     [Rank:0] aggregate weight is 0.4989010989010989                                                                                                                                                                                                             base.py:99
           INFO     [Rank:1] aggregate weight is 0.5010989010989011                                                                                                                                                                                                             base.py:99
           INFO     [Rank:1] computed max_aggregation is 3                                                                                                                                                                                                             trainer_base.py:503
           INFO     [Rank:1] parameters is {'num_train_epochs': 3, 'max_steps': 3, 'num_update_steps_per_epoch': 1, 'epochs_trained': 0, 'steps_trained_in_current_epoch': 0, 'max_aggregation': 3, 'aggregate_freq': 1, 'aggregation_strategy': 'epoch',              trainer_base.py:522
                    'can_aggregate_loss': True}                                                                                                                                                                                                                                           
           INFO     [Rank:0] computed max_aggregation is 3                                                                                                                                                                                                             trainer_base.py:503
           INFO     [Rank:0] parameters is {'num_train_epochs': 3, 'max_steps': 3, 'num_update_steps_per_epoch': 1, 'epochs_trained': 0, 'steps_trained_in_current_epoch': 0, 'max_aggregation': 3, 'aggregate_freq': 1, 'aggregation_strategy': 'epoch',              trainer_base.py:522
                    'can_aggregate_loss': True}                                                                                                                                                                                                                                           
           INFO     [Rank:2] checked parameters are {'max_aggregation': 3, 'can_aggregate_loss': True}                                                                                                                                                                trainer_base.py:1178
           INFO     [Rank:0] aggregation on epoch end                                                                                                                                                                                                                  trainer_base.py:675
           INFO     [Rank:0] begin to agg model                                                                                                                                                                                                                   aggregator_wrapper.py:60
           INFO     [Rank:1] aggregation on epoch end                                                                                                                                                                                                                  trainer_base.py:675
           INFO     [Rank:1] begin to agg model                                                                                                                                                                                                                   aggregator_wrapper.py:60
           INFO     [Rank:0] begin to agg model                                                                                                                                                                                                                   aggregator_wrapper.py:67
           INFO     [Rank:0] Aggregation count: 1 / 3                                                                                                                                                                                                                  trainer_base.py:388
{'loss': 0.772, 'learning_rate': 0.01, 'epoch': 1.0}
           INFO     [Rank:0] {'loss': 0.772, 'learning_rate': 0.01, 'epoch': 1.0, 'step': 1}                                                                                                                                                                           trainer_base.py:429
           INFO     [Rank:0] begin to agg loss                                                                                                                                                                                                                    aggregator_wrapper.py:70
           INFO     [Rank:0] end to gather loss                                                                                                                                                                                                                   aggregator_wrapper.py:73
           INFO     [Rank:1] begin to agg model                                                                                                                                                                                                                   aggregator_wrapper.py:67
           INFO     [Rank:1] Aggregation count: 1 / 3                                                                                                                                                                                                                  trainer_base.py:388
{'loss': 0.8367, 'learning_rate': 0.01, 'epoch': 1.0}
           INFO     [Rank:1] {'loss': 0.8367, 'learning_rate': 0.01, 'epoch': 1.0, 'step': 1}               
```

You can modify this demo to run your own task.

## Image Model: Resnet and Cifar10
By simply modify the setting function, we are able to efforlessly incorperate a classic image classification task into fedearted learning:
 ```python
def get_setting(ctx):

    from torchvision import models, datasets, transforms

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    total_trainset_length = len(trainset)
    split_length = total_trainset_length // 2
    if total_trainset_length % 2 != 0:
        split_length += 1
    trainset_part1, trainset_part2 = t.utils.data.random_split(
        trainset, [split_length, total_trainset_length - split_length])
    
    # we seperate the dataset into two parts, one for guest, one for host
    # to simulate the federated learning
    if ctx.is_on_guest:
        trainset= trainset_part1
    elif ctx.is_on_host:
        trainset = trainset_part2

    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = t.nn.Linear(num_ftrs, 10) 
    model = net

    # prepare loss
    loss = t.nn.CrossEntropyLoss()
    # prepare optimizer
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    args = TrainingArguments(
        num_train_epochs=4,
        per_device_train_batch_size=256,
        evaluation_strategy='epoch'
    )
    fed_arg = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    return trainset, model, optimizer, loss, args, fed_arg
 ```
 Replace the get_setting function in nn.py with the above code, and run it again, you are able to train a federated resnet18 model on cifar10 dataset.

```
python nn.py --parties guest:9999 host:10000 arbiter:10000 --log_level INFO
```

## Texte Classification with Generative model: GPT2 and IMDB

By simply modifying the setting function and replacing the get_setting function in nn.py with the following code, we are able to use huggingface pretrained model and builit-in dataset to train a federated text classification model on IMDB dataset. We can seamlessly integrate the huggingface pretrained model and dataset into our framework. Please notice that loss function is not needed in this case beacuse the model outputs the loss.
To run this demo, please make sure that you have installed datasets and have sufficent computing resources.

```python
def get_setting(ctx):

    from datasets import load_dataset
    dataset = load_dataset('imdb')
    from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=512, truncation=True)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('imdb')
    dataset = dataset.rename_column("label", "labels")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=16)
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = GPT2ForSequenceClassification.from_pretrained('gpt2')
    args = TrainingArguments(
        num_train_epochs=4,
        per_device_train_batch_size=1,
        evaluation_strategy='epoch',
        disable_tqdm=False
    )
    fed_arg = FedAVGArguments(
        aggregate_strategy='epoch',
        aggregate_freq=1
    )

    optimizer = t.optim.Adam(model.parameters(), lr=5e-5)

    # we don't seperate the dataset for the sake of simplicity
    return tokenized_dataset['train'], model, optimizer, None, args, fed_arg
```