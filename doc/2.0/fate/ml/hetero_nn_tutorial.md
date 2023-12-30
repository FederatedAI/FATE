# Hetero-NN Tutorial

In a hetero-federated learning (vertically partitioned data) setting, multiple parties have different feature sets for the same common user samples. Federated learning enables these parties to collaboratively train a model without sharing their actual data. In FATE-2.0 we introduce our brand new Hetero-NN framework which allows you to quickly set up a hetero federated NN learning task. Since our framework is developed based on pytorch and transformers, it will be easy for you seamlessly integrate your existing dataset, models into our framework. 

In this tutorial, we will show you how to run a Hetero-NN task under FATE-2.0 locally without using a FATE-Pipeline. You can refer to this example for local model experimentation, algorithm modification, and testing.
Besides, in FATE-2.0 we provides two protection strategis: the SSHE and the FedPass. We will show you how to use them in this tutorial.


## Setup Hetero-NN Step by Step

To run a Hetero-NN task, several steps are needed:
1. Import required classes in a new python script
2. Prepare data, datasets, models, loss and optimizers for guest side and host side
3. Configure training parameters; initialize a hetero-nn model; set protection strategy
4. Prepare the trainer
5. Run the training script


## Import Required Classes

In FATE-2.0, our neural network (NN) framework is constructed on the foundations of PyTorch and transformers libraries. This integration facilitates the incorporation of existing models and datasets into federated training.  In our HeteroNN module, we use HeteroNNTrainerGuest and HeteroNNTrainerHost to train the model on guest and host side respectively. They are develop based on huggingface trainer so you can specify the training argument in the same way, via
TrainingAruuments class.

We also provide a HeteroNNModelGuest and HeteroNNModelHost to wrap the top/bottom model and aggregate layer and provide a unified interface for the trainer. You can define your own bottom/top model structure and pass them to the HeteroNNModelGuest and HeteroNNModelHost. We offer two protion strategies: SSHE and FedPass. You can specify them in the HeteroNNModelGuest and HeteroNNModelHost with SSHEArgument and FedPassArgument.

```python
import torch as t
from fate.arch import Context
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.model_zoo.hetero_nn_model import SSHEArgument, FedPassArgument, TopModelStrategyArguments
```

## Tabular Data Example with SSHE

Here we show you an example of using our NN framework, it is a binary classification task whose features are tabular data. 
You can download our example data from: 

- [breast_hetero_guest.csv](https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_guest.csv)
- [breast_hetero_host.csv](https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_host.csv)

And place them in the same directory with your python script.

In this example we will use the SSHEStrategy to protect the data, thus a sshe aggregate layer will be responsible
for aggregate the forwards of guest and host side and propagate the gradients back to guest and host side.

### Fate Context

FATE-2.0 uses a context object to configure the running environment, including party setting(guest, host and theirs party ids). We can create a context object by calling the create_context function.

```python
def create_ctx(party, session_id='test_fate'):
    parties = [("guest", "9999"), ("host", "10000")]
    if party == "guest":
        local_party = ("guest", "9999")
    else:
        local_party = ("host", "10000")
    context = create_context(local_party, parties=parties, federation_session_id=session_id)
    return context
```

If we run our task with launch() (we will explain later), it can automatically handle the context creation, this chapter will introduce the concept of context and show you how to do it manually.

### Prepare

Before starting training, as in PyTorch, we first define the model structure, prepare data, choose a loss function, and instantiate an optimizer. The following code demonstrates the preparation of data, datasets, models, loss, and optimizers. In a hetero-neural network (Hetero-NN) setting, which differs from a homogeneous (homo) federated learning scenario, features and models are divided, with each party managing its own segment. The code uses 'ctx' to differentiate guest and host codes: the guest has labels and 10 features, thus it creates top/model models, while the host, with 20 features and no label, only creates a bottom model. During the initialization of HeteroNNGuestModel and HeteroNNHostModel, SSHEArgument is passed to build a secure share and homomorphic encryption (SSHE) aggregate layer during training, safeguarding the forward and backward processes.

Similar to using a HuggingFace trainer, TrainingArgument is used for setting training parameters. **Note that Hetero-NN currently does not support multi-GPU training, and the SSHE layer is incompatible with GPU training**

Once models, datasets are prepared, we can now start the training process.

```python
def get_setting(ctx):

    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_hetero_guest.csv")

        bottom_model = t.nn.Sequential(
            t.nn.Linear(10, 8),
            t.nn.ReLU(),
        )
        top_model = t.nn.Sequential(
            t.nn.Linear(8, 1),
            t.nn.Sigmoid()
        )
        model = HeteroNNModelGuest(
            top_model=top_model,
            bottom_model=bottom_model,
            agglayer_arg=SSHEArgument(
                guest_in_features=8,
                host_in_features=8,
                out_features=8,
                layer_lr=0.01
            )
        )

        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = t.nn.BCELoss()

    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_hetero_host.csv")
        bottom_model = t.nn.Sequential(
            t.nn.Linear(20, 8),
            t.nn.ReLU(),
        )

        model = HeteroNNModelHost(
            bottom_model=bottom_model,
            agglayer_arg=SSHEArgument(
                guest_in_features=8,
                host_in_features=8,
                out_features=8,
                layer_lr=0.01
            )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = None

    args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=256
    )

    return ds, model, optimizer, loss, args
```

### Run in hetero-federated mode

we add the train() function to initialize trainer for guest and host seperately and add run() function as the entrance for launching the task. The run() function will be called by launch() function in the end of the script. Below is the full code.

```python
import torch as t
from fate.arch import Context
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.model_zoo.hetero_nn_model import SSHEArgument, FedPassArgument, TopModelStrategyArguments



def train(ctx: Context, 
          dataset = None, 
          model = None, 
          optimizer = None, 
          loss_func = None, 
          args: TrainingArguments = None, 
          ):
    
    if ctx.is_on_guest:
        trainer = HeteroNNTrainerGuest(ctx=ctx,
                                       model=model,
                                       train_set=dataset,
                                       optimizer=optimizer,
                                       loss_fn=loss_func,
                                       training_args=args
                                       )
    else:
        trainer = HeteroNNTrainerHost(ctx=ctx,
                                      model=model,
                                      train_set=dataset,
                                      optimizer=optimizer,
                                      training_args=args
                                    )

    trainer.train()
    return trainer


def predict(trainer, dataset):
    return trainer.predict(dataset)

def get_setting(ctx):

    from fate.ml.nn.dataset.table import TableDataset
    # prepare data
    if ctx.is_on_guest:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_hetero_guest.csv")

        bottom_model = t.nn.Sequential(
            t.nn.Linear(10, 8),
            t.nn.ReLU(),
        )
        top_model = t.nn.Sequential(
            t.nn.Linear(8, 1),
            t.nn.Sigmoid()
        )
        model = HeteroNNModelGuest(
            top_model=top_model,
            bottom_model=bottom_model,
            agglayer_arg=SSHEArgument(
                guest_in_features=8,
                host_in_features=8,
                out_features=8,
                layer_lr=0.01
            )
        )

        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = t.nn.BCELoss()

    else:
        ds = TableDataset(to_tensor=True)
        ds.load("./breast_hetero_host.csv")
        bottom_model = t.nn.Sequential(
            t.nn.Linear(20, 8),
            t.nn.ReLU(),
        )

        model = HeteroNNModelHost(
            bottom_model=bottom_model,
            agglayer_arg=SSHEArgument(
                guest_in_features=8,
                host_in_features=8,
                out_features=8,
                layer_lr=0.01
            )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = None

    args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=256
    )

    return ds, model, optimizer, loss, args


def run(ctx):
    ds, model, optimizer, loss, args = get_setting(ctx)
    trainer = train(ctx, ds, model, optimizer, loss, args)
    pred = predict(trainer, ds)
    if ctx.is_on_guest:
        # print("pred:", pred)
        # compute auc here
        from sklearn.metrics import roc_auc_score
        print('auc is')
        print(roc_auc_score(pred.label_ids, pred.predictions))
    

if __name__ == '__main__':
    from fate.arch.launchers.multiprocess_launcher import launch
    launch(run)
```

Save the code as a python script named 'hetero_nn.py' and run it with the following command:

```bash 
python  hetero_nn.py --parties guest:9999 host:10000 --log_level INFO
```

Here is the partial outputs of the consle:

```bash
[15:16:49] INFO     [Rank:0] disabled tracing                                                                                                                                                                                                                                                _trace.py:31
           INFO     [Rank:0] sample id column not found, generate sample id from 0 to 569                                                                                                                                                                                                    table.py:139
label is None
           INFO     [Rank:0] use "y" as label column                                                                                                                                                                                                                                         table.py:150
[15:16:49] INFO     [Rank:1] disabled tracing                                                                                                                                                                                                                                                _trace.py:31
           INFO     [Rank:1] sample id column not found, generate sample id from 0 to 569                                                                                                                                                                                                    table.py:139
label is None
           INFO     [Rank:1] found no "y"/"label"/"target" in input table, no label will be set                                                                                                                                                                                              table.py:153
           INFO     [Rank:0] ***** Running training *****                                                                                                                                                                                                                                 trainer.py:1706
           INFO     [Rank:0]   Num examples = 569                                                                                                                                                                                                                                         trainer.py:1707
           INFO     [Rank:0]   Num Epochs = 3                                                                                                                                                                                                                                             trainer.py:1708
           INFO     [Rank:0]   Instantaneous batch size per device = 256                                                                                                                                                                                                                  trainer.py:1709
           INFO     [Rank:0]   Total train batch size (w. parallel, distributed & accumulation) = 256                                                                                                                                                                                     trainer.py:1712
           INFO     [Rank:0]   Gradient Accumulation steps = 1                                                                                                                                                                                                                            trainer.py:1713
           INFO     [Rank:0]   Total optimization steps = 9                                                                                                                                                                                                                               trainer.py:1714
           INFO     [Rank:0]   Number of trainable parameters = 97                                                                                                                                                                                                                        trainer.py:1715
           INFO     [Rank:1] ***** Running training *****                                                                                                                                                                                                                                 trainer.py:1706
           INFO     [Rank:1]   Num examples = 569                                                                                                                                                                                                                                         trainer.py:1707
           INFO     [Rank:1]   Num Epochs = 3                                                                                                                                                                                                                                             trainer.py:1708
           INFO     [Rank:1]   Instantaneous batch size per device = 256                                                                                                                                                                                                                  trainer.py:1709
           INFO     [Rank:1]   Total train batch size (w. parallel, distributed & accumulation) = 256                                                                                                                                                                                     trainer.py:1712
           INFO     [Rank:1]   Gradient Accumulation steps = 1                                                                                                                                                                                                                            trainer.py:1713
           INFO     [Rank:1]   Total optimization steps = 9                                                                                                                                                                                                                               trainer.py:1714
           INFO     [Rank:1]   Number of trainable parameters = 168                                                                                                                                                                                                                       trainer.py:1715
{'loss': 0.7817, 'learning_rate': 0.01, 'epoch': 1.0}
[15:17:13] INFO     [Rank:0] {'loss': 0.7817, 'learning_rate': 0.01, 'epoch': 1.0, 'step': 3}                                                                                                                                                                                         trainer_base.py:429
{'loss': 0.0, 'learning_rate': 0.01, 'epoch': 1.0}
[15:17:13] INFO     [Rank:1] {'loss': 0.0, 'learning_rate': 0.01, 'epoch': 1.0, 'step': 3}                                                                                                                                                                                            trainer_base.py:429
{'loss': 0.5714, 'learning_rate': 0.01, 'epoch': 2.0}
[15:17:30] INFO     [Rank:0] {'loss': 0.5714, 'learning_rate': 0.01, 'epoch': 2.0, 'step': 6}                                                                                                                                                                                         trainer_base.py:429
{'loss': 0.0, 'learning_rate': 0.01, 'epoch': 2.0}
[15:17:30] INFO     [Rank:1] {'loss': 0.0, 'learning_rate': 0.01, 'epoch': 2.0, 'step': 6}                                                                                                                                                                                            trainer_base.py:429
{'loss': 0.4975, 'learning_rate': 0.01, 'epoch': 3.0}
[15:17:48] INFO     [Rank:0] {'loss': 0.4975, 'learning_rate': 0.01, 'epoch': 3.0, 'step': 9}                                                                                                                                                                                         trainer_base.py:429
{'train_runtime': 58.4774, 'train_samples_per_second': 29.191, 'train_steps_per_second': 0.154, 'train_loss': 0.616881701681349, 'epoch': 3.0}
           INFO     [Rank:0] {'train_runtime': 58.4774, 'train_samples_per_second': 29.191, 'train_steps_per_second': 0.154, 'total_flos': 0.0, 'train_loss': 0.616881701681349, 'epoch': 3.0, 'step': 9}                                                                             trainer_base.py:429
           INFO     [Rank:0] ***** Running Prediction *****                                                                                                                                                                                                                               trainer.py:3154
           INFO     [Rank:0]   Num examples = 569                                                                                                                                                                                                                                         trainer.py:3156
           INFO     [Rank:0]   Batch size = 8                                                                                                                                                                                                                                             trainer.py:3159
{'loss': 0.0, 'learning_rate': 0.01, 'epoch': 3.0}
[15:17:48] INFO     [Rank:1] {'loss': 0.0, 'learning_rate': 0.01, 'epoch': 3.0, 'step': 9}                                                                                                                                                                                            trainer_base.py:429
{'train_runtime': 58.5118, 'train_samples_per_second': 29.174, 'train_steps_per_second': 0.154, 'train_loss': 0.0, 'epoch': 3.0}
           INFO     [Rank:1] {'train_runtime': 58.5118, 'train_samples_per_second': 29.174, 'train_steps_per_second': 0.154, 'total_flos': 0.0, 'train_loss': 0.0, 'epoch': 3.0, 'step': 9}                                                                                           trainer_base.py:429
           INFO     [Rank:1] ***** Running Prediction *****                                                                                                                                                                                                                               trainer.py:3154
           INFO     [Rank:1]   Num examples = 569                                                                                                                                                                                                                                         trainer.py:3156
           INFO     [Rank:1]   Batch size = 8                                                                                                                                                                                                                                             trainer.py:3159
[15:18:07] INFO     [Rank:1] Total: 76.9601s, Driver: 18.8432s(24.48%), Federation: 57.9809s(75.34%), Computing: 0.1361s(0.18%)                                                                                                                                                           _profile.py:279
auc is
0.9712488769092542
```

## Image Data Example with FedPass and Single Bottom Model

To execute an image classification task with the FedPass protection strategy, a few modifications to the settings are required. In our example, the guest possesses only the labels, while the host holds the image data. Consequently, the guest configures a top model (without a bottom model), and the host sets up a bottom model.

We employ the FedPass strategy, detailed in ['FedPass: Privacy-Preserving Vertical Federated Deep Learning with Adaptive Obfuscation'](https://arxiv.org/pdf/2301.12623.pdf). This approach enhances privacy in neural networks by integrating private passports for adaptive obfuscation. It incorporates a 'passport layer' that alters scale and bias in response to these private passports, thus offering robust privacy protection without compromising on model performance.

Let us replace the get_setting() function in the previous example with the following code:

```python
def get_setting(ctx):

    from fate.ml.nn.dataset.table import TableDataset
    import torchvision

    # define model
    from torch import nn
    from torch.nn import init

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm_type=None,
                    relu=False):
            super().__init__()

            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.norm_type = norm_type

            if self.norm_type:
                if self.norm_type == 'bn':
                    self.bn = nn.BatchNorm2d(out_channels)
                elif self.norm_type == 'gn':
                    self.bn = nn.GroupNorm(out_channels // 16, out_channels)
                elif self.norm_type == 'in':
                    self.bn = nn.InstanceNorm2d(out_channels)
                else:
                    raise ValueError("Wrong norm_type")
            else:
                self.bn = None

            if relu:
                self.relu = nn.ReLU(inplace=True)
            else:
                self.relu = None

            self.reset_parameters()

        def reset_parameters(self):
            init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        def forward(self, x, scales=None, biases=None):
            x = self.conv(x)
            if self.norm_type is not None:
                x = self.bn(x)
            if scales is not None and biases is not None:
                x = scales[-1] * x + biases[-1]

            if self.relu is not None:
                x = self.relu(x)
            return x
    
    # host top model
    class LeNetBottom(nn.Module):
        def __init__(self):
            super(LeNetBottom, self).__init__()
            self.layer0 = nn.Sequential(
                ConvBlock(1, 8, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

        def forward(self, x):
            x = self.layer0(x)
            return x

    # guest top model
    class LeNetTop(nn.Module):

        def __init__(self, out_feat=84):
            super(LeNetTop, self).__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc1act = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(120, 84)
            self.fc2act = nn.ReLU(inplace=True)
            self.fc3 = nn.Linear(84, out_feat)

        def forward(self, x_a):
            x = x_a
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc1act(x)
            x = self.fc2(x)
            x = self.fc2act(x)
            x = self.fc3(x)
            return x
        
    # fed simulate tool
    from torch.utils.data import Dataset

    class NoFeatureDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, item):
            return [self.ds[item][1]]
        
    class NoLabelDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, item):
            return [self.ds[item][0]]


    # prepare mnist data
    train_data = torchvision.datasets.MNIST(root='./',
                                            train=True, download=True, transform=torchvision.transforms.ToTensor())
    
    if ctx.is_on_guest:
        
        model = HeteroNNModelGuest(
            top_model=LeNetTop(),
            top_arg=TopModelStrategyArguments(
                protect_strategy='fedpass',
                fed_pass_arg=FedPassArgument(
                    layer_type='linear',
                    in_channels_or_features=84,
                    hidden_features=64,
                    out_channels_or_features=10,
                    passport_mode='multi',
                    activation='relu',
                    num_passport=1000,
                    low=-10
                )
            )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = t.nn.CrossEntropyLoss()
        ds = NoFeatureDataset(train_data)

    else:

        model = HeteroNNModelHost(
            bottom_model=LeNetBottom(),
            agglayer_arg=FedPassArgument(
                layer_type='conv',
                in_channels_or_features=8,
                out_channels_or_features=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                passport_mode='multi',
                activation='relu',
                num_passport=1000
            )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        loss = None
        ds = NoLabelDataset(train_data)

    args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=256,
        disable_tqdm=False
    )

    return ds, model, optimizer, loss, args


def run(ctx):
    ds, model, optimizer, loss, args = get_setting(ctx)
    trainer = train(ctx, ds, model, optimizer, loss, args)
    pred = predict(trainer, ds)
```

In this configuration, we utilize the LeNet model both as the bottom and top models. The dataset is sourced from torchvision.datasets.MNIST. We use FedPassArgument to establish the FedPass aggregate layer. It's important to note that the FedPass argument for the bottom model is set using agg_layer_arg, and for the top model using top_arg. Both models are equipped with FedPass protection: during training, random passports are generated, which obfuscate the forward hidden features and backward gradients.

Another key aspect is the use of NoFeatureDataset and NoLabelDataset to encapsulate the dataset. This approach reflects the scenario where the guest holds only labels and the host possesses only features. This simplification aids in effectively simulating the federated learning environment.

The task can be submitted using the same command as in the previous example:

```bash
python  hetero_nn.py --parties guest:9999 host:10000 --log_level INFO
```