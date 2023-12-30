import torch as t

from fate.arch import Context
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import FedPassArgument, TopModelStrategyArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost


def train(ctx: Context,
          dataset=None,
          model=None,
          optimizer=None,
          loss_func=None,
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


if __name__ == '__main__':
    from fate.arch.launchers.multiprocess_launcher import launch

    launch(run)
