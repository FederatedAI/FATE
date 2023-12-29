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

from torch.nn import init
import sys
import torch
import torch as t
import torchvision
from torch import nn
from datetime import datetime
from fate.ml.nn.model_zoo.agg_layer.fedpass._passport_block import ConvPassportBlock
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments, FedPassArgument


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


guest = ("guest", "10000")
host = ("host", "9999")
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
    computing = CSession(data_dir="./session_dir")
    return Context(computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host]))


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm_type=None, relu=False
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm_type = norm_type

        if self.norm_type:
            if self.norm_type == "bn":
                self.bn = nn.BatchNorm2d(out_channels)
            elif self.norm_type == "gn":
                self.bn = nn.GroupNorm(out_channels // 16, out_channels)
            elif self.norm_type == "in":
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
        init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, scales=None, biases=None):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.bn(x)
        # print("scales:", scales)
        # print("biases:", biases)
        if scales is not None and biases is not None:
            # print("convent forward")
            x = scales[-1] * x + biases[-1]

        if self.relu is not None:
            x = self.relu(x)
        return x


class LeNetBottom(nn.Module):
    def __init__(self):
        super(LeNetBottom, self).__init__()
        self.layer0 = nn.Sequential(ConvBlock(1, 8, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))

    def forward(self, x):
        x = self.layer0(x)
        return x


class LeNetBottom_(nn.Module):
    def __init__(self):
        super(LeNetBottom_, self).__init__()
        self.layer0 = nn.Sequential(ConvBlock(1, 8, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))

        self.layer1 = nn.Sequential(
            ConvPassportBlock(8, 16, 5, num_passport=64, passport_mode="multi"), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x


class LeNetBottomRaw(nn.Module):
    def __init__(self):
        super(LeNetBottomRaw, self).__init__()
        self.layer0 = nn.Sequential(ConvBlock(1, 8, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))

        self.layer1 = nn.Sequential(ConvBlock(8, 16, kernel_size=5), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x


class LeNet_Top(nn.Module):
    def __init__(self, out_feat=10):
        super(LeNet_Top, self).__init__()
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


def hash_model_parameters(model):
    import hashlib

    m = hashlib.md5()
    for param in model.parameters():
        m.update(param.data.cpu().numpy().tobytes())
    return m.hexdigest()


if __name__ == "__main__":
    party = sys.argv[1]

    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

    set_seed(42)

    train_data = torchvision.datasets.MNIST(
        root="./mnist", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

    test_data = torchvision.datasets.MNIST(
        root="./mnist", train=False, download=True, transform=torchvision.transforms.ToTensor()
    )

    # digit_indices = [[] for _ in range(10)]
    # for idx, (_, label) in enumerate(train_data):
    #     digit_indices[label].append(idx)

    # selected_train_indices = []
    # for indices in digit_indices:
    #     selected_train_indices.extend(torch.randperm(len(indices))[:10000].tolist())

    # selected_val_indices = []
    # for indices in digit_indices:
    #     remaining_indices = [idx for idx in indices if idx not in selected_train_indices]
    #     selected_val_indices.extend(torch.randperm(len(remaining_indices))[:1000].tolist())

    # subset_train_data = torch.utils.data.Subset(train_data, selected_train_indices)
    # subset_val_data = torch.utils.data.Subset(train_data, selected_val_indices)

    subset_train_data = train_data
    subset_val_data = test_data

    epochs = 10

    from torch.utils.data import Dataset

    class NoFeatureDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, item):
            return [self.ds[item][1]]

    arg = TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=512,
        disable_tqdm=False,
        per_gpu_eval_batch_size=512,
        eval_steps=1,
        evaluation_strategy="epoch",
        no_cuda=False,
    )

    if party == "guest":
        from fate.ml.evaluation.metric_base import MetricEnsemble
        from fate.ml.evaluation.classification import MultiAccuracy
        from fate.ml.nn.model_zoo.hetero_nn_model import TopModelStrategyArguments, FedPassArgument

        ctx = create_ctx(guest, get_current_datetime_str())
        top_model = LeNet_Top(out_feat=84)
        model = HeteroNNModelGuest(
            top_model=top_model,
            top_arg=TopModelStrategyArguments(
                protect_strategy="fedpass",
                fed_pass_arg=FedPassArgument(
                    layer_type="linear",
                    num_passport=64,
                    in_channels_or_features=84,
                    hidden_features=64,
                    out_channels_or_features=10,
                    passport_mode="single",
                ),
            ),
            ctx=ctx,
        )
        loss = nn.CrossEntropyLoss()
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        trainer = HeteroNNTrainerGuest(
            ctx,
            model,
            training_args=arg,
            train_set=NoFeatureDataset(subset_train_data),
            val_set=NoFeatureDataset(subset_val_data),
            loss_fn=loss,
            optimizer=optimizer,
            compute_metrics=MetricEnsemble().add_metric(MultiAccuracy()),
        )
        trainer.train()

    if party == "host":
        ctx = create_ctx(host, get_current_datetime_str())

        bottom_model = LeNetBottom()
        passport_mode = "multi"
        print("passport mode is {}".format(passport_mode))
        model = HeteroNNModelHost(
            bottom_model=bottom_model,
            agglayer_arg=FedPassArgument(
                layer_type="conv",
                in_channels_or_features=8,
                out_channels_or_features=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                passport_mode=passport_mode,
                activation="relu",
                num_passport=64,
            ),
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        trainer = HeteroNNTrainerHost(
            ctx, model, training_args=arg, train_set=subset_train_data, val_set=subset_val_data, optimizer=optimizer
        )
        trainer.train()

    elif party == "test":
        from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, TopModelStrategyArguments, FedPassArgument

        top_model = LeNet_Top(out_feat=84)
        model = HeteroNNModelGuest(top_model=top_model)
        model.setup(
            top_arg=TopModelStrategyArguments(
                protect_strategy="fedpass",
                fed_pass_arg=FedPassArgument(
                    num_passport=64,
                    in_channels_or_features=84,
                    hidden_features=64,
                    out_channels_or_features=10,
                    passport_mode="single",
                ),
            )
        )
