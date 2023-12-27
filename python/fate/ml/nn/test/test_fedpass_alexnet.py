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
from fate.ml.nn.homo.fedavg import FedAVGClient, TrainingArguments
from fate.ml.nn.model_zoo.agg_layer.fedpass._passport_block import ConvPassportBlock
from fate.ml.nn.model_zoo.agg_layer.fedpass.agg_layer import FedPassAggLayerHost, FedPassAggLayerGuest
from tqdm import tqdm
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


guest = ("guest", "10000")
host = ("host", "9999")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
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
    computing = CSession()
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


class AlexNet_Bottom(nn.Module):
    def __init__(self):
        super(AlexNet_Bottom, self).__init__()
        self.features = []
        self.layer0 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer1 = nn.Sequential(
            ConvBlock(64, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for i in range(2):
            layer = getattr(self, f"layer{i}")
            self.features.append(layer)

    def forward(self, x):
        for feature_layer in self.features:
            x = feature_layer(x)
        return x


class Alexnet_Top(nn.Module):
    def __init__(self, num_classes=10):
        super(Alexnet_Top, self).__init__()
        self.layer3 = nn.Sequential(
            ConvBlock(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes),
        )

    def forward(self, x_a, x_b=None):
        if x_b is not None:
            x = torch.cat([x_a, x_b], dim=-1)
        elif isinstance(x_a, list):
            x = sum(x_a)
        else:
            x = x_a

        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
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

    train_data = torchvision.datasets.CIFAR10(
        root="./cifar10", train=True, download=True, transform=torchvision.transforms.ToTensor()
    )

    digit_indices = [[] for _ in range(10)]
    for idx, (_, label) in enumerate(train_data):
        digit_indices[label].append(idx)

    selected_train_indices = []
    for indices in digit_indices:
        selected_train_indices.extend(torch.randperm(len(indices))[:500].tolist())

    selected_val_indices = []
    for indices in digit_indices:
        remaining_indices = [idx for idx in indices if idx not in selected_train_indices]
        selected_val_indices.extend(torch.randperm(len(remaining_indices))[:100].tolist())

    subset_train_data = torch.utils.data.Subset(train_data, selected_train_indices)
    subset_val_data = torch.utils.data.Subset(train_data, selected_val_indices)

    epochs = 50

    from torch.utils.data import Dataset

    class NoFeatureDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, item):
            return [self.ds[item][1]]

    arg = TrainingArguments(
        num_train_epochs=20,
        per_device_train_batch_size=16,
        disable_tqdm=False,
        eval_steps=1,
        evaluation_strategy="epoch",
    )

    if party == "local":
        from fate.ml.evaluation.metric_base import MetricEnsemble
        from fate.ml.evaluation.classification import MultiAccuracy

        ctx = create_ctx(guest, get_current_datetime_str())

        top_model = Alexnet_Top()
        model = t.nn.Sequential(
            AlexNet_Bottom(),
            ConvPassportBlock(
                in_channels=192,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
                num_passport=4,
                passport_mode="multi",
                activation="relu",
            ),
            Alexnet_Top(),
        )

        loss = nn.CrossEntropyLoss()
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        trainer = FedAVGClient(
            ctx=ctx,
            model=model,
            training_args=arg,
            train_set=subset_train_data,
            val_set=subset_val_data,
            loss_fn=loss,
            optimizer=optimizer,
            compute_metrics=MetricEnsemble().add_metric(MultiAccuracy()),
            fed_args=None,
        )
        trainer.set_local_mode()
        trainer.train()

    if party == "guest":
        from fate.ml.evaluation.metric_base import MetricEnsemble
        from fate.ml.evaluation.classification import MultiAccuracy

        ctx = create_ctx(guest, get_current_datetime_str())

        top_model = Alexnet_Top()
        model = HeteroNNModelGuest(top_model=top_model, agg_layer=FedPassAggLayerGuest())
        loss = nn.CrossEntropyLoss()
        optimizer = t.optim.Adam(model.parameters(), lr=0.001)

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

        bottom_model = AlexNet_Bottom()

        fedpass_layer = FedPassAggLayerHost(
            layer_type="conv",
            in_channels_or_features=192,
            out_channels_or_features=384,
            kernel_size=3,
            stride=1,
            padding=1,
            num_passport=4,
            passport_mode="multi",
            activation="relu",
        )

        model = HeteroNNModelHost(agg_layer=fedpass_layer, bottom_model=bottom_model)
        optimizer = t.optim.Adam(model.parameters(), lr=0.001)

        trainer = HeteroNNTrainerHost(
            ctx, model, training_args=arg, train_set=subset_train_data, val_set=subset_val_data, optimizer=optimizer
        )
        trainer.train()
