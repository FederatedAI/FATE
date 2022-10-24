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
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from ruamel import yaml
from torch.utils.data import Dataset

__all__ = ["TableDataSet", "VisionDataSet"]

from fate_arch.session import computing_session
from fate_arch.abc import CTableABC
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderClient


class DatasetMixIn(Dataset):
    def get_num_features(self):
        raise NotImplementedError()

    def get_num_labels(self):
        raise NotImplementedError()

    def get_label_align_mapping(self):
        return None


class TableDataSet(DatasetMixIn):
    def get_num_features(self):
        return self._num_features

    def get_num_labels(self):
        return self._num_labels

    def get_label_align_mapping(self):
        return self._label_align_mapping

    def get_keys(self):
        return self._keys

    def __init__(
        self,
        data_instances: CTableABC,
        expected_label_type=np.float32,
        label_align_mapping=None,
        **kwargs,
    ):

        # partition
        self.partitions = data_instances.partitions

        # size
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("num of instances is 0")

        # alignment labels
        if label_align_mapping is None:
            labels = data_instances.applyPartitions(
                lambda it: {item[1].label for item in it}
            ).reduce(lambda x, y: set.union(x, y))
            _, label_align_mapping = HomoLabelEncoderClient().label_alignment(labels)
            LOGGER.debug(f"label mapping: {label_align_mapping}")
        self._label_align_mapping = label_align_mapping

        # shape
        self.x_shape = data_instances.first()[1].features.shape
        self.x = np.zeros((self.size, *self.x_shape), dtype=np.float32)
        self.y = np.zeros((self.size,), dtype=expected_label_type)
        self._keys = []

        index = 0
        for key, instance in data_instances.collect():
            self._keys.append(key)
            self.x[index] = instance.features
            self.y[index] = label_align_mapping[instance.label]
            index += 1

        self._num_labels = len(label_align_mapping)
        self._num_features = self.x_shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), self.y[index]

    def __len__(self):
        return self.size


class VisionDataSet(torchvision.datasets.VisionDataset, DatasetMixIn):
    def get_num_labels(self):
        return None

    def get_num_features(self):
        return None

    def get_keys(self):
        return self._keys

    def as_data_instance(self):
        from federatedml.feature.instance import Instance

        def _as_instance(x):
            if isinstance(x, np.number):
                return Instance(label=x.tolist())
            else:
                return Instance(label=x)

        return computing_session.parallelize(
            data=zip(self._keys, map(_as_instance, self.targets)),
            include_key=True,
            partition=10,
        )

    def __init__(self, root, is_train=True, expected_label_type=np.float32, **kwargs):

        # fake alignment
        if is_train:
            HomoLabelEncoderClient().label_alignment(["fake"])

        # load data
        with open(os.path.join(root, "config.yaml")) as f:
            config = yaml.safe_load(f)
        if config["type"] == "vision":
            # read filenames
            with open(os.path.join(root, "filenames")) as f:
                file_names = [filename.strip() for filename in f]

            # read inputs
            if config["inputs"]["type"] != "images":
                raise TypeError("inputs type of vision type should be images")
            input_ext = config["inputs"]["ext"]
            self.images = [
                os.path.join(root, "images", f"{x}.{input_ext}") for x in file_names
            ]
            self._PIL_mode = config["inputs"].get("PIL_mode", "L")

            # read targets
            if config["targets"]["type"] == "images":
                target_ext = config["targets"]["ext"]
                self.targets = [
                    os.path.join(root, "targets", f"{x}.{target_ext}")
                    for x in file_names
                ]
                self.targets_is_image = True
            elif config["targets"]["type"] == "integer":
                with open(os.path.join(root, "targets")) as f:
                    targets_mapping = {}
                    for line in f:
                        filename, target = line.split(",")
                        targets_mapping[filename.strip()] = expected_label_type(
                            target.strip()
                        )
                    self.targets = [
                        targets_mapping[filename] for filename in file_names
                    ]
                self.targets_is_image = False
            self._keys = file_names

        else:
            raise TypeError(f"{config['type']}")

        assert len(self.images) == len(self.targets)

        transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize((100, 100)),
                torchvision.transforms.ToTensor(),
            ]
        )
        target_transform = None
        if self.targets_is_image:
            target_transform = transform

        super(VisionDataSet, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert(self._PIL_mode)
        if self.targets_is_image:
            target = Image.open(self.targets[index])
        else:
            target = self.targets[index]
        return self.transforms(img, target)

    def __len__(self):
        return len(self.images)
