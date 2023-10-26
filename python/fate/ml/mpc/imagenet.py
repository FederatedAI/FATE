#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import tempfile

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from .meters import AccuracyMeter
from .utils import NoopContextManager
from fate.arch.tensor import mpc
from . import MPCModule
from fate.arch.context import Context


try:
    from fate.arch.tensor.nn.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed
    SummaryWriter = None


class ImageNet(MPCModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        imagenet_folder="/Users/sage/Downloads/imagenet",
        tensorboard_folder="/tmp",
        num_samples=None,
        context_manager=None,
    ):
        self.model_name = model_name
        self.imagenet_folder = imagenet_folder
        self.tensorboard_folder = tensorboard_folder
        self.num_samples = num_samples
        self.context_manager = context_manager

    def fit(
        self,
        ctx: Context,
    ):
        """Runs inference using specified vision model on specified dataset."""

        # check inputs:
        assert hasattr(models, self.model_name), "torchvision does not provide %s model" % self.model_name
        if self.imagenet_folder is None:
            self.imagenet_folder = tempfile.gettempdir()
            download = True
        else:
            download = True
        if self.context_manager is None:
            self.context_manager = NoopContextManager()

        # load dataset and model:
        with self.context_manager:
            model = getattr(models, self.model_name)(pretrained=True)
            model.eval()
            dataset = datasets.ImageNet(self.imagenet_folder, split="val", download=download)

        # define appropriate transforms:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        to_tensor_transform = transforms.ToTensor()

        # encrypt model:
        dummy_input = to_tensor_transform(dataset[0][0])
        dummy_input.unsqueeze_(0)
        encrypted_model = mpc.nn.from_pytorch(model, dummy_input=dummy_input)
        encrypted_model.encrypt()

        # show encrypted model in tensorboard:
        if SummaryWriter is not None:
            writer = SummaryWriter(log_dir=self.tensorboard_folder)
            writer.add_graph(encrypted_model)
            writer.close()

        # loop over dataset:
        meter = AccuracyMeter()
        for idx, sample in enumerate(dataset):
            # preprocess sample:
            image, target = sample
            image = transform(image)
            image.unsqueeze_(0)
            target = torch.tensor([target], dtype=torch.long)

            # perform inference using encrypted model on encrypted sample:
            encrypted_image = mpc.cryptensor(image)
            encrypted_output = encrypted_model(encrypted_image)

            # measure accuracy of prediction
            output = encrypted_output.get_plain_text()
            meter.add(output, target)

            # progress:
            logging.info("[sample %d of %d] Accuracy: %f" % (idx + 1, len(dataset), meter.value()[1]))
            if self.num_samples is not None and idx == self.num_samples - 1:
                break

        # print final accuracy:
        logging.info("Accuracy on all %d samples: %f" % (len(dataset), meter.value()[1]))
