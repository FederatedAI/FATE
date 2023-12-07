#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
import random
import shutil
import tempfile

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from .utils import NoopContextManager
from torchvision import datasets, transforms

import logging
import time

import torch

from fate.arch import Context
from . import MPCModule
from .meters import AverageMeter


class Cifar(MPCModule):
    def __init__(
        self,
        epochs=25,
        start_epoch=0,
        batch_size=1,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-6,
        print_freq=1000,
        model_location="",
        resume=False,
        evaluate=True,
        seed=None,
        skip_plaintext=False,
        context_manager=None,
    ):
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.model_location = model_location
        self.resume = resume
        self.evaluate = evaluate
        self.seed = seed
        self.skip_plaintext = skip_plaintext
        self.context_manager = context_manager

    def fit(self, ctx: Context):
        ...

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        # create model
        model = LeNet()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )

        # optionally resume from a checkpoint
        best_prec1 = 0
        if self.resume:
            if os.path.isfile(self.model_location):
                logging.info("=> loading checkpoint '{}'".format(self.model_location))
                checkpoint = torch.load(self.model_location)
                start_epoch = checkpoint["epoch"]
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info("=> loaded checkpoint '{}' (epoch {})".format(self.model_location, checkpoint["epoch"]))
            else:
                raise IOError("=> no checkpoint found at '{}'".format(self.model_location))

        # Data loading code
        def preprocess_data(context_manager, data_dirname):
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            with context_manager:
                trainset = datasets.CIFAR10(data_dirname, train=True, download=True, transform=transform)
                testset = datasets.CIFAR10(data_dirname, train=False, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
            return trainloader, testloader

        if self.context_manager is None:
            context_manager = NoopContextManager()

        # data_dir = tempfile.TemporaryDirectory()
        data_dir = f"data/cifar10/{ctx.local.rank}"
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = preprocess_data(context_manager, data_dir)

        if self.evaluate:
            if not self.skip_plaintext:
                logging.info("===== Evaluating plaintext LeNet network =====")
                validate(ctx, val_loader, model, criterion, self.print_freq)
            logging.info("===== Evaluating Private LeNet network =====")
            input_size = get_input_size(val_loader, self.batch_size)
            private_model = construct_private_model(ctx.local.rank, input_size, model)
            validate(ctx, val_loader, private_model, criterion, self.print_freq)
            # logging.info("===== Validating side-by-side ======")
            # validate_side_by_side(val_loader, model, private_model)
            return

        # define loss function (criterion) and optimizer
        for epoch in range(start_epoch, self.epochs):
            adjust_learning_rate(optimizer, epoch, self.lr)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, self.print_freq)

            # evaluate on validation set
            prec1 = validate(ctx, val_loader, model, criterion, self.print_freq)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(
                ctx,
                {
                    "epoch": epoch + 1,
                    "arch": "LeNet",
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )
        data_dir.cleanup()


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.add(loss.item(), input.size(0))
        top1.add(prec1[0], input.size(0))
        top5.add(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        current_batch_time = time.time() - end
        batch_time.add(current_batch_time)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                "Epoch: [{}][{}/{}]\t"
                "Time {:.3f} ({:.3f})\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 {:.3f} ({:.3f})\t"
                "Prec@5 {:.3f} ({:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    current_batch_time,
                    batch_time.value(),
                    loss.item(),
                    losses.value(),
                    prec1[0],
                    top1.value(),
                    prec5[0],
                    top5.value(),
                )
            )


def validate_side_by_side(ctx: Context, val_loader, plaintext_model, private_model):
    """Validate the plaintext and private models side-by-side on each example"""
    # switch to evaluate mode
    plaintext_model.eval()
    private_model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output for plaintext
            output_plaintext = plaintext_model(input)
            # encrypt input and compute output for private
            # assumes that private model is encrypted with src=0
            input_encr = encrypt_data_tensor_with_src(ctx, input)
            output_encr = private_model(input_encr)
            # log all info
            logging.info("==============================")
            logging.info("Example %d\t target = %d" % (i, target))
            logging.info("Plaintext:\n%s" % output_plaintext)
            logging.info("Encrypted:\n%s\n" % output_encr.get_plain_text())
            # only use the first 1000 examples
            if i > 1000:
                break


def get_input_size(val_loader, batch_size):
    input, target = next(iter(val_loader))
    return input.size()


def construct_private_model(rank, input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = LeNet()

    from fate.arch.tensor.mpc import nn

    private_model = nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model


def encrypt_data_tensor_with_src(ctx: Context, input):
    if ctx.world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if ctx.local.rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())

    private_input = ctx.mpc.cryptensor(input_upd, src=src_id)
    return private_input


def validate(ctx: Context, val_loader, model, criterion, print_freq=10):
    from fate.arch.tensor.mpc import nn

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, nn.Module) and not ctx.mpc.is_encrypted_tensor(input):
                input = encrypt_data_tensor_with_src(ctx, input)
            # compute output
            output = model(input)
            if ctx.mpc.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top5.add(prec5[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec5[0],
                        top5.value(),
                    )
                )

        logging.info(" * Prec@1 {:.3f} Prec@5 {:.3f}".format(top1.value(), top5.value()))
    return top1.value()


def save_checkpoint(ctx: Context, state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint of plaintext model"""
    # only save from rank 0 process to avoid race condition
    if ctx.local.rank == 0:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, "model_best.pth.tar")


def adjust_learning_rate(optimizer, epoch, lr=0.01):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
