import logging
import time

import torch

from fate.arch import Context
from . import MPCModule
from .meters import AverageMeter


class SVM(MPCModule):
    def __init__(self, epochs=50, examples=50, features=100, lr=0.5, skip_plaintext=False):
        self.epochs = epochs
        self.examples = examples
        self.features = features
        self.lr = lr
        self.skip_plaintext = skip_plaintext

    def fit(self, ctx: Context):
        # Set random seed for reproducibility
        torch.manual_seed(1)

        # Initialize x, y, w, b
        x = torch.randn(self.features, self.examples)
        w_true = torch.randn(1, self.features)
        b_true = torch.randn(1)
        y = w_true.matmul(x) + b_true
        y = y.sign()

        if not self.skip_plaintext:
            logging.info("==================")
            logging.info("PyTorch Training")
            logging.info("==================")
            w_torch, b_torch = train_linear_svm(ctx, x, y, lr=self.lr, print_time=True)

        # Encrypt features / labels
        x = ctx.mpc.cryptensor(x)
        y = ctx.mpc.cryptensor(y)

        logging.info("==================")
        logging.info("CrypTen Training")
        logging.info("==================")
        w, b = train_linear_svm(ctx, x, y, lr=self.lr, print_time=True)

        if not self.skip_plaintext:
            logging.info("PyTorch Weights  :")
            logging.info(w_torch)
        logging.info("CrypTen Weights:")
        logging.info(w.get_plain_text())

        if not self.skip_plaintext:
            logging.info("PyTorch Bias  :")
            logging.info(b_torch)
        logging.info("CrypTen Bias:")
        logging.info(b.get_plain_text())


def train_linear_svm(ctx: Context, features, labels, epochs=50, lr=0.5, print_time=False):
    # Initialize random weights
    w = features.new(torch.randn(1, features.size(0)))
    b = features.new(torch.randn(1))

    if print_time:
        pt_time = AverageMeter()
        end = time.time()

    for epoch in range(epochs):
        # Forward
        label_predictions = w.matmul(features).add(b).sign()

        # Compute accuracy
        correct = label_predictions.mul(labels)
        accuracy = correct.add(1).div(2).mean()
        if ctx.mpc.is_encrypted_tensor(accuracy):
            accuracy = accuracy.get_plain_text()

        # Print Accuracy once
        if ctx.mpc.communicator.get_rank() == 0:
            print(f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100))

        # Backward
        loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss
        b_grad = loss_grad.mean()
        w_grad = loss_grad.matmul(features.t()).div(loss_grad.size(1))

        # Update
        w -= w_grad * lr
        b -= b_grad * lr

        if print_time:
            iter_time = time.time() - end
            pt_time.add(iter_time)
            logging.info("    Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            end = time.time()

    return w, b


def evaluate_linear_svm(ctx, features, labels, w, b):
    """Compute accuracy on a test set"""
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if ctx.communicator.get().get_rank() == 0:
        print("Test accuracy %.2f%%" % (accuracy.item() * 100))
