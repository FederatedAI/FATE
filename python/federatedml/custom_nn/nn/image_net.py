import torch as t
import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from federatedml.custom_nn.nn_base_module import NNBaseModule
from federatedml.util import LOGGER
from federatedml.util import consts
from torchvision.datasets import ImageFolder
from torchvision import transforms
import tqdm
from sklearn.metrics import roc_auc_score
import torchvision as tv


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.imagenet = nn.Sequential(
            nn.Conv2d(3, 8, (8, 8), (1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, padding=0, dilation=1, ceil_mode=False),
            nn.AdaptiveAvgPool2d(output_size=(8, 8))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        image_feature = self.imagenet(image).flatten(start_dim=1)
        return self.classifier(image_feature)


class ConvNNModule(NNBaseModule):

    def __init__(self):
        super(ConvNNModule, self).__init__()
        self.model = None

    def train(self, cpn_input, **kwargs):

        LOGGER.debug('input data is {}'.format(cpn_input))

        epochs = kwargs['epochs']
        lr = kwargs['lr']
        test_batch_size = kwargs['batch_size']

        preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size=256),
            transforms.ToTensor()
        ])

        if self.role == consts.GUEST:
            train_set = ImageFolder(
                '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/dogs-vs-cat-small/train',
                transform=preprocess)
        elif self.role == consts.HOST:
            train_set = ImageFolder(
                '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/dogs-vs-cat-small/test',
                transform=preprocess)
        else:
            raise ValueError('error role')
        dl = DataLoader(train_set, batch_size=test_batch_size, shuffle=False)

        # set training communication round
        self.set_fed_avg_round_num(comm_round=epochs)

        self.model = Net()
        optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        loss_func = t.nn.BCELoss()

        for i in range(epochs):
            LOGGER.debug('epoch is {}'.format(i))
            epoch_loss = 0
            batch_idx = 0
            pred_scores = []
            labels = []
            for image, label in tqdm.tqdm(dl):
                LOGGER.debug('running batch {}'.format(batch_idx))
                optimizer.zero_grad()
                pred = self.model(image)

                pred_scores.append(pred.detach().numpy())
                labels.append(label.numpy())

                batch_loss = loss_func(pred.flatten(), label.type(t.float32).flatten())
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().numpy()
                batch_idx += 1
            epoch_loss = epoch_loss / batch_idx
            # federation process
            LOGGER.debug('epoch loss is {}'.format(epoch_loss))
            self.fed_avg_model(optimizer, loss=epoch_loss, loss_weight=len(train_set))
            y_true = np.concatenate(labels, axis=0)
            y_pred = np.concatenate(pred_scores, axis=0)
            LOGGER.debug('train auc is {}'.format(roc_auc_score(y_true, y_pred)))

    def predict(self, cpn_input, **kwargs):
        pass


if __name__ == '__main__':

    model = ConvNNModule()
    model.set_role(consts.HOST)
    model.local_mode()
    params = {'epochs': 10, 'batch_size': 256, 'lr': 0.0001}
    model.train(None, **params)

