import torch as t
import tqdm
from fate.arch import Context
from torch.utils.data import Dataset
from fate.ml.nn.trainer.hetero_trainer import HeteroNNTrainer
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost

class HeteroNNTrainerGuest(HeteroNNTrainer):

    def __init__(self,
                 ctx: Context,
                 model: HeteroNNModelGuest,
                 optimizer: t.optim.Optimizer,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 loss_func: t.nn.Module = None,
                 data_loader_worker: int = 4,
                 shuffle: bool = False,
                 epochs: int = 1,
                 batch_size: int = 16):

        super().__init__(ctx, model, optimizer, train_set, val_set, loss_func, data_loader_worker, shuffle,
                         epochs, batch_size)
        assert isinstance(self._model, HeteroNNModelGuest), 'Model should be a HeteroNNModelGuest instance.'
        self._loss_history = []
        self._model = model

    def train(self):

        self.on_train_begin(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set, self._val_set)
        has_guest_input = True

        if self._model._bottom_model is None:
            has_guest_input = False
        train_dataloader = t.utils.data.DataLoader(self._train_set, batch_size=self._batch_size,
                                                   shuffle=self._shuffle, num_workers=self._data_loader_worker)
        if self._val_set is not None:
            val_dataloader = t.utils.data.DataLoader(self._val_set, batch_size=self._batch_size,
                                                     shuffle=self._shuffle, num_workers=self._data_loader_worker)
        # set ctx
        self._model.set_context(self._ctx)

        for epoch in range(self._epochs):
            epoch_loss = 0
            self.on_epoch_begin(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set, self._val_set)
            for data in tqdm.tqdm(train_dataloader):
                self.on_batch_begin(self._ctx, self._model, self._optimizer,
                                    self._loss_func, self._train_set, self._val_set)

                self._optimizer.zero_grad()
                if isinstance(data, list) or isinstance(data, tuple):
                    if has_guest_input:
                        batch_data, batch_label = data
                        pred = self._model(batch_data)
                    else:
                        batch_label = data[0]
                        pred = self._model(batch_label)
                else:
                    batch_data = data
                    batch_label = None
                    pred = self._model(**batch_data)

                if not self._loss_func and hasattr(pred, "loss"):
                    batch_loss = pred.loss
                elif self._loss_func is not None:
                    if batch_label is None:
                        raise ValueError(
                            "When loss is set, please provide label to calculate loss"
                        )
                    if not isinstance(pred, t.Tensor) and hasattr(pred, "logits"):
                        pred = pred.logits
                    batch_loss = self._loss_func(pred, batch_label)
                else:
                    raise ValueError("loss_func is None and pred has no loss attribute, unable to calculate loss")

                epoch_loss += batch_loss.detach().numpy()

                self._model.backward(batch_loss)
                self._optimizer.step()

                self.on_batch_end(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set, self._val_set)
            self.on_epoch_end(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set, self._val_set)

    def predict(self, test_dataset: Dataset):
        pass


class HeteroNNTrainerHost(HeteroNNTrainer):

    def __init__(self,
                 ctx: Context,
                 model: HeteroNNModelHost,
                 optimizer: t.optim.Optimizer,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 loss_func: t.nn.Module = None,
                 data_loader_worker: int = 4,
                 shuffle: bool = False,
                 epochs: int = 1,
                 batch_size: int = 16):

        super().__init__(ctx, model, optimizer, train_set, val_set, loss_func, data_loader_worker, shuffle,
                         epochs, batch_size)

    def train(self):

        self.on_train_begin(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set, self._val_set)
        train_dataloader = t.utils.data.DataLoader(self._train_set, batch_size=self._batch_size,
                                                   shuffle=self._shuffle, num_workers=self._data_loader_worker)
        if self._val_set is not None:
            val_dataloader = t.utils.data.DataLoader(self._val_set, batch_size=self._batch_size,
                                                     shuffle=self._shuffle, num_workers=self._data_loader_worker)
        # set ctx
        self._model.set_context(self._ctx)
        for epoch in range(self._epochs):
            self.on_epoch_begin(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set,
                                self._val_set)
            for data in tqdm.tqdm(train_dataloader):
                self.on_batch_begin(self._ctx, self._model, self._optimizer,
                                    self._loss_func, self._train_set, self._val_set)

                self._optimizer.zero_grad()
                if isinstance(data, list) or isinstance(data, tuple):
                    batch_data = data[0]
                    self._model(batch_data)
                else:
                    batch_data = data
                    self._model(**batch_data)
                self._model.backward()
                self._optimizer.step()
                self.on_batch_end(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set,
                                  self._val_set)
            self.on_epoch_end(self._ctx, self._model, self._optimizer, self._loss_func, self._train_set, self._val_set)

    def predict(self, test_dataset: Dataset):
        pass