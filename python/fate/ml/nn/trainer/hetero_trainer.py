import torch as t
from fate.arch import Context
from torch.utils.data import Dataset

class HeteroNNTrainer(object):

    def __init__(self,
                 ctx: Context,
                 model: t.nn.Module,
                 optimizer: t.optim.Optimizer,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 loss_func: t.nn.Module = None,
                 data_loader_worker: int = 4,
                 shuffle: bool = False,
                 epochs: int = 1,
                 batch_size: int = 16,
                 ):
        self._ctx = ctx
        self._model = model
        self._optimizer = optimizer
        self._train_set = train_set
        self._val_set = val_set
        self._loss_func = loss_func
        self._data_loader_worker = data_loader_worker
        self._epochs = epochs
        self._batch_size = batch_size
        self._shuffle = shuffle

    def _prepare_model(self):
        pass

    def on_train_begin(self, ctx: Context, model: t.nn.Module, optimizer: t.optim.Optimizer, loss_func: t.nn.Module,
                        train_set: Dataset, val_set: Dataset):
        pass

    def on_train_end(self, ctx: Context, model: t.nn.Module, optimizer: t.optim.Optimizer, loss_func: t.nn.Module,
                        train_set: Dataset, val_set: Dataset):
        pass

    def on_epoch_begin(self, ctx: Context, model: t.nn.Module, optimizer: t.optim.Optimizer, loss_func: t.nn.Module,
                        train_set: Dataset, val_set: Dataset):
        pass

    def on_epoch_end(self, ctx: Context, model: t.nn.Module, optimizer: t.optim.Optimizer, loss_func: t.nn.Module,
                        train_set: Dataset, val_set: Dataset):
        pass

    def on_batch_begin(self, ctx: Context, model: t.nn.Module, optimizer: t.optim.Optimizer, loss_func: t.nn.Module,
                        train_set: Dataset, val_set: Dataset):
        pass

    def on_batch_end(self, ctx: Context, model: t.nn.Module, optimizer: t.optim.Optimizer, loss_func: t.nn.Module,
                        train_set: Dataset, val_set: Dataset):
        pass
    def train(self):
        pass

    def predict(self, test_dataset: Dataset):
        pass