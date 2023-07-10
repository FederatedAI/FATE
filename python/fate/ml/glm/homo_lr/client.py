from typing import Optional, Union
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HomoModule, Model, Module
from fate.arch import Context
import logging
import pandas as pd
import torch as t
from fate.ml.nn.algo.homo.fedavg import FedAVGCLient, FedAVGServer, TrainingArguments, FedAVGArguments
from torch.utils.data import TensorDataset
import numpy as np
from torch.nn import functional as F
import functools


logger = logging.getLogger(__name__)


class Data(object):

    def __init__(self, features: pd.DataFrame, sample_ids: pd.DataFrame, match_ids: pd.DataFrame, labels: pd.DataFrame) -> None:
        # set var
        self.features = features
        self.sample_ids = sample_ids
        self.match_ids = match_ids
        self.labels = labels

    @staticmethod
    def from_fate_dataframe(df: DataFrame):
        schema = df.schema
        sample_id = schema.sample_id_name
        match_id = schema.match_id_name
        label = schema.label_name
        pd_df = df.as_pd_df()
        features = pd_df.drop([sample_id, match_id, label], axis=1)
        sample_ids = pd_df[[sample_id]]
        match_ids = pd_df[[match_id]]
        labels = pd_df[[label]]
        return Data(features, sample_ids, match_ids, labels)


class HomoLRModel(t.nn.Module):

    def __init__(self, feature_num, label_num=2) -> None:
        super().__init__()
        assert feature_num >= 2 and isinstance(feature_num, int), "feature_num must be int greater than 2"
        assert label_num >= 1 and isinstance(label_num, int), "label_num must be int greater than 1"
        self.models = t.nn.ModuleList()

        if label_num <= 2 and label_num > 0:
            self.models.append(
                t.nn.Linear(feature_num, 1)
            )
        else:
            # OVR Setting
            for i in range(label_num):
                self.models.append(
                    t.nn.Linear(feature_num, 1)
                )
        self.sigmoid = t.nn.Sigmoid()
        self.softmax = t.nn.Softmax(dim=1)

    def forward(self, x):

        if len(self.models) == 1:
            linear_out = self.models[0](x)
        else:
            linear_out = t.cat([model(x) for model in self.models], dim=1)

        linear_out = self.sigmoid(linear_out).reshape((-1, len(self.models)))

        if not self.training:
            prob = self.softmax(linear_out)
            return prob
        else:
            return linear_out
        
    def to_dict(self):
        model_dict = {
            "feature_num": self.models[0].in_features,
            "label_num": len(self.models),
            "state_dict": {k: v.tolist() for k, v in self.state_dict().items()}  # convert tensor to list
        }
        return model_dict

    @classmethod
    def from_dict(cls, model_dict):
        model = cls(model_dict["feature_num"], model_dict["label_num"])
        model_state_dict = {k: t.tensor(v) for k, v in model_dict["state_dict"].items()}  # convert list back to tensor
        model.load_state_dict(model_state_dict)
        return model


def homo_lr_loss(pred, labels, dim=1):
    """
    The function assumes that pred has shape (n, num_classes) where each class has its own linear model.
    labels have shape (n,) and the values are integers denoting the class.
    """
    
    # initialize the loss
    loss = 0.0
    if dim == 2:
        dim -= 1

    loss_fn = t.nn.BCELoss()

    for c in range(dim):
        # get binary labels for this class
        binary_labels = (labels == c).float().flatten()
        bin_pred = pred[:, c].flatten()
        # compute binary cross-entropy loss
        loss = loss_fn(bin_pred, binary_labels)
    # normalize loss by the number of classes
    loss /= dim

    return loss


def optimizer_to_dict(optimizer):
    # Convert the optimizer state to a dictionary that can be transformed to JSON
    optimizer_dict = {
        "state": {k: v.tolist() for k, v in optimizer.state_dict()['state'].items()},
        "param_groups": optimizer.state_dict()['param_groups'],
    }
    return optimizer_dict


class HomoLRClient(HomoModule):

    def __init__(self, max_iter: int, batch_size: int, optimizer_param=None,
                learning_rate_param=None,
                init_param=None,
                threshold=0.5
                ) -> None:
        
        super().__init__()
        self.df_schema = None
        self.train_data = None
        self.validate_data = None
        self.predict_data = None

        # set vars
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer_param = optimizer_param
        self.learning_rate_param = learning_rate_param
        self.init_param = init_param
        self.threshold = threshold
        self.run_ovr = False
        self.train_feature_num = None
        self.validate_feature_num = None
        
        # models & optimizer & schduler
        self.model = None
        self.optimizer = None
        self.scheduler = None

        # checkping param
        assert self.max_iter > 0 and isinstance(self.max_iter, int), "max_iter must be int greater than 0"
        assert self.batch_size > 0 and isinstance(self.batch_size, int), "batch_size must be int greater than 0"
        assert self.threshold > 0 and self.threshold < 1, "threshold must be float between 0 and 1"
    

    def _make_dataset(self, data: Data):

        X = np.array(data.features.values).astype(np.float32)
        y = np.array(data.labels.values).astype(np.float32)
        X_tensor = t.tensor(X, dtype=t.float32)
        y_tensor = t.tensor(y.reshape((-1, 1)), dtype=t.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:

        self.train_data = Data.from_fate_dataframe(train_data)
        self.train_feature_num = self.train_data.features.values.shape[1]
        if validate_data is not None:
            self.validate_data = Data.from_fate_dataframe(validate_data)
            self.validate_feature_num = self.validate_data.features.values.shape[1]
            assert self.train_feature_num == self.validate_feature_num, "train and validate feature num not match: {} vs {}".format(self.train_feature_num, self.validate_feature_num)

        unique_label_set = set(self.train_data.labels.values.reshape(-1))
        if validate_data is not None:
            unique_label_set = unique_label_set.union(set(self.validate_data.labels.values.reshape(-1)))
            logger.info("unique label set updated to: {}".format(unique_label_set))

        train_set = self._make_dataset(self.train_data)

        if self.validate_data is not None:
            validate_set = self._make_dataset(self.validate_data)
        else:
            validate_set = None

        loss_fn = functools.partial(homo_lr_loss, dim=len(unique_label_set))

        model = HomoLRModel(self.train_feature_num, label_num=len(unique_label_set))
        self.model = model
        logger.info('model structure is {}'.format(model))

        optimizer = t.optim.SGD(model.parameters(), lr=self.learning_rate_param)
        self.optimizer = optimizer
        # training
        fed_arg = FedAVGArguments()
        train_arg = TrainingArguments(num_train_epochs=self.max_iter, 
                                      per_device_train_batch_size=self.batch_size, per_gpu_eval_batch_size=self.batch_size)
        trainer = FedAVGCLient(ctx, model=model, loss_fn=loss_fn, optimizer=optimizer, train_set=train_set, 
                               val_set=validate_set, training_args=train_arg, fed_args=fed_arg)
        
        # !!!!!!!!!!
        # TODO
        # !!!!!!!!!!
        trainer.set_local_mode()
        trainer.train()
        
    
    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        return super().predict(ctx, predict_data)
    
    def get_model(self) -> dict:
        param = {}
        if self.model is not None:
            param['model'] = self.model.to_dict()
        if self.optimizer is not None:
            param['optimizer'] = optimizer_to_dict(self.optimizer)
        
        meta = {'batch_size': self.batch_size, 'max_iter': self.max_iter, 'threshold': self.threshold, 
                'optimizer_param': self.optimizer_param, 'learning_rate_param': self.learning_rate_param, 'init_param': self.init_param}
        ret = {'meta': meta, 'param': param}

        return ret
    
    @classmethod
    def from_model(cls, model: dict) -> Module:
        if not hasattr(model, 'model'):
            raise ('key "param" is not found in the input model dict')
        param = model['param']
        if not hasattr(param, 'model'):
            raise ValueError("param dict must have key 'model' that contains the model parameter and structure info") 
        


