import torch.nn as nn
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HomoModule
from fate.ml.utils.model_io import ModelIO
from fate.arch import Context
import logging
import pandas as pd
import torch as t
from fate.ml.nn.algo.homo.fedavg import FedAVGCLient, TrainingArguments, FedAVGArguments
from transformers import default_data_collator
import numpy as np
from torch.nn import functional as F
import functools
import tempfile
from torch.utils.data import Dataset


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
        logger.info('columns are {} {} {}'.format(sample_id, match_id, label))
        pd_df = df.as_pd_df()
        features = pd_df.drop([sample_id, match_id, label], axis=1)
        sample_ids = pd_df[[sample_id]]
        match_ids = pd_df[[match_id]]
        labels = pd_df[[label]]
        return Data(features, sample_ids, match_ids, labels)


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


class HomoLRModel(t.nn.Module):

    def __init__(self, feature_num, label_num=2, l1=0) -> None:
        super().__init__()
        assert feature_num >= 2 and isinstance(feature_num, int), "feature_num must be int greater than 2"
        assert label_num >= 1 and isinstance(label_num, int), "label_num must be int greater than 1"
        self.models = t.nn.ModuleList()

        if 2 >= label_num > 0:
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
        self.l1 = l1

    def forward(self, x, labels=None):

        if len(self.models) == 1:
            linear_out = self.models[0](x)
        else:
            linear_out = t.cat([model(x) for model in self.models], dim=1)

        ret_dict = {}
        linear_out = self.sigmoid(linear_out).reshape((-1, len(self.models)))

        if not self.training:
            if len(self.models) > 1:
                linear_out = self.softmax(linear_out)

        ret_dict['pred'] = linear_out
        
        if labels is not None:
            loss = homo_lr_loss(linear_out, labels, dim=len(self.models))
            if self.l1 != 0:
                l1_regularization = t.tensor(0.)
                for param in self.models.parameters():
                    l1_regularization += t.norm(param, 1)
                loss += self.l1 * l1_regularization
            ret_dict['loss'] = loss

        return ret_dict
        
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


def init_model(model, method='random', val=1.0):
    if method == 'zeros':
        init_fn = nn.init.zeros_
    elif method == 'ones':
        init_fn = nn.init.ones_
    elif method == 'consts':
        init_fn = lambda x: nn.init.constant_(x, val)
    elif method == 'random':
        init_fn = nn.init.normal_
    else:
        raise ValueError("Invalid method. Options are: 'zeros', 'ones', 'consts', 'random'")
    
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.zeros_(param)  # usually it's good practice to initialize biases to zero
        else:
            init_fn(param)


# read model from model bytes
def recover_torch_bytes(model_bytes):

    with tempfile.TemporaryFile() as f:
        f.write(model_bytes)
        f.seek(0)
        model_dict = t.load(f)

    return model_dict


def get_torch_bytes(model_dict):

    with tempfile.TemporaryFile() as f:
        t.save(model_dict, f)
        f.seek(0)
        model_saved_bytes = f.read()

        return model_saved_bytes


class DictDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data):
        self.X = np.array(data.features.values).astype(np.float32)
        self.y = np.array(data.labels.values).astype(np.float32)
        self.X_tensor = t.tensor(self.X, dtype=t.float32)
        self.y_tensor = t.tensor(self.y.reshape((-1, 1)), dtype=t.float32)

    def __getitem__(self, index):
        return {'x': self.X_tensor[index], 'label': self.y_tensor[index]}

    def __len__(self):
        return self.X_tensor.shape[0]


class HomoLRClient(HomoModule):

    def __init__(self, epochs: int=5, batch_size: int=32, optimizer_param=None,
                learning_rate_scheduler=None,
                init_param=None,
                threshold: float=0.5
                ) -> None:
        
        super().__init__()
        self.df_schema = None
        self.train_data = None
        self.validate_data = None
        self.predict_data = None

        # set vars
        self.max_iter = epochs
        self.batch_size = batch_size
        self.optimizer_param = optimizer_param
        self.learning_rate_param = learning_rate_scheduler
        self.init_param = init_param
        self.threshold = threshold
        self.run_ovr = False
        self.train_feature_num = None
        self.validate_feature_num = None
        
        # models & optimizer & schduler
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.optimizer_state_dict = None
        self.trainer = None

        # loaded meta
        self.loaded_meta = None

        # l1 & l2
        self.l1 = 0
        self.l2 = 0

        # checkping param
        assert self.max_iter > 0 and isinstance(self.max_iter, int), "max_iter must be int greater than 0"
        assert self.batch_size > 0 and isinstance(self.batch_size, int), "batch_size must be int greater than 0"
        assert self.threshold > 0 and self.threshold < 1, "threshold must be float between 0 and 1"
    
    def _make_dataset(self, data: Data):
        return DictDataset(data)

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

        # prepare loss function
        loss_fn = functools.partial(homo_lr_loss, dim=len(unique_label_set))

        # initialize model
        if self.model is None:

            self.model = HomoLRModel(self.train_feature_num, label_num=len(unique_label_set), l1=self.l1)

            # init model here
            init_model(self.model)

            logger.info('model initialized')
            logger.info('model parameters are {}'.format(list(self.model.parameters())))
        else:
            logger.info('model is loaded')
        logger.info('model structure is {}'.format(self.model))

        # initialize optimizer
        self.optimizer = t.optim.SGD(self.model.parameters(), lr=self.learning_rate_param, weight_decay=self.l2)  
        if self.optimizer_state_dict is not None:
            optimizer_state_dict = {
                "state": {k: t.tensor(v) for k, v in self.optimizer_state_dict['state'].items()},
                "param_groups": self.optimizer_state_dict['param_groups'],
            }
            self.optimizer.load_state_dict(optimizer_state_dict)
            logger.info('load warmstart optimizer state dict')
        # training
        fed_arg = FedAVGArguments()
        train_arg = TrainingArguments(num_train_epochs=self.max_iter, 
                                      per_device_train_batch_size=self.batch_size, per_gpu_eval_batch_size=self.batch_size)
        self.trainer = FedAVGCLient(ctx, model=self.model, loss_fn=loss_fn, optimizer=self.optimizer, train_set=train_set, 
                               val_set=validate_set, training_args=train_arg, fed_args=fed_arg, data_collator=default_data_collator)
        self.trainer.train()
        
    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        
        if self.model is None:
            raise ValueError("model is not initialized")
        self.predict_data = Data.from_fate_dataframe(predict_data)
        predict_set = self._make_dataset(self.predict_data)
        if self.trainer is None:
            train_arg = TrainingArguments(num_train_epochs=self.max_iter, per_device_eval_batch_size=self.batch_size)
            trainer = FedAVGCLient(ctx, train_set=predict_set, model=self.model, training_args=train_arg, 
                                        fed_args=FedAVGArguments(), data_collator=default_data_collator)
            trainer.set_local_mode()
        else:
            trainer = self.trainer
        predict_rs = trainer.predict(predict_set)
        rs = {"predict_score": predict_rs.predictions, 'label': predict_rs.label_ids}
        return rs

    def get_model(self) -> ModelIO:
        param = {}
        if self.model is not None:
            param['model'] = self.model.to_dict()
        if self.optimizer is not None:
            param['optimizer'] = get_torch_bytes(self.optimizer.state_dict())
        
        meta = {'batch_size': self.batch_size, 'max_iter': self.max_iter, 'threshold': self.threshold, 
                'optimizer_param': self.optimizer_param, 'learning_rate_param': self.learning_rate_param, 'init_param': self.init_param}
        export_ = ModelIO(data=param, meta=meta)

        return export_

    def from_model(self, model: ModelIO):

        model = model.dict()
        if not 'data' in model:
            raise ('key "data" is not found in the input model dict')
        
        model_param = model['data']
        if not 'model' in model_param:
            raise ValueError("param dict must have key 'model' that contains the model parameter and structure info")
        self.model = HomoLRModel.from_dict(model_param['model'])
        self.model.l1 = self.l1
        if hasattr(model_param, 'optimizer'):
            self.optimizer_state_dict = recover_torch_bytes(model_param['optimizer'])
        self.loaded_meta = model['meta']


