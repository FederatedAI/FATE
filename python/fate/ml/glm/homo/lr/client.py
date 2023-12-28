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

import torch.nn as nn
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HomoModule
from fate.ml.utils.model_io import ModelIO
from fate.arch import Context
import logging
import torch as t
from fate.ml.nn.homo.fedavg import FedAVGClient, TrainingArguments, FedAVGArguments
from transformers import default_data_collator
import functools
import tempfile
from fate.ml.utils.predict_tools import array_to_predict_df
from fate.ml.utils.predict_tools import MULTI, BINARY
from fate.ml.nn.dataset.table import TableDataset
from fate.ml.utils._optimizer import optimizer_factory, lr_scheduler_factory


logger = logging.getLogger(__name__)


def homo_lr_loss(pred, labels, dim=1):
    """
    The function assumes that pred has shape (n, num_classes) where each class has its own linear model.
    labels have shape (n,) and the values are integers denoting the class.
    """

    # initialize the loss
    loss = 0.0
    loss_fn = t.nn.BCELoss()
    if dim <= 2:
        return loss_fn(pred[:, 0], labels)

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
    def __init__(self, feature_num, label_num=2, l1=0, bias=True) -> None:
        super().__init__()
        assert feature_num >= 2 and isinstance(feature_num, int), "feature_num must be int greater than 2"
        assert label_num >= 1 and isinstance(label_num, int), "label_num must be int greater than 1"
        self.models = t.nn.ModuleList()

        if 2 >= label_num > 0:
            self.models.append(t.nn.Linear(feature_num, 1, bias=bias))
        else:
            # OVR Setting
            for i in range(label_num):
                self.models.append(t.nn.Linear(feature_num, 1, bias=bias))
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

        ret_dict["pred"] = linear_out

        if labels is not None:
            loss = homo_lr_loss(linear_out, labels, dim=len(self.models))
            if self.l1 != 0:
                l1_regularization = t.tensor(0.0)
                for param in self.models.parameters():
                    l1_regularization += t.norm(param, 1)
                loss += self.l1 * l1_regularization
            ret_dict["loss"] = loss

        return ret_dict

    def to_dict(self):
        model_dict = {
            "feature_num": self.models[0].in_features,
            "label_num": len(self.models),
            # convert tensor to list
            "state_dict": {k: v.tolist() for k, v in self.state_dict().items()},
        }
        return model_dict

    @classmethod
    def from_dict(cls, model_dict):
        model = cls(model_dict["feature_num"], model_dict["label_num"])
        model_state_dict = {k: t.tensor(v) for k, v in model_dict["state_dict"].items()}  # convert list back to tensor
        model.load_state_dict(model_state_dict)
        return model


def init_model(model, method="random", fill_val=1.0):
    if method == "zeros":
        init_fn = nn.init.zeros_
    elif method == "ones":
        init_fn = nn.init.ones_
    elif method == "consts":

        def init_fn(x):
            return nn.init.constant_(x, fill_val)

    elif method == "random":
        init_fn = nn.init.normal_
    else:
        raise ValueError("Invalid method. Options are: 'zeros', 'ones', 'consts', 'random'")

    for name, param in model.named_parameters():
        if "bias" in name:
            # usually it's good practice to initialize biases to zero
            nn.init.zeros_(param)
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


def update_params(new_params, default, name="optimizer"):
    import copy

    params = copy.deepcopy(default)
    if not isinstance(new_params, dict):
        raise ValueError("{} param dict must be a dict but got {}".format(name, new_params))

    def _update(default, new):
        for key in new.keys():
            if key in default:
                default[key] = new[key]

    _update(params, new_params)

    return params


DEFAULT_OPT_PARAM = {
    "method": "sgd",
    "penalty": "l2",
    "alpha": 0.0,
    "optimizer_params": {"lr": 0.01, "weight_decay": 0},
}
DEFAULT_INIT_PARAM = {"method": "random", "fill_val": 1.0, "fit_intercept": True}
DEFAULT_LR_SCHEDULER_PARAM = {"method": "constant", "scheduler_params": {"factor": 1.0}}


class HomoLRClient(HomoModule):
    def __init__(
        self,
        epochs: int = 5,
        batch_size: int = None,
        optimizer_param={"method": "sgd", "optimizer_params": {"lr": 0.01, "weight_decay": 0}},
        learning_rate_scheduler={"method": "constant", "scheduler_params": {"factor": 1.0}},
        init_param={"method": "random", "fill_val": 1.0, "fit_intercept": True},
        threshold: float = 0.5,
        ovr=False,
        label_num=None,
    ) -> None:
        super().__init__()
        self.df_schema = None
        self.train_set = None
        self.validate_set = None
        self.predict_set = None

        # set vars
        self.max_iter = epochs
        self.batch_size = batch_size
        self.optimizer_param = update_params(optimizer_param, DEFAULT_OPT_PARAM, name="optimizer")
        self.learning_rate_param = update_params(
            learning_rate_scheduler, DEFAULT_LR_SCHEDULER_PARAM, name="learning_rate_scheduler"
        )
        self.init_param = update_params(init_param, DEFAULT_INIT_PARAM, name="init_param")
        self.threshold = threshold
        self.run_ovr = False
        self.train_feature_num = None
        self.validate_feature_num = None
        self.ovr = ovr
        self.label_num = label_num

        if self.ovr:
            if self.label_num is None or self.label_num < 2:
                raise ValueError(
                    "label_num must be greater than 2 when ovr is True, but got {}".format(self.label_num)
                )

        # models & optimizer & schduler
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.optimizer_state_dict = None
        self.trainer = None

        # loaded meta
        self.loaded_meta = None

        # reg
        self.l1 = 0
        self.l2 = 0

        # for testing
        self.local_mode = False

        # checkping param
        assert self.max_iter > 0 and isinstance(self.max_iter, int), "max_iter must be int greater than 0"
        if self.batch_size is not None:
            assert self.batch_size > 0 and isinstance(
                self.batch_size, int
            ), "batch_size must be int greater than 0 or None"
        assert self.threshold > 0 and self.threshold < 1, "threshold must be float between 0 and 1"

    def _make_dataset(self, data) -> TableDataset:
        ds = TableDataset(return_dict=True, to_tensor=True)
        ds.load(data)
        return ds

    def _make_output_df(self, ctx, predict_rs, data: TableDataset, threshold: float):
        classes = [i for i in range(len(self.model.models))]
        if len(classes) == 1:  # binary:
            classes = [0, 1]
        task_type = BINARY if len(classes) == 2 else MULTI

        out_df = array_to_predict_df(
            ctx,
            task_type,
            predict_rs.predictions,
            match_ids=data.get_match_ids(),
            sample_ids=data.get_sample_ids(),
            match_id_name=data.get_match_id_name(),
            sample_id_name=data.get_sample_id_name(),
            label=predict_rs.label_ids,
            threshold=threshold,
            classes=classes,
        )

        return out_df

    def _check_labels(self, label_set, has_validate=False):
        dataset_descrb = "train dataset" if not has_validate else "train and validate dataset"
        if not self.ovr and len(label_set) > 2:
            raise ValueError(
                "please set ovr=True to enable multi-label classification, multiple labels found in {}: {}".format(
                    dataset_descrb, label_set
                )
            )
        if not self.ovr and len(label_set) == 2:
            # 0, 1 is required
            if 0 not in label_set or 1 not in label_set:
                # ask for label 0, 1 when running binary classification
                raise ValueError(
                    "when doing binary classification, lables must be 0, 1, but found in {}'s label set is {}".format(
                        label_set, dataset_descrb
                    )
                )
        if self.ovr:
            if max(label_set) > self.label_num - 1:
                # make sure labels start from 0 and not the label indices not
                # exceed the label num parameter
                raise ValueError(
                    "when doing multi-label classification, labels must start from 0 and not exceed the label num parameter, \
                                 but {}'s label set is {}, while label num is {}".format(
                        label_set, dataset_descrb, self.label_num
                    )
                )

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        # check data, must be fate Dataframe
        assert isinstance(train_data, DataFrame), "train_data must be a fate DataFrame"
        if validate_data is not None:
            assert isinstance(validate_data, DataFrame), "validate_data must be a fate DataFrame"

        self.train_set = self._make_dataset(train_data)
        if not self.train_set.has_label():
            raise RuntimeError("train data must have label column")
        self.train_feature_num = self.train_set.features.shape[1]
        unique_label_set = set(self.train_set.get_classes())

        if validate_data is not None:
            self.validate_set = self._make_dataset(validate_data)
            if not self.validate_set.has_label():
                raise RuntimeError("validate data must have label column")
            self.validate_feature_num = self.validate_set.features.shape[1]
            assert (
                self.train_feature_num == self.validate_feature_num
            ), "train and validate feature num not match: {} vs {}".format(
                self.train_feature_num, self.validate_feature_num
            )
            unique_label_set = unique_label_set.union(set(self.validate_set.get_classes()))

        self._check_labels(unique_label_set, validate_data is not None)

        if self.batch_size is None:
            self.batch_size = len(self.train_set)

        # prepare loss function
        loss_fn = functools.partial(homo_lr_loss, dim=len(unique_label_set))
        optimizer_params = self.optimizer_param["optimizer_params"]
        opt_method = self.optimizer_param["method"]
        if self.optimizer_param["penalty"] == "l2":
            self.l2 = self.optimizer_param["alpha"]
            optimizer_params["weight_decay"] = self.l2
        elif self.optimizer_param["penalty"] == "l1":
            self.l1 = self.optimizer_param["alpha"]

        # initialize model
        if self.model is None:
            fit_intercept = self.init_param["fit_intercept"]
            self.model = HomoLRModel(
                self.train_feature_num, label_num=len(unique_label_set), l1=self.l1, bias=fit_intercept
            )
            # init model here
            init_model(self.model, method=self.init_param["method"], fill_val=self.init_param["fill_val"])
            logger.info("model initialized")
            logger.info("model parameters are {}".format(list(self.model.parameters())))
        else:
            logger.info("model is loaded, warm start training")
        logger.info("model structure is {}".format(self.model))

        self.optimizer = optimizer_factory(self.model.parameters(), opt_method, optimizer_params)
        self.lr_scheduler = lr_scheduler_factory(
            self.optimizer, self.learning_rate_param["method"], self.learning_rate_param["scheduler_params"]
        )

        if self.optimizer_state_dict is not None:
            optimizer_state_dict = {
                "state": {k: t.tensor(v) for k, v in self.optimizer_state_dict["state"].items()},
                "param_groups": self.optimizer_state_dict["param_groups"],
            }
            self.optimizer.load_state_dict(optimizer_state_dict)
            logger.info("load warmstart optimizer state dict")

        # training
        fed_arg = FedAVGArguments()
        train_arg = TrainingArguments(
            num_train_epochs=self.max_iter,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
        )
        self.trainer = FedAVGClient(
            ctx,
            model=self.model,
            loss_fn=loss_fn,
            optimizer=self.optimizer,
            train_set=self.train_set,
            val_set=self.validate_set,
            training_args=train_arg,
            fed_args=fed_arg,
            data_collator=default_data_collator,
            scheduler=self.lr_scheduler,
        )
        if self.local_mode:  # for debugging
            self.trainer.set_local_mode()
        self.trainer.train()

        logger.info("homo lr fit done")

    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        if self.model is None:
            raise ValueError("model is not initialized")
        self.predict_set = self._make_dataset(predict_data)
        if self.trainer is None:
            batch_size = len(self.predict_set) if self.batch_size is None else self.batch_size
            train_arg = TrainingArguments(num_train_epochs=self.max_iter, per_device_eval_batch_size=batch_size)
            trainer = FedAVGClient(
                ctx,
                train_set=self.predict_set,
                model=self.model,
                training_args=train_arg,
                fed_args=FedAVGArguments(),
                data_collator=default_data_collator,
            )
            trainer.set_local_mode()
        else:
            trainer = self.trainer
        predict_rs = trainer.predict(self.predict_set)
        predict_out_df = self._make_output_df(ctx, predict_rs, self.predict_set, self.threshold)
        return predict_out_df

    def get_model(self) -> ModelIO:
        param = {}
        if self.model is not None:
            param["model"] = self.model.to_dict()
        if self.optimizer is not None:
            param["optimizer"] = str(get_torch_bytes(self.optimizer.state_dict()))

        meta = {
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "threshold": self.threshold,
            "optimizer_param": self.optimizer_param,
            "learning_rate_param": self.learning_rate_param,
            "init_param": self.init_param,
            "ovr": self.ovr,
            "label_num": self.label_num,
        }

        return {"param": param, "meta": meta}

    def from_model(self, model: dict):
        if "param" not in model:
            raise ('key "data" is not found in the input model dict')

        model_param = model["param"]
        if "model" not in model_param:
            raise ValueError("param dict must have key 'model' that contains the model parameter and structure info")
        self.model = HomoLRModel.from_dict(model_param["model"])
        if self.ovr:
            assert len(self.model.models) == self.label_num, ""
        self.model.l1 = self.l1
        if hasattr(model_param, "optimizer"):
            self.optimizer_state_dict = recover_torch_bytes(bytes(model_param["optimizer"], "utf-8"))
        self.loaded_meta = model["meta"]
