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

from fate.components.components.nn.nn_runner import (
    NNRunner,
    load_model_dict_from_path,
    dir_warning,
    loader_load_from_conf,
    run_dataset_func,
)
from fate.ml.nn.homo.fedavg import FedAVG, FedAVGArguments, FedAVGClient, FedAVGServer
from typing import Dict
from fate.components.components.nn.loader import Loader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from fate.ml.nn.trainer.trainer_base import FedArguments, TrainingArguments, HomoTrainerClient, HomoTrainerServer
from typing import Union, Type, Callable, Optional
from transformers.trainer_utils import get_last_checkpoint
from typing import Literal
import logging
from fate.components.components.utils import consts
from fate.ml.nn.dataset.table import TableDataset, Dataset
from fate.arch.dataframe import DataFrame


logger = logging.getLogger(__name__)


SUPPORTED_ALGO = ["fedavg"]


def _check_instances(
    trainer: Union[Type[HomoTrainerClient], Type[HomoTrainerServer]] = None,
    fed_args: FedArguments = None,
    model: nn.Module = None,
    optimizer: optim.Optimizer = None,
    loss: Callable = None,
    scheduler: _LRScheduler = None,
    train_args: TrainingArguments = None,
    data_collator: Callable = None,
) -> None:
    if trainer is not None and not (
        issubclass(type(trainer), HomoTrainerClient) or issubclass(type(trainer), HomoTrainerServer)
    ):
        raise TypeError(
            f"SetupReturn Error: trainer must be a subclass of either FedTrainerClient or FedTrainerServer but got {type(trainer)}"
        )

    if fed_args is not None and not isinstance(fed_args, FedArguments):
        raise TypeError(f"SetupReturn Error: fed_args must be an instance of FedArguments but got {type(fed_args)}")

    if model is not None and not issubclass(type(model), nn.Module):
        raise TypeError(f"SetupReturn Error: model must be a subclass of torch.nn.Module but got {type(model)}")

    if optimizer is not None and not issubclass(type(optimizer), optim.Optimizer):
        raise TypeError(
            f"SetupReturn Error: optimizer must be a subclass of torch.optim.Optimizer but got {type(optimizer)}"
        )

    if loss is not None and not callable(loss):
        raise TypeError(f"SetupReturn Error: loss must be callable but got {type(loss)}")

    if scheduler is not None and not issubclass(type(scheduler), _LRScheduler):
        raise TypeError(
            f"SetupReturn Error: scheduler must be a subclass of torch.optim.lr_scheduler._LRScheduler but got {type(scheduler)}"
        )

    if train_args is not None and not isinstance(train_args, TrainingArguments):
        raise TypeError(
            f"SetupReturn Error: train_args must be an instance of TrainingArguments but got {type(train_args)}"
        )

    if data_collator is not None and not callable(data_collator):
        raise TypeError(f"SetupReturn Error: data_collator must be callable but got {type(data_collator)}")


class DefaultRunner(NNRunner):
    def __init__(
        self,
        algo: str = "fedavg",
        model_conf: Optional[Dict] = None,
        dataset_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        fed_args_conf: Optional[Dict] = None,
        loss_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        task_type: Literal["binary", "multi", "regression", "causal_lm", "others"] = "binary",
        threshold: float = 0.5,
        local_mode: bool = False,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.model_conf = model_conf
        self.dataset_conf = dataset_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.fed_args_conf = fed_args_conf
        self.loss_conf = loss_conf
        self.data_collator_conf = data_collator_conf
        self.local_mode = local_mode
        self.tokenizer_conf = tokenizer_conf
        self.task_type = task_type
        self.threshold = threshold

        # check param
        if self.algo not in SUPPORTED_ALGO:
            raise ValueError("algo should be one of [fedavg]")
        if self.task_type not in ["binary", "multi", "regression", "causal_lm", "others"]:
            raise ValueError("task_type should be one of [binary, multi, regression, others]")
        assert self.threshold >= 0 and self.threshold <= 1, "threshold should be in [0, 1]"
        assert isinstance(self.local_mode, bool), "local should be bool"

        # setup var
        self.trainer = None
        self.training_args = None

    def _prepare_data(self, data, data_name):
        if data is None:
            return None
        if isinstance(data, DataFrame) and self.dataset_conf is None:
            logger.info(
                "Input data {} is FATE DataFrame and dataset conf is None, will automatically handle the input data".format(
                    data_name
                )
            )
            if self.task_type == consts.MULTI:
                dataset = TableDataset(flatten_label=True, label_dtype="long", to_tensor=True)
            else:
                dataset = TableDataset(to_tensor=True)
            dataset.load(data)
        else:
            dataset = loader_load_from_conf(self.dataset_conf)
            if hasattr(dataset, "load"):
                logger.info("load path is {}".format(data))
                load_output = dataset.load(data)
                if load_output is not None:
                    dataset = load_output
                    return dataset
            else:
                raise ValueError(
                    f"The dataset {dataset} lacks a load() method, which is required for data parsing in the DefaultRunner. \
                                Please implement this method in your dataset class. You can refer to the base class 'Dataset' in 'fate.ml.nn.dataset.base' \
                                for the necessary interfaces to implement."
                )
        if dataset is not None and not issubclass(type(dataset), Dataset):
            raise TypeError(
                f"SetupReturn Error: {data_name}_set must be a subclass of fate built-in Dataset but got {type(dataset)}, \n"
                f"You can get the class via: from fate.ml.nn.dataset.table import Dataset"
            )

        return dataset

    def client_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None, stage="train"):
        if stage == "predict":
            self.local_mode = True

        if self.algo == "fedavg":
            client_class: FedAVGClient = FedAVG.client
        else:
            raise ValueError(f"algo {self.algo} not supported")

        ctx = self.get_context()
        model = loader_load_from_conf(self.model_conf)
        if model is None:
            raise ValueError(f"model is None, cannot load model from conf {self.model_conf}")

        if output_dir is None:
            output_dir = "./"

        resume_path = None
        if saved_model is not None:
            model_dict = load_model_dict_from_path(saved_model)
            model.load_state_dict(model_dict)
            logger.info(f"loading model dict from {saved_model} to model done")
            if get_last_checkpoint(saved_model) is not None:
                resume_path = saved_model
                logger.info(f"checkpoint detected, resume_path set to {resume_path}")
        # load optimizer
        if self.optimizer_conf:
            optimizer_loader = Loader.from_dict(self.optimizer_conf)
            optimizer_ = optimizer_loader.load_item()
            optimizer_params = optimizer_loader.kwargs
            optimizer = optimizer_(model.parameters(), **optimizer_params)
        else:
            optimizer = None
        # load loss
        loss = loader_load_from_conf(self.loss_conf) if self.loss_conf else None
        # load collator func
        data_collator = loader_load_from_conf(self.data_collator_conf)
        # load tokenizer if import conf provided
        tokenizer = loader_load_from_conf(self.tokenizer_conf)
        # args
        dir_warning(self.training_args_conf)
        training_args = TrainingArguments(**self.training_args_conf)
        self.training_args = training_args
        # reset to default, saving to arbitrary path is not allowed in
        # DefaultRunner
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path
        fed_args = FedAVGArguments(**self.fed_args_conf)
        if fed_args.aggregate_strategy == 'steps':
            raise ValueError('aggregate_strategy "steps" is not supported in FATE-pipeline which will be used in production.')
        # prepare trainer
        trainer = client_class(
            ctx=ctx,
            model=model,
            loss_fn=loss,
            optimizer=optimizer,
            training_args=training_args,
            fed_args=fed_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_set=train_set,
            val_set=validate_set,
            local_mode=self.local_mode,
        )

        _check_instances(
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            loss=loss,
            train_args=training_args,
            fed_args=fed_args,
            data_collator=data_collator,
        )
        return trainer

    def server_setup(self, stage="train"):
        if stage == "predict":
            self.local_mode = True
        if self.algo == "fedavg":
            server_class: FedAVGServer = FedAVG.server
        else:
            raise ValueError(f"algo {self.algo} not supported")
        ctx = self.get_context()
        trainer = server_class(ctx=ctx, local_mode=self.local_mode)
        _check_instances(trainer)
        return trainer

    def train(
        self,
        train_data: Optional[Union[str, DataFrame]] = None,
        validate_data: Optional[Union[str, DataFrame]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ):
        if self.is_client():
            train_set = self._prepare_data(train_data, "train_data")
            validate_set = self._prepare_data(validate_data, "val_data")
            trainer = self.client_setup(
                train_set=train_set, validate_set=validate_set, output_dir=output_dir, saved_model=saved_model_path
            )
            self.trainer = trainer
            trainer.train()
            if output_dir is not None:
                if self.training_args.deepspeed and self.training_args.local_rank != 0:
                    pass
                else:
                    trainer.save_model(output_dir)
        elif self.is_server():
            trainer = self.server_setup()
            trainer.train()

    def predict(self, test_data: Union[str, DataFrame], saved_model_path: str = None) -> Union[DataFrame, None]:
        
        if self.is_client():
            test_set = self._prepare_data(test_data, "test_data")
            if self.trainer is not None:
                trainer = self.trainer
                logger.info("trainer found, skip setting up")
            else:
                trainer = self.client_setup(saved_model=saved_model_path, stage="predict")

            classes = run_dataset_func(test_set, "get_classes")
            match_ids = run_dataset_func(test_set, "get_match_ids")
            sample_ids = run_dataset_func(test_set, "get_sample_ids")
            match_id_name = run_dataset_func(test_set, "get_match_id_name")
            sample_id_name = run_dataset_func(test_set, "get_sample_id_name")
            pred_rs = trainer.predict(test_set)

            if self.training_args and self.training_args.deepspeed and self.training_args.local_rank != 0:
                return

            rs_df = self.get_nn_output_dataframe(
                self.get_context(),
                pred_rs.predictions,
                pred_rs.label_ids if hasattr(pred_rs, "label_ids") else None,
                match_ids,
                sample_ids,
                match_id_name=match_id_name,
                sample_id_name=sample_id_name,
                dataframe_format="dist_df",
                task_type=self.task_type,
                classes=classes,
            )
            return rs_df
        else:
            # server not predict
            return
