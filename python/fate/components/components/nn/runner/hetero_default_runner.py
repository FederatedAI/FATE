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

import logging
from fate.components.components.nn.loader import Loader
from typing import Optional, Union, Dict, Literal
from fate.arch.dataframe import DataFrame
from fate.components.components.nn.nn_runner import (
    NNRunner,
    loader_load_from_conf,
    load_model_dict_from_path,
    dir_warning,
    run_dataset_func,
)
from transformers.trainer_utils import get_last_checkpoint
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.components.components.utils import consts
from fate.ml.nn.dataset.table import TableDataset, Dataset
from fate.ml.nn.model_zoo.hetero_nn_model import (
    FedPassArgument,
    StdAggLayerArgument,
    parse_agglayer_conf,
    TopModelStrategyArguments,
)


logger = logging.getLogger(__file__)


class DefaultRunner(NNRunner):
    def __init__(
        self,
        bottom_model_conf: Optional[Dict] = None,
        top_model_conf: Optional[Dict] = None,
        agglayer_arg_conf: Optional[Dict] = None,
        top_model_strategy_arg_conf: Optional[Dict] = None,
        dataset_conf: Optional[Dict] = None,
        optimizer_conf: Optional[Dict] = None,
        training_args_conf: Optional[Dict] = None,
        loss_conf: Optional[Dict] = None,
        data_collator_conf: Optional[Dict] = None,
        tokenizer_conf: Optional[Dict] = None,
        task_type: Literal["binary", "multi", "regression", "others"] = "binary",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.bottom_model_conf = bottom_model_conf
        self.top_model_conf = top_model_conf
        self.agglayer_arg_conf = agglayer_arg_conf
        self.top_model_strategy_arg_conf = top_model_strategy_arg_conf
        self.dataset_conf = dataset_conf
        self.optimizer_conf = optimizer_conf
        self.training_args_conf = training_args_conf
        self.loss_conf = loss_conf
        self.data_collator_conf = data_collator_conf
        self.tokenizer_conf = tokenizer_conf
        self.task_type = task_type
        self.threshold = threshold

        # setup var
        self.trainer = None
        self.model = None

    def _prepare_data(self, data, data_name):
        if data is None:
            return data

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
                dataset.load(data)
            else:
                raise ValueError(
                    f"The dataset {dataset} lacks a load() method, which is required for data parsing in the DefaultRunner.Please implement this method in your dataset class. You can refer to the base class 'Dataset' in 'fate.ml.nn.dataset.base' \
for the necessary interfaces to implement."
                )
        if dataset is not None and not issubclass(type(dataset), Dataset):
            raise TypeError(
                f"SetupReturn Error: {data_name}_set must be a subclass of fate built-in Dataset but got {type(dataset)}, \n"
                f"You can get the class via: from fate.ml.nn.dataset.table import Dataset"
            )

        return dataset

    def _setup(self, model, output_dir=None, saved_model=None):
        output_dir = "./" if output_dir is None else output_dir
        logger.info("output dir is {}".format(output_dir))
        resume_path = None
        if saved_model is not None:
            model_dict = load_model_dict_from_path(saved_model)
            logger.info("model is {}".format(model))
            model.load_state_dict(model_dict)
            logger.info(f"loading model dict from {saved_model} to model done")
            if get_last_checkpoint(saved_model) is not None:
                resume_path = saved_model
                logger.info(f"checkpoint detected, resume_path set to {resume_path}")

        # load optimizer
        optimizer_loader = Loader.from_dict(self.optimizer_conf)
        optimizer_ = optimizer_loader.load_item()
        optimizer_params = optimizer_loader.kwargs
        optimizer = optimizer_(model.parameters(), **optimizer_params)
        # load loss
        loss = loader_load_from_conf(self.loss_conf)
        # load collator func
        data_collator = loader_load_from_conf(self.data_collator_conf)
        # load tokenizer if import conf provided
        tokenizer = loader_load_from_conf(self.tokenizer_conf)
        # args
        dir_warning(self.training_args_conf)
        training_args = TrainingArguments(**self.training_args_conf)
        training_args.output_dir = output_dir
        training_args.resume_from_checkpoint = resume_path  # resume path

        return optimizer, loss, data_collator, tokenizer, training_args

    def guest_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        # load bottom model
        b_model = loader_load_from_conf(self.bottom_model_conf)
        # load top model
        t_model = loader_load_from_conf(self.top_model_conf)

        agglayer_arg = None
        if self.agglayer_arg_conf is not None:
            agglayer_arg = parse_agglayer_conf(self.agglayer_arg_conf)

        top_model_strategy = None
        if self.top_model_strategy_arg_conf is not None:
            top_model_strategy = TopModelStrategyArguments(**self.top_model_strategy_arg_conf)

        if b_model is None:
            logger.info("guest side bottom model is None")

        model = HeteroNNModelGuest(
            top_model=t_model, bottom_model=b_model, agglayer_arg=agglayer_arg, top_arg=top_model_strategy
        )
        logger.info("model initialized, model is {}.".format(model))
        optimizer, loss, data_collator, tokenizer, training_args = self._setup(model, output_dir, saved_model)
        trainer = HeteroNNTrainerGuest(
            ctx=self.get_context(),
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            val_set=validate_set,
            training_args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            loss_fn=loss,
        )

        return trainer, model

    def host_setup(self, train_set=None, validate_set=None, output_dir=None, saved_model=None):
        # load bottom model
        b_model = loader_load_from_conf(self.bottom_model_conf)
        if self.agglayer_arg_conf is not None:
            logger.info(self.agglayer_arg_conf)
            agglayer_arg = parse_agglayer_conf(self.agglayer_arg_conf)
            logger.info(agglayer_arg)
            if type(agglayer_arg) == StdAggLayerArgument:
                raise ValueError("Plaintext agglayer is not supported in Hetero-NN Pipeline Host party")
        else:
            raise ValueError(
                "A aggregate layer for privacy preserving is needed in the Hetero-NN pipeline Host party, "
                "please set the agglayer config: use fedpass alone in host, or configure sshe layers for guest&host"
            )

        model = HeteroNNModelHost(bottom_model=b_model, agglayer_arg=agglayer_arg)
        logger.info("model initialized, model is {}.".format(model))
        optimizer, loss, data_collator, tokenizer, training_args = self._setup(model, output_dir, saved_model)
        trainer = HeteroNNTrainerHost(
            ctx=self.get_context(),
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            val_set=validate_set,
            training_args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        return trainer, model

    def _check_label(self, dataset):
        if not hasattr(dataset, "has_label"):
            raise ValueError(
                "dataset has no has_label func, please make sure that your"
                " class subclasses fate built-in Dataset class"
            )

        has_label = dataset.has_label()
        if has_label is None or has_label == False:
            raise ValueError(
                "label is required on the guest side for hetero training, has label return False or None\n"
                "- Please check your input data.\n"
                "- Please make sure the has_label func is implemented correctly in your dataset class."
            )

    def train(
        self,
        train_data: Optional[Union[str, DataFrame]] = None,
        validate_data: Optional[Union[str, DataFrame]] = None,
        output_dir: str = None,
        saved_model_path: str = None,
    ) -> None:
        train_set = self._prepare_data(train_data, "train_data")
        validate_set = self._prepare_data(validate_data, "val_data")

        if self.is_guest():
            self._check_label(train_set)
            if validate_set is not None:
                self._check_label(validate_set)
            trainer, model = self.guest_setup(train_set, validate_set, output_dir, saved_model_path)
        elif self.is_host():
            trainer, model = self.host_setup(train_set, validate_set, output_dir, saved_model_path)
        else:
            raise RuntimeError("invalid role in hetero nn")

        self.trainer, self.model = trainer, model
        self.trainer.train()
        if output_dir is not None:
            self.trainer.save_model(output_dir)

    def predict(
        self, test_data: Optional[Union[str, DataFrame]] = None, output_dir: str = None, saved_model_path: str = None
    ) -> DataFrame:
        test_set = self._prepare_data(test_data, "test_data")
        if self.trainer is not None:
            trainer = self.trainer
            logger.info("trainer found, skip setting up")
        else:
            if self.is_guest():
                trainer = self.guest_setup(output_dir=output_dir, saved_model=saved_model_path)[0]
            else:
                trainer = self.host_setup(output_dir=output_dir, saved_model=saved_model_path)[0]

        if self.is_guest():
            classes = run_dataset_func(test_set, "get_classes")
            match_ids = run_dataset_func(test_set, "get_match_ids")
            sample_ids = run_dataset_func(test_set, "get_sample_ids")
            match_id_name = run_dataset_func(test_set, "get_match_id_name")
            sample_id_name = run_dataset_func(test_set, "get_sample_id_name")

            pred_rs = trainer.predict(test_set)

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

        elif self.is_host():
            trainer.predict(test_set)
