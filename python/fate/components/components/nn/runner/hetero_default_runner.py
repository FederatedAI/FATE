import logging
from fate.components.components.nn.loader import Loader
from typing import Optional, Union, Dict, Literal
from fate.arch.dataframe import DataFrame
from fate.components.components.nn.nn_runner import (NNRunner, loader_load_from_conf, load_model_dict_from_path,
                                                     dir_warning)
from transformers.trainer_utils import get_last_checkpoint
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost


logger = logging.getLogger(__file__)

class DefaultRunner(NNRunner):

    def __init__(self,
                 bottom_model_conf: Optional[Dict] = None,
                 agg_layer_conf: Optional[Dict] = None,
                 top_model_conf: Optional[Dict] = None,
                 dataset_conf: Optional[Dict] = None,
                 optimizer_conf: Optional[Dict] = None,
                 training_args_conf: Optional[Dict] = None,
                 loss_conf: Optional[Dict] = None,
                 data_collator_conf: Optional[Dict] = None,
                 tokenizer_conf: Optional[Dict] = None,
                 task_type: Literal['binary',
                                    'multi',
                                    'regression',
                                    'others'] = 'binary',
                 threshold: float = 0.5
                 ):

        super().__init__()

        self.bottom_model_conf = bottom_model_conf
        self.interactive_layer_conf = agg_layer_conf
        self.top_model_conf = top_model_conf
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

    def guest_setup(self,
                    train_set=None,
                    validate_set=None,
                    output_dir=None,
                    saved_model=None
                    ):


        # load bottom model
        b_model = loader_load_from_conf(self.bottom_model_conf)
        # load agg layer
        agg_layer = loader_load_from_conf(self.interactive_layer_conf)
        # load top model
        t_model = loader_load_from_conf(self.top_model_conf)

        if b_model is None:
            logger.info('guest side bottom model is None')

        model = HeteroNNModelGuest(
            top_model=t_model,
            agg_layer=agg_layer,
            bottom_model=b_model
        )

        output_dir = './' if output_dir is None else output_dir
        resume_path = None
        if saved_model is not None:
            model_dict = load_model_dict_from_path(saved_model)
            model.load_state_dict(model_dict)
            logger.info(f"loading model dict from {saved_model} to model done")
            if get_last_checkpoint(saved_model) is not None:
                resume_path = saved_model
                logger.info(
                    f"checkpoint detected, resume_path set to {resume_path}")

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

        model.set_context(self.get_context())
        trainer = HeteroNNTrainerGuest(
            ctx=self.get_context(),
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            val_set=validate_set,
            training_args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            loss_fn=loss
        )

    def host_setup(self):
        pass


    def train(self,
              train_data: Optional[Union[str,
                                         DataFrame]] = None,
              validate_data: Optional[Union[str,
                                            DataFrame]] = None,
              output_dir: str = None,
              saved_model_path: str = None) -> None:

        if self.is_guest():
            pass

        elif self.is_host():
            pass

    def predict(self,
                test_data: Optional[Union[str,
                                          DataFrame]] = None,
                output_dir: str = None,
                saved_model_path: str = None) -> DataFrame:
        pass

