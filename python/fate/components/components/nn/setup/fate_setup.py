import torch as t
import os
from fate.components.components.nn.nn_setup import NNSetup, SetupReturn
from transformers import TrainingArguments
from fate.ml.nn.algo.homo.fedavg import FedAVG, FedAVGArguments, FedAVGCLient, FedAVGServer
from typing import Optional, Dict
from fate.components.components.nn.loader import Loader


SUPPORTED_ALGO = ['fedavg']


def load_model_dict_from_path(path):
    # Ensure that the path is a string
    assert isinstance(path, str), "Path must be a string"

    # Append the filename to the path
    model_path = os.path.join(path, 'pytorch_model.bin')

    # Check if the file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No 'pytorch_model.bin' file found at {model_path}")

    # Load the state dict from the specified path
    model_dict = t.load(model_path)

    return model_dict


class FateSetup(NNSetup):

    def __init__(self, 
                 algo: str = 'fedavg',
                 model_conf: Optional[Dict] = None, 
                 dataset_conf: Optional[Dict] = None,
                 optimizer_conf: Optional[Dict] = None,
                 training_args_conf: Optional[Dict] = None,
                 fed_args_conf: Optional[Dict] = None,
                 loss_conf: Optional[Dict] = None,
                 data_collator_conf: Optional[Dict] = None,
                 tokenizer_conf: Optional[Dict] = None,
                 use_hf_default_behavior: bool = False,
                 local_mode: bool = False
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
        self.use_hf_default_behavior = use_hf_default_behavior
        self.local_mode = local_mode
        self.tokenizer_conf = tokenizer_conf

    def _loader_load_from_conf(self, conf, return_class=False):
        if conf is None:
            return None
        if return_class:
            return Loader.from_dict(conf).load_item()
        return Loader.from_dict(conf).call_item()

    def setup(self, mode='train'):

        if mode == 'predict':
            self.local_mode = True

        if self.algo == 'fedavg':
            client_class: FedAVGCLient = FedAVG.client
            server_class: FedAVGServer = FedAVG.server

        ctx = self.get_context()
            
        if self.is_client():

            # load arguments, models, etc
            # prepare datatset
            dataset = self._loader_load_from_conf(self.dataset_conf)
            cpn_input = self.get_cpn_input_data()

            if hasattr(dataset, 'load'):
                dataset.load(cpn_input)
            else:
                raise ValueError(f"dataset {dataset} has no load() method")
            
            # load model
            model = self._loader_load_from_conf(self.model_conf)
            # if has provided model dict, load it from path
            input_model = self.get_cpn_input_model()
            if input_model is not None:
                path = input_model.get('model_output_path')
                model_dict = load_model_dict_from_path(path)
                model.load_state_dict(model_dict)

            # load optimizer
            optimizer_loader = Loader.from_dict(self.optimizer_conf)
            optimizer_ = optimizer_loader.load_item()
            optimizer_params = optimizer_loader.kwargs
            optimizer = optimizer_(model.parameters(), **optimizer_params)
            # load loss
            loss = self._loader_load_from_conf(self.loss_conf)
            # load collator func
            data_collator = self._loader_load_from_conf(self.data_collator_conf)
            # 
            tokenizer = self._loader_load_from_conf(self.tokenizer_conf)
            # args
            training_args = TrainingArguments(**self.training_args_conf)
            training_args.output_dir = './'  # reset to default, saving to arbitrary path is not allowed in NN component
            fed_args = FedAVGArguments(**self.fed_args_conf)
            # prepare trainer
            trainer = client_class(ctx=ctx, model=model, loss_fn=loss,
                                   optimizer=optimizer, training_args=training_args,
                                   fed_args=fed_args, data_collator=data_collator,
                                   tokenizer=tokenizer, train_set=dataset, local_mode=self.local_mode)
            
            return SetupReturn(trainer=trainer, model=model, optimizer=optimizer, loss=loss, 
                               train_args=training_args, fed_args=fed_args, data_collator=data_collator,
                               train_set=dataset, validate_set=dataset)

        elif self.is_server():
            trainer = server_class(ctx=ctx, local_mode=self.local_mode)
            return SetupReturn(trainer=trainer)

    def train_setup(self):
        return self.setup(mode='train')

    def predict_setup(self):
        return self.setup(mode='predict')
    