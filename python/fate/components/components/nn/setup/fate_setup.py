from fate.components.components.nn.nn_setup import NNSetup
from transformers import TrainingArguments
from fate.ml.nn.homo.algo.fedavg import FedAVG, FedAVGArguments, FedAVGCLient, FedAVGServer
from typing import Optional, Dict
from fate.components.components.nn.loader import Loader

SUPPORTED_ALGO = ['fedavg']

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
                 use_hf_default_behavior: bool = False
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

    def _loader_load_from_conf(self, conf):
        if conf is None:
            return None
        return Loader.from_json(conf).load_inst()

    def setup(self):

        algo_class = None
        if self.algo == 'fedavg':
            client_class: FedAVGCLient = FedAVG.client
            server_class: FedAVGServer = FedAVG.server

        ctx = self.get_context()
            
        if self.is_client():
            # load arguments, models, etc
            model = self._loader_load_from_conf(self.model_conf)
            dataset = self._loader_load_from_conf(self.dataset_conf)
            optimizer = self._loader_load_from_conf(self.optimizer_conf)
            loss = self._loader_load_from_conf(self.loss_conf)
            data_collator = self._loader_load_from_conf(self.data_collator_conf)
            training_args = TrainingArguments(**self.training_args_conf)
            fed_args = FedAVGArguments(**self.fed_args_conf)

        elif self.is_server():
            trainer = server_class(ctx=ctx)
