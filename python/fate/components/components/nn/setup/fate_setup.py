from fate.components.components.nn.nn_setup import NNSetup
from transformers import TrainingArguments
from fate.ml.nn.homo.algo.fedavg import FedAVG, FedAVGArguments
from typing import Optional, Dict

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

    def setup(self):

        algo_class = None
        if self.algo == 'fedavg':
            algo_class = FedAVG
            
        if self.is_client():
            pass
        elif self.is_server():
            pass
