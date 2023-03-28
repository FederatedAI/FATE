import torch as t
import transformers
from transformers.adapters import AutoAdapterModel
from federatedml.util import LOGGER
from transformers.adapters import _import_structure 
from transformers.configuration_utils import PretrainedConfig
from federatedml.util import LOGGER


AVAILABLE_CONFIG = [i for i in _import_structure['configuration'] if i.endswith('Config')]


class PELLM(t.nn.Module):

    config_class: PretrainedConfig = None
    
    def __init__(self, config: dict=None, 
                 pretrained_path: str=None,
                 adapter_type: str=None,
                 adapter_config: dict=None,
                 ) -> None:
        
        super().__init__()

        self._pe_lm: AutoAdapterModel = None
        self.config = config
        self.config_path = pretrained_path
        self._adapter_name = 'federation'
        self.adapter_type = adapter_type
        self.adapter_config = adapter_config
        assert self.adapter_type is not None, "The adapter_type must be set.Available adapter_type: {}".format(AVAILABLE_CONFIG)
        assert self.config_path is not None or self.config is not None, "At least one of config_path and config must be set."
        self._init_pelm()

    def _init_pelm(self, ):
        
        if self.config_path is not None:
            LOGGER.info('Prioritize reading from the pretrained model path. Loading pretrained model from {}'.format(self.config_path))
            self._pe_lm = AutoAdapterModel.from_pretrained(self.config_path)
        elif self.config is not None:
            LOGGER.info('Loading model from model config dict')
            self._pe_lm = AutoAdapterModel.from_config(self.config_class().from_dict(self.config))
        else:
            raise ValueError('config_path to pretrained model folder and model config dict cannot be None at the same time, you need to specify one of them')

        self._pe_lm.freeze_model()
        # get adapter config
        assert self.adapter_type in AVAILABLE_CONFIG, 'adapter name {} not in availabe config {}'.format(self.adapter_type, AVAILABLE_CONFIG)
        # initialize adapter
        if self.adapter_config is None:
            config = getattr(transformers, self.adapter_type)()
        else:
            config = getattr(transformers, self.adapter_type)(**self.adapter_config)
        self._pe_lm.add_adapter(self._adapter_name, config)
        self._pe_lm.train_adapter(self._adapter_name)

        self.model_summary()

    def model_summary(self):
        summary = self._pe_lm.adapter_summary()
        LOGGER.debug('PELM model summary: \n{}'.format(summary))

    def _get_trainable_parameters(self):

        trainable = []
        for n, p in self._pe_lm.named_parameters():
            if p.requires_grad:
                trainable.append(p)
        return trainable
    
    def forward(self, tokenized_data: dict):
        return self._pe_lm(**tokenized_data)
    
    
class AutoPELLM(PELLM):

    def __init__(self, pretrained_path, adapter_type, adapter_config=None) -> None:
        super().__init__(pretrained_path=pretrained_path, adapter_type=adapter_type, adapter_config=adapter_config)
    
