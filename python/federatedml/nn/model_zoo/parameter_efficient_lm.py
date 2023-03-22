import torch as t
import transformers
from transformers.adapters import AutoAdapterModel
from federatedml.util import LOGGER
from transformers.adapters import _import_structure 

AVAILABLE_CONFIG = [i for i in _import_structure['configuration'] if i.endswith('Config')]


class FedPELanguageModel(t.nn.Module):

    def __init__(self, model_loadpath, adapter_name, adapter_config=None) -> None:
        super(FedPELanguageModel, self).__init__()
        self._pe_lm: AutoAdapterModel = None
        self._model_loadpath = model_loadpath
        self._pe_lm = AutoAdapterModel.from_pretrained(self._model_loadpath)
        self._pe_lm.freeze_model()
        self._adapter_name = 'federation'
        # get adapter config
        assert adapter_name in AVAILABLE_CONFIG, 'adapter name {} not in availabe config {}'.format(adapter_name, AVAILABLE_CONFIG)
        # initialize adapter
        if adapter_config is None:
            config = getattr(transformers, adapter_name)()
        else:
            config = getattr(transformers, adapter_name)(**adapter_config)
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
    

if __name__ == '__main__':
    pass