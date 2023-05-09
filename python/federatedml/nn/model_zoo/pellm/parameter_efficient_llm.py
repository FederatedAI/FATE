import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig
from transformers import AutoModel
from transformers.adapters import AutoAdapterModel
from transformers.adapters import _import_structure
from transformers.configuration_utils import PretrainedConfig
from typing import Union
from federatedml.util import LOGGER


AVAILABLE_ADAPTERS_CONFIG = list(
    filter(
        lambda adapter_config: adapter_config.endswith('Config'), _import_structure['configuration']
    )
)
AVAILABLE_PEFT_CONFIG = list(
    filter(
        lambda peft_type: peft_type.endswith("Config"), dir(peft)
    )
)


class PELLM(torch.nn.Module):

    config_class: PretrainedConfig = None
    enable_save_pretrained: bool = False

    def __init__(self, config: dict = None,
                 pretrained_path: str = None,
                 adapter_type: str = None,
                 adapter_config: dict = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 **kwargs
                 ) -> None:

        super().__init__()
        self._pe_lm: Union[AutoAdapterModel, PeftModel] = None
        self.config = config
        self.config_path = pretrained_path
        self._adapter_name = 'federation'
        self.adapter_type = adapter_type
        self.adapter_config = adapter_config
        self.peft_type = peft_type
        self.peft_config = peft_config

        if self.adapter_type is not None:
            assert self.adapter_type in AVAILABLE_ADAPTERS_CONFIG, \
                "The adapter_type must be set available adapter_type: {}".format(AVAILABLE_ADAPTERS_CONFIG)
        if peft_type is not None:
            assert self.peft_type in AVAILABLE_PEFT_CONFIG, \
                'The peft_type must be set available peft_type: {}'.format(AVAILABLE_PEFT_CONFIG)

        assert self.config_path is not None or self.config is not None, \
            "At least one of config_path and config must be set."
        self._init_pelm()

    def _init_pelm(self):
        if self.adapter_type is not None:
            self.init_lm_with_adapter()
        else:
            self.init_lm_with_peft()

    def init_lm_with_adapter(self):
        if self.config_path is not None:
            LOGGER.info(
                'Prioritize reading from the pretrained model path. Loading pretrained model from {}'.format(
                    self.config_path))
            self._pe_lm = AutoAdapterModel.from_pretrained(self.config_path)
        elif self.config is not None:
            LOGGER.info('Loading model from model config dict')
            self._pe_lm = AutoAdapterModel.from_config(self.config_class().from_dict(self.config))
        else:
            raise ValueError(
                'config_path to pretrained model folder and model config dict cannot be None at the same time, '
                'you need to specify one of them')

        self._pe_lm.freeze_model()
        # get adapter config
        assert self.adapter_type in AVAILABLE_ADAPTERS_CONFIG, 'adapter name {} not in availabe config {}'.format(
            self.adapter_type, AVAILABLE_ADAPTERS_CONFIG)
        # initialize adapter
        if self.adapter_config is None:
            config = getattr(transformers, self.adapter_type)()
        else:
            config = getattr(transformers, self.adapter_type)().from_dict(self.adapter_config)
        self._pe_lm.add_adapter(self._adapter_name, config)
        self._pe_lm.train_adapter(self._adapter_name)

        self.model_summary()

    def init_lm_with_peft(self):
        self.init_config()
        self.init_base_lm()
        self.add_peft()

    def init_config(self):
        if self.config_path is not None:
            self.config = AutoConfig.from_pretrained(self.config_path)
        elif self.config is not None and self.config_class is not None:
            self.config = self.config_class().from_dict(self.config)
        else:
            raise ValueError(
                'config_path to pretrained model folder and model config dict cannot be None at the same time, '
                'you need to specify one of them')

    def init_base_lm(self, **kwargs):
        if self.config is not None:
            self._pe_lm = AutoModel.from_pretrained(self.config_path, config=self.config, **kwargs)
        elif self.config_path is not None:
            self._pe_lm = AutoModel.from_pretrained(self.config_path, **kwargs)
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')

    def add_peft(self):
        assert self.peft_type in AVAILABLE_PEFT_CONFIG, 'peft name {} not in availabe config {}'.format(
            self.peft_type, AVAILABLE_PEFT_CONFIG)

        if self.peft_config is None:
            peft_config = getattr(peft, self.peft_type)()
        else:
            peft_config = getattr(peft, self.peft_type)(**self.peft_config)

        self._pe_lm = peft.get_peft_model(self._pe_lm, peft_config)

    def model_summary(self):
        try:
            summary = self._pe_lm.adapter_summary()
        except AttributeError:
            summary = self._pe_lm.print_trainable_parameters()

        LOGGER.debug('PELM model summary: \n{}'.format(summary))

    def _get_trainable_parameters(self):
        trainable = []
        for n, p in self._pe_lm.named_parameters():
            if p.requires_grad:
                trainable.append(p)
        return trainable

    def forward(self, tokenized_data: dict):
        return self._pe_lm(**tokenized_data)

    def save_pretrained(self, path):
        if not self.enable_save_pretrained:
            raise ValueError("To save trainable parameters only, set enable_save_pretrained=True in your model")

        from pathlib import Path

        state_dict = {
            k: p.to("cpu") for k, p in self._pe_lm.named_parameters() if p.requires_grad
        }
        Path.mkdir(Path(path), exist_ok=True)
        torch.save(state_dict, Path(path).joinpath("adapter_model.bin"))


class AutoPELLM(PELLM):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
